import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

# SATO: Spatially-Aware Transformer Operator
# 100% Mirror implementation of SATO/models/SATO.py with vectorized optimizations.

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class Physics_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.record_attention = False

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        if getattr(self, "record_attention", False):
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
        out_slice_token = torch.matmul(attn, v)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class Serialized_Attention(nn.Module):
    def __init__(self, patch_size, shift, dim, num_heads, dropout=0.1):
        super(Serialized_Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.patch_size = patch_size
        self.shift = shift
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Precompute group index for shifting
        index = torch.tensor([i for i in range(0, patch_size*shift, shift)], dtype=torch.int64)[None, ...]
        self.register_buffer('group_index', torch.cat([index+i for i in range(shift)], dim=0))
        
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, quadrature_weights=None):
        """Patch attention over serialized sequence.

        quadrature_weights: optional [B, N] node areas (already in serialization order).
        When provided, patch Softmax becomes area-weighted:
            attn ∝ Softmax(dots) * q  (renormalize; q=0 kills pad / invalid keys).
        Equivalent to Softmax(dots + log q) when q>0 everywhere in the patch.
        """
        B, N, C = x.shape
        
        # padding
        pad_size = int((self.patch_size*self.shift) - N % (self.patch_size*self.shift))
        if pad_size == (self.patch_size*self.shift): pad_size = 0 # Handle exact multiple
        
        if pad_size > 0:
            x_pad = torch.cat([x, torch.zeros(B, pad_size, C, device=x.device)], dim=1)
        else:
            x_pad = x

        q_pad = None
        if quadrature_weights is not None:
            q = quadrature_weights
            if not torch.is_tensor(q):
                q = torch.as_tensor(q, device=x.device, dtype=x.dtype)
            else:
                q = q.to(device=x.device, dtype=x.dtype)
            if q.dim() == 1:
                q = q.unsqueeze(0).expand(B, -1)
            if q.shape != (B, N):
                raise ValueError(
                    f"Serialized_Attention quadrature_weights shape {tuple(q.shape)} != {(B, N)}"
                )
            if pad_size > 0:
                q_pad = torch.cat([q, torch.zeros(B, pad_size, device=x.device, dtype=x.dtype)], dim=1)
            else:
                q_pad = q
            
        N_padded = x_pad.shape[1]

        # Optimized index generation for vectorized batch processing
        # Note: Original SATO uses a while loop to generate patch_index. 
        # Here we use a more efficient vectorized approach while maintaining the same logic.
        num_patch_groups = N_padded // (self.patch_size * self.shift)
        offsets = torch.arange(num_patch_groups, device=x.device) * (self.patch_size * self.shift)
        patch_index = (self.group_index.unsqueeze(0) + offsets.view(-1, 1, 1)).reshape(-1, self.patch_size)
        
        # pad2patch
        x_patch = x_pad[:, patch_index, :] # (B, num_patches, patch_size, C)
        
        # patch attention
        B_orig = B
        x_patch = rearrange(x_patch, 'b n s c -> (b n) s c')
        B_p, S_p, C_p = x_patch.shape
        
        qkv = self.qkv_proj(x_patch).reshape(B_p, S_p, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        if q_pad is not None:
            # Area-weighted Softmax: attn ∝ Softmax(dots) * q  (same idea as Physics slice * q)
            # Pad / invalid nodes have q=0; all-zero patch → attn≈0 (safe via +eps).
            q_patch = q_pad[:, patch_index]  # (B, num_patches, patch_size)
            q_patch = rearrange(q_patch, 'b n s -> (b n) s')
            attn = attn * q_patch[:, None, None, :]
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)
        attn = self.dropout(attn)
        out_token = torch.matmul(attn, v)
        
        out_token = rearrange(out_token, 'bn h s d -> bn s (h d)')
        out_token = self.out_proj(out_token)
        out_token = rearrange(out_token, '(b n) s c -> b n s c', b=B_orig)
        
        # patch2pad (Scatter back)
        # We need to be careful with in-place updates if we want to avoid issues with gradients
        # but original SATO does x_pad[:, patch_index, :] = out_token
        res = torch.zeros_like(x_pad)
        # Using advanced indexing for scatter-like behavior
        batch_idx = torch.arange(B_orig, device=x.device).view(-1, 1, 1)
        res[batch_idx, patch_index.unsqueeze(0)] = out_token
        
        return res[:, :N, :]

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act_fn = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.res = res
        self.n_layers = n_layers
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class SATO_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4, last_layer=False, slice_num=32, patch_size=20, shift=1):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.global_attention = Physics_Attention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        self.local_ln_1 = nn.LayerNorm(hidden_dim)
        self.local_attention = Serialized_Attention(patch_size, shift, hidden_dim, num_heads, dropout=0.1)
        self.local_ln_2 = nn.LayerNorm(hidden_dim)
        self.local_gate = nn.Parameter(torch.tensor([0.0]))

        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        
        if self.last_layer:
            self.ln_4 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, fx, order, inverse):
        x0 = fx 
        fxn = self.ln_1(fx)

        # global_attention
        fx1 = self.global_attention(fxn)
        
        # local_attention
        B, N, C = fx.shape
        batch_indices = torch.arange(B, device=fx.device).unsqueeze(-1)
        
        # serialized: fx2 = (fxn - fx1)[order]
        fx2_input = fxn - fx1
        fx2 = fx2_input[batch_indices, order]
        
        fx2 = self.local_ln_2(self.local_attention(self.local_ln_1(fx2)))
        
        # deserialized
        fx2 = fx2[batch_indices, inverse]
        
        # Fusion: fx = self.mlp(self.ln_3(x0 + fx1 + self.local_gate * fx2)) + x0
        fx = self.mlp(self.ln_3(x0 + fx1 + self.local_gate * fx2)) + x0
        
        if self.last_layer:
            return self.mlp2(self.ln_4(fx))
        else:
            return fx

class Model(nn.Module):
    def __init__(self, space_dim=2, n_layers=3, n_hidden=64, dropout=0, n_head=1, act='gelu', mlp_ratio=4, fun_dim=14, slice_num=32, patch_size=32, shift=1, n_iter=1, out_dim=1, **kwargs):
        super(Model, self).__init__()
        self.__name__ = 'SATO'
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_iter = n_iter

        self.blocks = nn.ModuleList([
            SATO_block(num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act, mlp_ratio=mlp_ratio, slice_num=slice_num, patch_size=patch_size, shift=shift, last_layer=(_ == n_layers - 1))
            for _ in range(n_layers)
        ])
        
        # Note: Original SATO has a hardcoded mlp2 in the last block returning 1 dim.
        # We keep out_dim for flexibility but default to 1 for MultiNet compatibility.
        if out_dim != 1:
            # If out_dim > 1, we need to override the last layer's mlp2
            self.blocks[-1].mlp2 = nn.Linear(n_hidden, out_dim)

        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pos, fx=None, sato_indices=None, **kwargs):
        # pos: (B, N, D), fx: (B, N, F)
        if fx is not None:
            x = torch.cat([pos, fx], dim=-1)
        else:
            x = pos
            
        fx_surf = self.preprocess(x)
        fx_surf = fx_surf + self.placeholder[None, None, :]
        
        # Handle order/inverse from sato_indices dict
        if sato_indices is not None:
            order = sato_indices["order"]
            inverse = sato_indices["inverse"]
        else:
            # Fallback (though SATO needs these)
            B, N, _ = pos.shape
            order = torch.arange(N, device=pos.device).view(1, 1, N).repeat(B, 1, 1)
            inverse = order.clone()

        if self.n_iter == 1:
            for i in range(self.n_layers):
                serialization_type = int(i % order.shape[1])
                fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])
        else:
            for _ in range(self.n_iter):
                for i in range(self.n_layers-1):
                    serialization_type = int(i % order.shape[1])
                    fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])
                i = self.n_layers - 1
                serialization_type = int(i % order.shape[1])
                fx_surf = self.blocks[i](fx_surf, order[:, serialization_type, :], inverse[:, serialization_type, :])

        return fx_surf

    def enable_attention_recording(self, enabled=True):
        """Enable/disable attention recording for visualization."""
        for block in self.blocks:
            if hasattr(block, "global_attention") and hasattr(block.global_attention, "record_attention"):
                block.global_attention.record_attention = bool(enabled)
                if enabled:
                    if hasattr(block.global_attention, "last_attn"):
                        delattr(block.global_attention, "last_attn")
                    if hasattr(block.global_attention, "last_slice_weights"):
                        delattr(block.global_attention, "last_slice_weights")

    def get_last_attention(self):
        """Get attention weights and slice weights from the last forward pass."""
        results = []
        for layer_index, block in enumerate(self.blocks):
            if hasattr(block, "global_attention"):
                attn = getattr(block.global_attention, "last_attn", None)
                slice_weights = getattr(block.global_attention, "last_slice_weights", None)
                if (attn is not None) or (slice_weights is not None):
                    results.append({"layer": layer_index, "attn": attn, "slice_weights": slice_weights})
        return results
