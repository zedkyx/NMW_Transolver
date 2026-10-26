import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from .Embedding import timestep_embedding
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .Physics_Attention import (
    Physics_Attention_Irregular_Mesh,
    LinearNO_Attention_Irregular_Mesh,
    SpectralLayer,
    _broadcast_quadrature_weights,
)
from .sato import Serialized_Attention


def _geom_inject_policy(n_layers: int, policy: str) -> list[bool]:
    """Which Transolver blocks get FiLM (0-based; lift is never injected).

    Split at L/2 (float): early l < L/2, late l >= L/2, last = final block, all.
    L=4 → early 0,1; late 2,3; last 3; all 0–3.
    L=3 → early 0,1; late 2; last 2 (same as late); all 0–2.
    """
    L = int(n_layers)
    half = L / 2.0
    p = str(policy or "early").strip().lower()
    if p == "early":
        return [i < half for i in range(L)]
    if p == "late":
        return [i >= half for i in range(L)]
    if p == "last":
        return [i == L - 1 for i in range(L)]
    if p == "all":
        return [True] * L
    raise ValueError(f"geom_memory_policy must be early|late|last|all, got {policy!r}")


class PointwiseGeomEncoder(nn.Module):
    """G_i=[x,y,SDF] → m_i (shared across layers)."""

    def __init__(self, in_dim: int = 3, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class GeomFiLM(nn.Module):
    """Paper FiLM: V' = γ(m)⊙V + β(m).
    Init: γ→1, β→0 (identity); if γ,β both 0-init then V'≡0 and kills features.
    """

    def __init__(self, mem_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(mem_dim, hidden_dim)
        self.to_beta = nn.Linear(mem_dim, hidden_dim)
        # identity start under γ⊙V+β
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.to_gamma(m) * v + self.to_beta(m)


def _gumbel_softmax(logits, temperature, tau=1.0):
    """Gumbel-Softmax: 可微的离散采样近似，temperature 为 per-token (B H N 1)。"""
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = logits + gumbel_noise
    y = y / (temperature + 1e-8)
    return F.softmax(y, dim=-1)


class Physics_Attention_Irregular_EideticSlice(nn.Module):
    """
    Physics Attention for irregular mesh，Slice 分配采用 Transolver++ 的
    token-dependent 温度 + Gumbel-Softmax，其余（mask、spectral、record）与原版一致。
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, use_spectral=False,
                 spectral_modes=16, spectral_hi_freq_boost=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.record_attention = False
        self.use_spectral = use_spectral
        if self.use_spectral:
            self.spectral_layer = SpectralLayer(dim_head, spectral_modes, heads, hi_freq_boost=spectral_hi_freq_boost)

    def forward(self, x, mask=None, quadrature_weights=None):
        B, N, C = x.shape
        mask_n = None
        if mask is not None:
            if mask.dim() == 3:
                mask_n = mask
            else:
                mask_n = mask.unsqueeze(-1)
            mask_n = mask_n.to(dtype=x.dtype, device=x.device).view(B, 1, N, 1)

        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # Eidetic Slice: token-dependent 温度 + Gumbel-Softmax
        logits = self.in_project_slice(x_mid)
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = _gumbel_softmax(logits, temperature)
        if mask_n is not None:
            slice_weights = slice_weights * mask_n
        q = _broadcast_quadrature_weights(
            quadrature_weights, B=B, N=N, dtype=slice_weights.dtype, device=slice_weights.device
        )
        slice_weights_q = slice_weights * q if q is not None else slice_weights
        slice_norm = slice_weights_q.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_q)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)

        if self.use_spectral:
            out_slice_token = out_slice_token + self.spectral_layer(out_slice_token)

        if self.record_attention:
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
            self.last_slice_weights_q = slice_weights_q.detach()
            self.last_slice_mass = slice_norm.detach()
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class LocalEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='gelu'):
        super().__init__()
        if act in ACTIVATION.keys():
            act_fn = ACTIVATION[act]
        else:
            act_fn = nn.GELU
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            act_fn(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, knn_idx, knn_valid=None):
        # x: [B, N, C]
        # knn_idx: [B, N, K]
        # knn_valid: [B, N, K]
        B, N, C = x.shape
        K = knn_idx.shape[-1]
        
        # Get neighbor features using advanced indexing
        # knn_idx: [B, N, K] -> [B, N, K, 1] for gather
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, K)  # [B, N, K]
        x_j = x[batch_idx, knn_idx, :]  # [B, N, K, C]
        
        # x_i: [B, N, K, C] - expand center features
        x_i = x.unsqueeze(2).expand(-1, -1, K, -1)  # [B, N, K, C]
        
        # Edge features: [x_i, x_j - x_i]
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)  # [B, N, K, 2*C]
        
        # MLP
        edge_feat = self.mlp(edge_feat)  # [B, N, K, C_out]
        
        if knn_valid is not None:
            # Mask invalid neighbors (padding)
            edge_feat = edge_feat * knn_valid.unsqueeze(-1).to(edge_feat.dtype)
            # Aggregate (mean)
            out = edge_feat.sum(dim=2) / (knn_valid.sum(dim=2, keepdim=True).to(edge_feat.dtype) + 1e-5)
        else:
            # Aggregate (mean)
            out = edge_feat.mean(dim=2)  # [B, N, C_out]
            
        return out


class Transolver_block(nn.Module):
    """Transformer encoder block.
    use_linearno=True: LinearNO (解耦 Softmax@M/@N，无 slice 间注意力)
    use_eidetic=True: Slice 分配用 token-dependent 温度 + Gumbel-Softmax (Transolver++)
    use_sato=True: 忠实 SATO 融合——全局仍为 self.Attn，另加 Serialized_Attention 局部支路
    quad_sato=True: 局部 Softmax 用 node_weight 面积加权（需一并传入 quadrature_weights）
    默认: 原始 Physics_Attention_Irregular_Mesh (Softmax + 固定温度)
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            use_spectral=False,
            spectral_modes=16,
            spectral_hi_freq_boost=1.0,
            use_edge_conv=False,
            use_eidetic=False,
            use_linearno=False,
            use_sato=False,
            quad_sato=False,
            patch_size=32,
            shift=1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_edge_conv = use_edge_conv
        self.use_eidetic = use_eidetic
        self.use_linearno = use_linearno
        self.use_sato = use_sato
        self.quad_sato = bool(quad_sato) and bool(use_sato)
        if self.use_edge_conv:
            self.edge_conv = LocalEdgeConv(hidden_dim, hidden_dim, act=act)
            self.ln_edge = nn.LayerNorm(hidden_dim)

        self.ln_1 = nn.LayerNorm(hidden_dim)
        if use_linearno:
            if use_eidetic:
                raise ValueError("use_linearno and use_eidetic are mutually exclusive")
            if use_spectral:
                raise ValueError("use_linearno does not support use_spectral (LinearNO has no slice Attn/spectral)")
            self.Attn = LinearNO_Attention_Irregular_Mesh(
                hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                dropout=dropout, slice_num=slice_num)
        elif use_eidetic:
            self.Attn = Physics_Attention_Irregular_EideticSlice(
                hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                dropout=dropout, slice_num=slice_num,
                use_spectral=use_spectral, spectral_modes=spectral_modes,
                spectral_hi_freq_boost=spectral_hi_freq_boost)
        else:
            self.Attn = Physics_Attention_Irregular_Mesh(
                hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                dropout=dropout, slice_num=slice_num,
                use_spectral=use_spectral, spectral_modes=spectral_modes,
                spectral_hi_freq_boost=spectral_hi_freq_boost)
        if use_sato:
            self.local_ln_1 = nn.LayerNorm(hidden_dim)
            self.local_attention = Serialized_Attention(
                patch_size, shift, hidden_dim, num_heads, dropout=min(dropout, 0.1) if dropout else 0.1
            )
            self.local_ln_2 = nn.LayerNorm(hidden_dim)
            self.local_gate = nn.Parameter(torch.tensor([0.0]))
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        fx,
        mask=None,
        knn_idx=None,
        knn_valid=None,
        quadrature_weights=None,
        order=None,
        inverse=None,
        return_hidden=False,
    ):
        if self.use_edge_conv:
            if knn_idx is not None:
                fx = self.edge_conv(self.ln_edge(fx), knn_idx, knn_valid) + fx

        if self.use_sato:
            if order is None or inverse is None:
                raise ValueError("use_sato=1 requires order/inverse from sato_indices")
            x0 = fx
            fxn = self.ln_1(fx)
            fx1 = self.Attn(fxn, mask=mask, quadrature_weights=quadrature_weights)
            B, N, _C = fx.shape
            batch_indices = torch.arange(B, device=fx.device).unsqueeze(-1)
            fx2 = (fxn - fx1)[batch_indices, order]
            q_local = None
            if self.quad_sato and quadrature_weights is not None:
                # [B,N] areas in mesh order → serialization order (same as fx2)
                q_b = quadrature_weights
                if not torch.is_tensor(q_b):
                    q_b = torch.as_tensor(q_b, device=fx.device, dtype=fx.dtype)
                else:
                    q_b = q_b.to(device=fx.device, dtype=fx.dtype)
                if q_b.dim() == 1:
                    q_b = q_b.unsqueeze(0).expand(B, -1)
                elif q_b.dim() > 2:
                    q_b = q_b.reshape(B, N)
                q_local = q_b[batch_indices, order]
            fx2 = self.local_ln_2(
                self.local_attention(self.local_ln_1(fx2), quadrature_weights=q_local)
            )
            fx2 = fx2[batch_indices, inverse]
            if mask is not None:
                m = mask.squeeze(-1) if mask.dim() == 3 else mask
                fx2 = fx2 * m.unsqueeze(-1).to(dtype=fx2.dtype)
            fx = self.mlp(self.ln_2(x0 + fx1 + self.local_gate * fx2)) + x0
        else:
            ln_fx = self.ln_1(fx)
            fx = self.Attn(ln_fx, mask=mask, quadrature_weights=quadrature_weights) + fx
            fx = self.mlp(self.ln_2(fx)) + fx
        # fx here = node features after Attn+MLP residual (pre output proj)
        if self.last_layer:
            out = self.mlp2(self.ln_3(fx))
            if return_hidden:
                return out, fx
            return out
        if return_hidden:
            return fx, fx
        return fx


class Model(nn.Module):
    def __init__(self,
                 # space_dim=1,
                 space_dim=3,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 # fun_dim=1,
                 fun_dim=2,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 use_spectral=False,
                 spectral_modes=16,
                 spectral_hi_freq_boost=1.0,
                 use_edge_conv=False,
                 use_eidetic=False,
                 use_linearno=False,
                 use_sato=False,
                 quad_sato=False,
                 patch_size=32,
                 shift=1,
                 use_geom_memory=False,
                 geom_memory_policy="early",
                 geom_memory_dim=None,
                 geom_encoder_hidden=None,
                 geom_sdf_channel=-1,
                 ):
        super(Model, self).__init__()
        if use_sato and use_linearno:
            self.__name__ = 'LinearNO_SATO'
        elif use_sato:
            self.__name__ = 'Transolver_SATO'
        elif use_linearno:
            self.__name__ = 'LinearNO'
        elif use_eidetic:
            self.__name__ = 'Transolver_plus'
        else:
            self.__name__ = 'Transolver_1D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.n_layers = int(n_layers)
        self.space_dim = int(space_dim)
        # First 2 dims are geometry (x,y); extras (e.g. SDF) ride along pos.
        self.geom_dim = 2
        self.pos_extra_dim = max(0, self.space_dim - self.geom_dim)
        self.use_edge_conv = use_edge_conv
        self.use_linearno = use_linearno
        self.use_sato = use_sato
        self.quad_sato = bool(quad_sato) and bool(use_sato)
        self.use_geom_memory = bool(use_geom_memory)
        self.geom_sdf_channel = int(geom_sdf_channel)
        if self.unified_pos:
            # get_grid(xy) → ref*ref, then concat pos extras (SDF, ...)
            pos_feat = self.ref * self.ref + self.pos_extra_dim
        else:
            pos_feat = self.space_dim
        self.preprocess = MLP(fun_dim + pos_feat, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      use_spectral=use_spectral,
                                                      spectral_modes=spectral_modes,
                                                      spectral_hi_freq_boost=spectral_hi_freq_boost,
                                                      use_edge_conv=use_edge_conv,
                                                      use_eidetic=use_eidetic,
                                                      use_linearno=use_linearno,
                                                      use_sato=use_sato,
                                                      quad_sato=self.quad_sato,
                                                      patch_size=patch_size,
                                                      shift=shift,
                                                      last_layer=(_ == n_layers - 1))
                                     for _ in range(n_layers)])

        # Geometry memory (FiLM): shared pointwise encoder + per-layer γ/β heads.
        # G=[x_norm,y_norm,SDF_norm]; globals in fx are NOT used here.
        if self.use_geom_memory:
            mem_dim = int(geom_memory_dim) if geom_memory_dim is not None else int(n_hidden)
            enc_h = int(geom_encoder_hidden) if geom_encoder_hidden is not None else int(n_hidden)
            self.geom_inject = _geom_inject_policy(n_layers, geom_memory_policy)
            self.geom_encoder = PointwiseGeomEncoder(in_dim=3, hidden=enc_h, out_dim=mem_dim)
            # Inactive layers: Identity placeholder (ModuleList cannot hold bare None)
            self.geom_films = nn.ModuleList(
                [
                    GeomFiLM(mem_dim, n_hidden) if flag else nn.Identity()
                    for flag in self.geom_inject
                ]
            )
            print(
                f"[*] GeomMemory FiLM on: policy={geom_memory_policy!r} "
                f"inject={self.geom_inject} C_g={mem_dim} enc_h={enc_h} "
                f"sdf_ch={self.geom_sdf_channel}",
                flush=True,
            )
        else:
            self.geom_inject = [False] * int(n_layers)
            self.geom_encoder = None
            self.geom_films = None

        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        # Restore FiLM identity init after trunc_normal on all Linears
        if self.use_geom_memory:
            for film in self.geom_films:
                if isinstance(film, GeomFiLM):
                    nn.init.zeros_(film.to_gamma.weight)
                    nn.init.ones_(film.to_gamma.bias)
                    nn.init.zeros_(film.to_beta.weight)
                    nn.init.zeros_(film.to_beta.bias)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        # x: B N 2 (geometry only)
        # grid_ref
        if x.shape[-1] != 2:
            raise ValueError(f"get_grid expects xy with C=2, got {tuple(x.shape)}")
        dev = x.device
        dt = x.dtype
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float32, device=dev)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1]).to(dtype=dt)
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float32, device=dev)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1]).to(dtype=dt)
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def _geom_memory_features(self, x_xy, fx_raw, x_extra):
        """Build G=[xy, SDF] in R^{B,N,3}. Prefer SDF from fx; else pos extra."""
        if fx_raw is not None and fx_raw.shape[-1] > 0:
            c = int(fx_raw.shape[-1])
            ch = int(self.geom_sdf_channel)
            if ch < 0:
                ch = c + ch
            if ch < 0 or ch >= c:
                raise ValueError(f"geom_sdf_channel={self.geom_sdf_channel} out of range for fx C={c}")
            sdf = fx_raw[..., ch : ch + 1]
        elif x_extra is not None and x_extra.shape[-1] > 0:
            sdf = x_extra[..., -1:]
        else:
            raise ValueError("use_geom_memory=1 needs SDF in fx (or pos extra when sdf_to_pos=1)")
        return torch.cat([x_xy, sdf], dim=-1)

    def _apply_geom_film(self, layer_index: int, v: torch.Tensor, geom_m: torch.Tensor, mask=None):
        film = self.geom_films[layer_index]
        if not isinstance(film, GeomFiLM):
            return v
        out = film(v, geom_m)
        if mask is not None:
            m = mask.squeeze(-1) if mask.dim() == 3 else mask
            out = out * m.unsqueeze(-1).to(dtype=out.dtype)
        return out

    def forward(
        self,
        x,
        fx,
        T=None,
        mask=None,
        knn_idx=None,
        knn_valid=None,
        quadrature_weights=None,
        sato_indices=None,
        return_hidden=False,
    ):
        # x: [B,N,2] or [B,N,2+extra] (extra e.g. SDF). Grid uses xy only.
        # return_hidden=True → (out, [V0,V1,...,VL]) node features; VL = last
        # block pre-projection hidden (not the PDE head output). Post-FiLM if injected.
        if x.shape[-1] < self.geom_dim:
            raise ValueError(f"pos C={x.shape[-1]} < geom_dim={self.geom_dim}")
        x_xy = x[..., : self.geom_dim]
        x_extra = x[..., self.geom_dim :] if x.shape[-1] > self.geom_dim else None
        fx_raw = fx  # keep original fx for GeomEncoder (before grid concat)

        geom_m = None
        if self.use_geom_memory:
            g = self._geom_memory_features(x_xy, fx_raw, x_extra)
            geom_m = self.geom_encoder(g)

        if self.unified_pos:
            x = self.get_grid(x_xy, x_xy.shape[0])
            if x_extra is not None and x_extra.shape[-1] > 0:
                x = torch.cat((x, x_extra), dim=-1)
        else:
            x = x
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None and self.Time_Input:
            Time_emb = timestep_embedding(T, self.n_hidden).unsqueeze(1).expand(-1, fx.shape[1], -1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        hiddens = [fx] if return_hidden else None

        order_all = inverse_all = None
        if self.use_sato:
            if sato_indices is None:
                raise ValueError("use_sato=1 requires sato_indices={'order','inverse'}")
            order_all = sato_indices["order"]
            inverse_all = sato_indices["inverse"]

        for layer_index, block in enumerate(self.blocks):
            order_i = inverse_i = None
            if self.use_sato:
                st = int(layer_index % order_all.shape[1])
                order_i = order_all[:, st, :]
                inverse_i = inverse_all[:, st, :]
            inject = bool(self.use_geom_memory and self.geom_inject[layer_index])

            if block.last_layer and inject:
                # FiLM on pre-projection hidden, then re-apply output head
                _out, h = block(
                    fx,
                    mask=mask,
                    knn_idx=knn_idx,
                    knn_valid=knn_valid,
                    quadrature_weights=quadrature_weights,
                    order=order_i,
                    inverse=inverse_i,
                    return_hidden=True,
                )
                h = self._apply_geom_film(layer_index, h, geom_m, mask=mask)
                fx = block.mlp2(block.ln_3(h))
                if return_hidden:
                    hiddens.append(h)
            elif return_hidden:
                fx, h = block(
                    fx,
                    mask=mask,
                    knn_idx=knn_idx,
                    knn_valid=knn_valid,
                    quadrature_weights=quadrature_weights,
                    order=order_i,
                    inverse=inverse_i,
                    return_hidden=True,
                )
                if inject:
                    # non-last: fx==h; last without inject uses pre-proj h for probe
                    if not block.last_layer:
                        fx = self._apply_geom_film(layer_index, fx, geom_m, mask=mask)
                        h = fx
                    # last + no inject: keep projected fx, probe sees pre-proj h
                hiddens.append(h)
            else:
                fx = block(
                    fx,
                    mask=mask,
                    knn_idx=knn_idx,
                    knn_valid=knn_valid,
                    quadrature_weights=quadrature_weights,
                    order=order_i,
                    inverse=inverse_i,
                )
                if inject and not block.last_layer:
                    fx = self._apply_geom_film(layer_index, fx, geom_m, mask=mask)

        if return_hidden:
            return fx, hiddens
        return fx

    def enable_attention_recording(self, enabled=True):
        """Enable/disable attention recording for visualization."""
        for block in self.blocks:
            if hasattr(block, 'Attn') and hasattr(block.Attn, 'record_attention'):
                block.Attn.record_attention = bool(enabled)
                if enabled:
                    for attr in ('last_attn', 'last_slice_weights', 'last_phi', 'last_psi',
                                 'last_slice_weights_q', 'last_slice_mass'):
                        if hasattr(block.Attn, attr):
                            delattr(block.Attn, attr)

    def get_last_attention(self):
        """Get attention weights and slice weights from the last forward pass.
        
        Returns:
            list: List of dicts, each containing:
                - 'layer': int, layer index
                - 'attn': Tensor[B,H,G,G] or None, attention matrix between slice tokens
                - 'slice_weights': Tensor[B,H,N,G] or None, φ(Q) / shared W
                - 'phi' / 'psi': LinearNO 时额外返回
        """
        results = []
        for layer_index, block in enumerate(self.blocks):
            attn = getattr(block.Attn, 'last_attn', None)
            slice_weights = getattr(block.Attn, 'last_slice_weights', None)
            phi = getattr(block.Attn, 'last_phi', None)
            psi = getattr(block.Attn, 'last_psi', None)
            if (attn is not None) or (slice_weights is not None) or (phi is not None) or (psi is not None):
                results.append({
                    'layer': layer_index,
                    'attn': attn,
                    'slice_weights': slice_weights,
                    'phi': phi,
                    'psi': psi,
                })
        return results
