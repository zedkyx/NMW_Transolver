import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def _broadcast_quadrature_weights(quadrature_weights, *, B, N, dtype, device):
    """Normalize optional q to [B,1,N,1] for multiplying slice_weights [B,H,N,G]."""
    if quadrature_weights is None:
        return None
    q = quadrature_weights
    if not torch.is_tensor(q):
        q = torch.as_tensor(q, dtype=dtype, device=device)
    else:
        q = q.to(device=device, dtype=dtype)
    if q.dim() == 1:
        if q.shape[0] != N:
            raise ValueError(f"quadrature_weights length {q.shape[0]} != N={N}")
        q = q.view(1, 1, N, 1).expand(B, 1, N, 1)
    elif q.dim() == 2:
        if q.shape != (B, N):
            raise ValueError(f"quadrature_weights shape {tuple(q.shape)} != {(B, N)}")
        q = q.view(B, 1, N, 1)
    elif q.dim() == 4:
        if q.shape[0] != B or q.shape[2] != N:
            raise ValueError(f"quadrature_weights shape {tuple(q.shape)} incompatible with B={B}, N={N}")
        q = q
    else:
        raise ValueError(f"quadrature_weights must be [N], [B,N], or [B,*,N,*], got {tuple(q.shape)}")
    return q


class SpectralLayer(nn.Module):
    def __init__(self, dim, modes, n_heads, hi_freq_boost=1.0):
        super().__init__()
        self.modes = modes
        self.n_heads = n_heads
        self.scale = (1 / (dim * modes))
        self.hi_freq_boost = hi_freq_boost
        # 存储为纯实数，完全避开复数类型的内存对齐 Bug
        self.weights = nn.Parameter(self.scale * torch.randn(n_heads, dim, modes, 2))

    def forward(self, x):
        # x: [B, H, G, D] where G is slice_num
        B, H, G, D = x.shape
        x_fft = torch.fft.rfft(x, dim=-2)  # [B, H, G//2 + 1, D]

        # 手动拆解实部和虚部进行复数乘法，不再调用 view_as_complex
        w_real = self.weights[:, :, :, 0].permute(0, 2, 1).unsqueeze(0)  # [1, H, modes, D]
        w_imag = self.weights[:, :, :, 1].permute(0, 2, 1).unsqueeze(0)  # [1, H, modes, D]

        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        m = min(self.modes, x_fft.shape[-2])
        out_fft_real = torch.zeros_like(x_fft_real)
        out_fft_imag = torch.zeros_like(x_fft_imag)

        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_fft_real[:, :, :m, :] = x_fft_real[:, :, :m, :] * w_real[:, :, :m, :] - \
                                    x_fft_imag[:, :, :m, :] * w_imag[:, :, :m, :]
        out_fft_imag[:, :, :m, :] = x_fft_real[:, :, :m, :] * w_imag[:, :, :m, :] + \
                                    x_fft_imag[:, :, :m, :] * w_real[:, :, :m, :]
        if m > 1:
            freq = torch.arange(m, device=x.device, dtype=x_fft_real.dtype)
            hi_freq_weight = 1.0 + self.hi_freq_boost * (freq / (m - 1))
            hi_freq_weight = hi_freq_weight.view(1, 1, m, 1)
            out_fft_real[:, :, :m, :] = out_fft_real[:, :, :m, :] * hi_freq_weight
            out_fft_imag[:, :, :m, :] = out_fft_imag[:, :, :m, :] * hi_freq_weight

        out_fft = torch.complex(out_fft_real, out_fft_imag)
        out = torch.fft.irfft(out_fft, n=G, dim=-2)
        return out


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, use_spectral=False,
                 spectral_modes=16, spectral_hi_freq_boost=1.0):
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
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.use_spectral = use_spectral
        if self.use_spectral:
            self.spectral_layer = SpectralLayer(dim_head, spectral_modes, heads, hi_freq_boost=spectral_hi_freq_boost)

    def forward(self, x, mask=None, quadrature_weights=None):
        # B N C
        B, N, C = x.shape
        mask_n = None
        if mask is not None:
            if mask.dim() == 3:
                mask_n = mask
            else:
                mask_n = mask.unsqueeze(-1)
            mask_n = mask_n.to(dtype=x.dtype, device=x.device).view(B, 1, N, 1)

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        if mask_n is not None:
            slice_weights = slice_weights * mask_n
        q = _broadcast_quadrature_weights(
            quadrature_weights, B=B, N=N, dtype=slice_weights.dtype, device=slice_weights.device
        )
        slice_weights_q = slice_weights * q if q is not None else slice_weights
        slice_norm = slice_weights_q.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights_q)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (2.5) Optional Spectral Filtering (FNO-style)
        if hasattr(self, "use_spectral") and self.use_spectral:
            out_slice_token = out_slice_token + self.spectral_layer(out_slice_token)

        ### (3) Deslice (assignment s only; do not multiply by q)
        if getattr(self, "record_attention", False):
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
            self.last_slice_weights_q = slice_weights_q.detach()
            self.last_slice_mass = slice_norm.detach()
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class LinearNO_Attention_Irregular_Mesh(nn.Module):
    """LinearNO (HiPRL/LinearNO AirfRANS): canonical linear attention on irregular mesh.

    φ(Q)=Softmax@M, ψ(K)=Softmax@N, no slice-token attention.
    Keeps optional mask / quadrature_weights (slice-side ψ only).
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.record_attention = False

        # Match official AirfRANS LinearNO: shared in_project_x, separate to_q/k/v
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, slice_num, bias=False)
        self.to_k = nn.Linear(dim_head, slice_num, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, quadrature_weights=None):
        # B N C
        B, N, C = x.shape
        mask_n = None
        if mask is not None:
            if mask.dim() == 3:
                mask_n = mask
            else:
                mask_n = mask.unsqueeze(-1)
            mask_n = mask_n.to(dtype=x.dtype, device=x.device).view(B, 1, N, 1)

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N D

        q = self.to_q(x_mid)  # B H N G  (φ logits)
        k = self.to_k(x_mid)  # B H N G  (ψ logits)
        v = self.to_v(x_mid)  # B H N D

        # Softmax@M for φ (deslice)
        q = F.softmax(q, dim=-1)
        if mask_n is not None:
            q = q * mask_n

        # Softmax@N for ψ (slice); mask via -inf so padded nodes get zero mass
        if mask_n is not None:
            k = k.masked_fill(mask_n.expand_as(k) <= 0, torch.finfo(k.dtype).min)
        k = F.softmax(k, dim=-2)

        # Quadrature on slice side only; renormalize along N (like Physics-Attn sum-norm)
        qw = _broadcast_quadrature_weights(
            quadrature_weights, B=B, N=N, dtype=k.dtype, device=k.device
        )
        if qw is not None:
            k = k * qw
            # ψ 已沿节点 Softmax，分母是 q 的加权均值，不能照搬 Transolver 的 +1e-5
            denom = k.sum(dim=2, keepdim=True)
            k = k / denom.clamp_min(torch.finfo(k.dtype).eps)

        # Linear attention: out = φ (ψ^T V)
        kv = torch.einsum("bhng,bhnd->bhgd", k, v)
        out_x = torch.einsum("bhng,bhgd->bhnd", q, kv)

        if getattr(self, "record_attention", False):
            self.last_attn = None
            self.last_slice_weights = q.detach()  # φ for viz compatibility
            self.last_phi = q.detach()
            self.last_psi = k.detach()
            if qw is not None:
                self.last_slice_weights_q = k.detach()
                self.last_slice_mass = k.sum(2).detach()

        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.record_attention = False

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (2.5) Optional Spectral Filtering (FNO-style)
        if hasattr(self, "use_spectral") and self.use_spectral:
            out_slice_token = out_slice_token + self.spectral_layer(out_slice_token)

        ### (3) Deslice
        if getattr(self, "record_attention", False):
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D
        self.record_attention = False

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (2.5) Optional Spectral Filtering (FNO-style)
        if hasattr(self, "use_spectral") and self.use_spectral:
            out_slice_token = out_slice_token + self.spectral_layer(out_slice_token)

        ### (3) Deslice
        if getattr(self, "record_attention", False):
            self.last_attn = attn.detach()
            self.last_slice_weights = slice_weights.detach()
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
