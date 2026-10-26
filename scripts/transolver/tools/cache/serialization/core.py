import torch
from tqdm import tqdm
from .hilbert import encode as hilbert_encode
from .z_order import xyz2key as z_order_encode

@torch.inference_mode()
def get_serialization_index(pos, grid_size=0.05, depth=16, orders=None):
    """
    Generate serialization order and inverse index for a batch of point clouds.
    
    Args:
        pos: (B, N, 2) or (B, N, 3) torch.Tensor
        grid_size: float or list of 3 floats
        depth: int, bits for encoding
        orders: list of str, e.g., ["z", "z-trans", "hilbert", "hilbert-trans"]
        
    Returns:
        order: (B, num_orders, N)
        inverse: (B, num_orders, N)
    """
    if orders is None:
        orders = ["z", "z-trans", "hilbert", "hilbert-trans"]
    
    B, N, D = pos.shape
    device = pos.device
    
    # 1. Prepare 3D coordinates
    if D == 2:
        pos_3d = torch.cat([pos, torch.zeros_like(pos[:, :, :1])], dim=-1)
    else:
        pos_3d = pos
        
    if isinstance(grid_size, float):
        grid_size = torch.tensor([grid_size, grid_size, grid_size], device=device)
    else:
        grid_size = torch.tensor(grid_size, device=device)

    # 2. Quantization (Batch-wise)
    # To handle batch-wise min correctly, we can either use global min or per-sample min.
    # SATO uses per-sample min in its Point class logic (self.coord - self.coord.min(0)[0])
    # But for a fixed cache, we can just use the batch min to keep everything in the same frame.
    
    # Let's follow SATO's per-sample normalization for better local resolution
    all_orders = []
    all_inverses = []
    
    for b in tqdm(range(B), desc="SATO serialization", unit="case"):
        p = pos_3d[b] # (N, 3)
        p_min = p.min(0)[0]
        grid_coord = torch.div(p - p_min, grid_size, rounding_mode="trunc").int()
        
        case_codes = []
        for order_type in orders:
            if order_type == "z":
                code = z_order_encode(grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2], depth=depth)
            elif order_type == "z-trans":
                code = z_order_encode(grid_coord[:, 1], grid_coord[:, 0], grid_coord[:, 2], depth=depth)
            elif order_type == "hilbert":
                code = hilbert_encode(grid_coord, num_dims=3, num_bits=depth)
            elif order_type == "hilbert-trans":
                code = hilbert_encode(grid_coord[:, [1, 0, 2]], num_dims=3, num_bits=depth)
            else:
                raise ValueError(f"Unknown order type: {order_type}")
            case_codes.append(code)
            
        case_codes = torch.stack(case_codes) # (num_orders, N)
        order = torch.argsort(case_codes, dim=1) # (num_orders, N)
        
        # Generate inverse: inverse[order[i]] = i
        # Using scatter_ to be efficient
        num_orders = len(orders)
        inverse = torch.zeros_like(order)
        src = torch.arange(N, device=device).repeat(num_orders, 1)
        inverse.scatter_(dim=1, index=order, src=src)
        
        all_orders.append(order)
        all_inverses.append(inverse)
        
    return torch.stack(all_orders), torch.stack(all_inverses)
