import torch
import os
import sys
import argparse

# Add workspace root to sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, _REPO_ROOT)

from scripts.transolver.tools.cache.serialization import get_serialization_index


def parse_args():
    p = argparse.ArgumentParser(description="Precompute SATO serialization indices for a cache file")
    p.add_argument("--cache_path", type=str, default="", help="Path to cache .pt file (or take from --config)")
    p.add_argument("--save_path", type=str, default="", help="Output path (default: <cache_path> with _sato_index.pt)")
    p.add_argument(
        "--grid_size",
        type=float,
        default=0.01,
        help="Quantization grid size (keep in sync with yaml data.sato_grid_size)",
    )
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional train yaml: read data.sato_grid_size (and queue/cache_path if --cache_path omitted)",
    )
    return p.parse_args()


def _grid_size_from_config(config_path: str):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data = cfg.get("data") or {}
    if "sato_grid_size" in data and data["sato_grid_size"] is not None:
        return float(data["sato_grid_size"])
    return None


def _cache_from_config(config_path: str):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    data = cfg.get("data") or {}
    if data.get("cache_path"):
        return str(data["cache_path"])
    key = cfg.get("key") or {}
    if key.get("cache_path"):
        return str(key["cache_path"])
    queue = cfg.get("queue") or {}
    jobs = queue.get("jobs") or []
    if jobs and isinstance(jobs[0], dict) and jobs[0].get("cache_path"):
        return str(jobs[0]["cache_path"])
    return ""


def main():
    args = parse_args()
    cache_path = args.cache_path
    grid_size = args.grid_size
    if args.config:
        if not cache_path:
            cache_path = _cache_from_config(args.config)
        gs = _grid_size_from_config(args.config)
        if gs is not None:
            grid_size = gs
            print(f"[*] sato_grid_size from config: {grid_size}")
    if not cache_path:
        print("Error: --cache_path required (or provide --config with cache_path)")
        return
    save_path = args.save_path or cache_path.replace(".pt", "_sato_index.pt")
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache not found at {cache_path}")
        return

    print(f"Loading cache from {cache_path}...")
    data = torch.load(cache_path, map_location="cpu")
    
    # pos shape is likely (B, N, 2)
    pos = data["pos"]
    B, N, D = pos.shape
    print(f"Dataset info: B={B}, N={N}, D={D}")

    orders_list = ["z", "z-trans", "hilbert", "hilbert-trans"]
    
    print(f"Computing serialization indices (grid_size={grid_size})...")
    
    order, inverse = get_serialization_index(pos, grid_size=grid_size, orders=orders_list)
    
    print(f"Computed order shape: {order.shape}") # (B, 4, N)
    print(f"Computed inverse shape: {inverse.shape}") # (B, 4, N)
    
    # Save results
    save_dict = {
        "order": order,
        "inverse": inverse,
        "orders_list": orders_list,
        "grid_size": grid_size
    }
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Successfully saved serialization indices to {save_path}")

if __name__ == "__main__":
    main()
