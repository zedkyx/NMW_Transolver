import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DDPContext:
    enabled: bool
    device: torch.device
    is_main_process: bool
    world_size: int
    rank: int
    local_rank: int


def init_ddp_if_needed(args) -> DDPContext:
    """
    Initialize torch.distributed if args.ddp is True.
    Keeps the legacy behavior used in exp_keti_multi_ddp.py.
    """
    if not getattr(args, "ddp", False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DDPContext(
            enabled=False,
            device=device,
            is_main_process=True,
            world_size=1,
            rank=0,
            local_rank=-1,
        )

    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    if local_rank == -1:
        raise ValueError("DDP requires torchrun (LOCAL_RANK/RANK env vars).")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        # fallback for legacy launchers
        master_addr = getattr(args, "master_addr", "localhost")
        master_port = getattr(args, "master_port", "12355")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", str(local_rank))),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    if is_main:
        print(f"Initialized DDP: world_size={world_size}, rank=0, device={device}")

    return DDPContext(
        enabled=True,
        device=device,
        is_main_process=is_main,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def cleanup_ddp(ctx: DDPContext) -> None:
    if ctx.enabled:
        dist.destroy_process_group()


