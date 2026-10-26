#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一训练启动器：单任务 或 queue 顺序多任务。

用法：
  python scripts/transolver/tools/launch/train_launcher.py --config scripts/transolver/configs/train.yaml
  python scripts/transolver/tools/launch/train_launcher.py --config ... --dry_run
  python scripts/transolver/tools/launch/train_launcher.py --config ... --start_from qjt_u
  python scripts/transolver/tools/launch/train_launcher.py --config ... --ddp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

import yaml

LEGACY_REPO_PREFIX = "/root/transolver"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.transolver.keti_data_loader import default_coord_norm_path

JOB_SKIP_KEYS = {"id", "cache", "cache_path", "save_name", "run_dir", "gpu", "coord_norm_path"}


def _repo_root() -> str:
    return os.environ.get("TRANSOLVER_ROOT", PROJECT_ROOT)


def _rewrite_legacy(value: Any, repo_root: str) -> Any:
    if isinstance(value, str):
        if LEGACY_REPO_PREFIX in value:
            return value.replace(LEGACY_REPO_PREFIX, repo_root)
        return value
    if isinstance(value, list):
        return [_rewrite_legacy(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {k: _rewrite_legacy(v, repo_root) for k, v in value.items()}
    return value


def _resolve_cache_path(data_root: str, cache_spec: str, repo_root: str) -> str:
    cache_spec = _rewrite_legacy(cache_spec, repo_root)
    if not cache_spec:
        return ""
    if os.path.isabs(cache_spec):
        return cache_spec
    base = _rewrite_legacy(data_root, repo_root) if data_root else os.path.join(repo_root, "data")
    return os.path.join(base, cache_spec)


def _expand_run_dir(tpl: str, save_name: str, job_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        tpl.replace("{save_name}", save_name)
        .replace("{job_id}", job_id)
        .replace("{timestamp}", ts)
    )


def _build_jobs(config: dict, key: dict, repo_root: str) -> list[dict]:
    queue = config.get("queue") or {}
    if queue.get("enabled"):
        jobs = queue.get("jobs") or []
        if not jobs:
            raise SystemExit("[!] queue.enabled=true 但 jobs 为空")
        return jobs

    cache_path = key.get("cache_path", "")
    if not cache_path:
        raise SystemExit("[!] 单任务模式请在 key.cache_path 填写 cache，或启用 queue")

    job_id = key.get("job_id", "single")
    return [{"id": job_id, "cache_path": cache_path, **{
        k: v for k, v in key.items()
        if k not in ("data_root", "cache_path", "save_name", "run_dir", "job_id")
    }}]


def _job_train_args(
    config: dict,
    key: dict,
    job: dict,
    repo_root: str,
    config_path: str,
) -> tuple[list[str], str, str, str]:
    data_root = key.get("data_root", "")
    save_name_base = key.get("save_name") or (config.get("train") or {}).get("save_name", "train")
    run_dir_tpl = key.get("run_dir") or (config.get("train") or {}).get(
        "run_dir", "/root/transolver/runs/{save_name}/{job_id}_{timestamp}"
    )
    run_dir_tpl = _rewrite_legacy(run_dir_tpl, repo_root)

    job_id = str(job.get("id", "job"))
    cache_spec = job.get("cache_path") or job.get("cache") or ""
    cache_path = _resolve_cache_path(data_root, cache_spec, repo_root)
    if not cache_path:
        raise ValueError(f"job {job_id}: 缺少 cache / cache_path")

    queue_enabled = bool((config.get("queue") or {}).get("enabled"))
    if job.get("save_name"):
        job_save_name = str(job["save_name"])
    elif queue_enabled:
        job_save_name = f"{save_name_base}_{job_id}"
    else:
        job_save_name = save_name_base
    run_dir = job.get("run_dir") or _expand_run_dir(run_dir_tpl, job_save_name, job_id)

    data_cfg = config.get("data") or {}
    frame_stride = int(job.get("frame_stride") or key.get("frame_stride") or data_cfg.get("frame_stride") or 1)
    frame_offset = int(job.get("frame_offset") or key.get("frame_offset") or data_cfg.get("frame_offset") or 0)
    max_frames = int(job.get("max_frames") or key.get("max_frames") or data_cfg.get("max_frames") or 0)
    user_coord = str(job.get("coord_norm_path") or key.get("coord_norm_path") or "")
    coord_norm_path = default_coord_norm_path(
        cache_path,
        frame_stride=frame_stride,
        frame_offset=frame_offset,
        max_frames=max_frames,
        user_path=user_coord,
    )

    train_args = [
        "--config", config_path,
        "--cache_path", cache_path,
        "--coord_norm_path", coord_norm_path,
        "--save_name", job_save_name,
        "--run_dir", run_dir,
    ]

    for k, v in job.items():
        if k in JOB_SKIP_KEYS or v is None:
            continue
        if k == "model":
            train_args.extend(["--model", str(v)])
            continue
        if k == "output_cols" and isinstance(v, list):
            train_args.extend(["--output_cols", *v])
            continue
        if k == "orders" and isinstance(v, list):
            train_args.extend(["--orders", *v])
            continue
        train_args.extend([f"--{k}", str(v)])

    return train_args, cache_path, job_save_name, run_dir


def _run_cmd(cmd: list[str], run_dir: str, env: dict) -> int:
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    log_file = os.path.join(run_dir, "logs", "train.log")

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
            lf.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser("Transolver train launcher")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_from", type=str, default="", help="从指定 job id 开始（断点续跑）")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_repo_root(), config_path)
    if not os.path.exists(config_path):
        raise SystemExit(f"[!] Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = _rewrite_legacy(yaml.safe_load(f) or {}, _repo_root())

    key = config.get("key") or {}
    queue = config.get("queue") or {}
    jobs = _build_jobs(config, key, _repo_root())

    start_from = args.start_from or queue.get("start_from", "")
    start_idx = 0
    if start_from:
        for i, job in enumerate(jobs):
            if job.get("id") == start_from:
                start_idx = i
                break
        print(f"[*] start_from={start_from} (index {start_idx})")

    keti_dp = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "keti_dp.py"))
    keti_ddp = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "keti_ddp.py"))

    gpu_ids = str(key.get("gpu") or (config.get("train") or {}).get("gpu", "0"))
    gpu_list = [g.strip() for g in gpu_ids.split(",") if g.strip()]
    num_gpus = len(gpu_list) if gpu_list else 1
    mode = "DDP" if args.ddp else "DP"

    for i, job in enumerate(jobs):
        if i < start_idx:
            print(f"[Skip] {job.get('id')} (before start_from)")
            continue

        job_id = job.get("id", f"job_{i}")
        try:
            train_args, cache_path, job_save_name, run_dir = _job_train_args(
                config, key, job, _repo_root(), config_path
            )
        except ValueError as exc:
            print(f"[!] {exc}, skip")
            continue

        if not os.path.exists(cache_path):
            print(f"[!] Cache not found: {cache_path}, skip")
            continue

        if args.ddp:
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                f"--master_port={29600 + i}",
                keti_ddp,
                *train_args,
            ]
        else:
            cmd = [sys.executable, keti_dp, *train_args, "--gpu", gpu_ids]

        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(jobs)}] {job_id} | {mode}")
        print(f"  cache:     {cache_path}")
        print(f"  save_name: {job_save_name}")
        print(f"  run_dir:   {run_dir}")
        print(f"  cmd:       {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        if args.dry_run:
            continue

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TRANSOLVER_ROOT"] = _repo_root()
        if args.ddp:
            env["OMP_NUM_THREADS"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

        ret = _run_cmd(cmd, run_dir, env)
        if ret != 0:
            print(f"[!] Job {job_id} failed (exit {ret})")
            print(f"    Log: {os.path.join(run_dir, 'logs', 'train.log')}")
            sys.exit(ret)
        print(f"[OK] Job {job_id} finished")

    print("\n[OK] All jobs completed")


if __name__ == "__main__":
    main()
