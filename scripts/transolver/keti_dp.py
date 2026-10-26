import os
import sys

# Headless-safe backend before any submodule imports pyplot (avoids exit-time crashes with some stacks)
import matplotlib

matplotlib.use("Agg")

# add project root to sys.path (keeps the same pattern as other entry scripts)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.transolver.keti_argparser import build_argparser, load_yaml_config
from scripts.transolver.train.keti_trainer import run


def main():
    """
    DataParallel (DP) entry for KETI training.

    Notes:
    - This script is DP-only: it forces args.ddp = False.
    - Use --gpu "0,1,2,3" (comma-separated) to choose multiple GPUs.
    """
    parser = build_argparser()
    args = parser.parse_args()
    args = load_yaml_config(args, parser=parser)

    # DP-only
    args.ddp = False
    run(args)


if __name__ == "__main__":
    main()


