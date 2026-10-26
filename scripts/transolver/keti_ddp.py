import os
import sys

# add project root to sys.path (keeps the same pattern as exp_* scripts)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.transolver.keti_argparser import build_argparser, load_yaml_config
from scripts.transolver.train.keti_trainer import run


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args = load_yaml_config(args, parser=parser)
    # keti_ddp.py is a DDP-only entry: force enable DDP so users don't need to pass --ddp
    args.ddp = True
    run(args)


if __name__ == "__main__":
    main()


