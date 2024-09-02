import argparse
from .defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="path to the config file",
        default="not_provided",
        type=str,
    )
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all options.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    cfg = get_cfg()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.exp_name = args.exp_name

    assert cfg.train.enable ^ cfg.test.enable, 'must choose exactly one from either train.enable or test.enable'
    
    cfg.freeze()
    return cfg
