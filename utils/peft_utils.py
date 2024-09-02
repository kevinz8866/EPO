from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from ..parser import load_config, parse_args


def generate_peft_config(config):
    params = {}
    for i in config.model.peft:
        if i == 'method': continue
        params.update({i : config.model.peft[i]})

    r = LoraConfig(**params)
    return r


def sanity_check():

    args = parse_args()
    cfg = load_config(args)
    res = generate_peft_config(cfg)
    print(res)

if __name__ == '__main__':
    sanity_check()
