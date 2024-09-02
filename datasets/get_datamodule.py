from .alfred_dataset_action import AlfredDataModule
from .alfred_dataset_action_reward import AlfredRewardModule

def get_dm(cfg):
    if cfg.data.dataset == "alfred":
        return AlfredDataModule(cfg)
    elif cfg.data.dataset == "alfred_reward":
        return AlfredRewardModule(cfg)
    else:
        raise ValueError("config not available")