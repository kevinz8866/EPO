from .video_language_task import VideoLanguageTask
from .video_reward_task import VideoRewardTask

def load_task(cfg, steps_in_epoch=1):
    if cfg.data.dataset == "alfred":
        return VideoLanguageTask(cfg, steps_in_epoch)
    elif cfg.data.dataset == "alfred_reward":        
        return VideoRewardTask(cfg, steps_in_epoch)  
    else:
        raise ValueError("config not available.")
