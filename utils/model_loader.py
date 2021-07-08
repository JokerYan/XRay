import torch
from torch.backends import cudnn

from lib.config import config
from lib.config import update_config
from lib.models.xray_net import XRayNet


class Args:
    def __init__(self):
        self.cfg = None
        self.modelDir = None
        self.logDir = None
        self.dataDir = None
        self.testModel = None


def construct_model(cfg_path, pretrained_model):
    args = Args()
    args.cfg = cfg_path
    args.testModel = pretrained_model
    update_config(config, args)
    config.GPUS = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    model = XRayNet(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    return model, args, config

