import json

import torch
from torch.backends import cudnn

from lib.config import config
from lib.config import update_config
from lib.models.xray_net import XRayNet
from lib.models.xray_net_ensemble import XRayNetEnsemble


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
    config.defrost()
    config.GPUS = [i for i in range(torch.cuda.device_count())]
    config.freeze()
    model = XRayNet(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    return model, args, config


def construct_ensemble_model(cfg_json_path):
    config_json = json.load(open(cfg_json_path))
    model_list = []
    for model_info in config_json['models']:
        model, args, target_config = construct_model(model_info['model_cfg_path'], pretrained_model="")
        state_dict = torch.load(model_info['model_path'])
        model.load_state_dict(state_dict)
        model = model.cuda()
        model_list.append(model)
    model_ensemble = XRayNetEnsemble(model_list)
    gpus = [i for i in range(torch.cuda.device_count())]
    model_ensemble = torch.nn.DataParallel(model_ensemble, device_ids=gpus).cuda()
    return model_ensemble
