import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn.functional import interpolate, sigmoid

from lib.models.cls_hrnet import get_cls_net, BN_MOMENTUM


class TempSigmoid(nn.Sigmoid):
    def __init__(self, T=1):
        super().__init__()
        self.T = T
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input / self.T)


class XRayNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(XRayNet, self).__init__()

        self.cfg = cfg
        self.hrnet = get_cls_net(cfg)
        self.sigmoid_T = self.cfg["MODEL"]["TEMPERATURE"]
        self.temp_sigmoid = TempSigmoid(self.cfg["MODEL"]["TEMPERATURE"])
        self._make_head(self.hrnet.last_pre_stage_channels)

    # make xray head
    # adapted from hrnet face landmarks
    def _make_head(self, pre_stage_channels):
        final_inp_channels = sum(pre_stage_channels)
        extra = self.cfg['MODEL']['EXTRA']
        # self.head = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=final_inp_channels,
        #         out_channels=final_inp_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
        #     BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=final_inp_channels,
        #         out_channels=self.cfg.MODEL.NUM_JOINTS,
        #         kernel_size=extra.FINAL_CONV_KERNEL,
        #         stride=1,
        #         padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        # )
        self.xray_head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )
        )

        self.classification_head = nn.Sequential(
            nn.AvgPool2d(self.cfg.MODEL.IMAGE_SIZE[0]),
            nn.Linear(1, 1),
            nn.Sigmoid(),
            # self.temp_sigmoid,
        )

    def forward(self, x):
        x = self.hrnet.forward(x)

        # Head Part
        height, width = x[0].size(2), x[0].size(3)
        x1 = interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.xray_head(x)
        x = interpolate(x, size=(self.cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[0]),
                        mode='bilinear', align_corners=False)
        x_temp = sigmoid(x / self.sigmoid_T)
        x = sigmoid(x)
        # x_temp = self.temp_sigmoid(x)
        c = self.classification_head(x)
        c = c.reshape([-1])

        return x_temp, c

    def load_hrnet_pretrained(self, model_file):
        self.hrnet.load_state_dict(model_file)

    def freeze_hrnet(self):
        self.hrnet.freeze_weights()

    def unfreeze_hrnet(self):
        self.hrnet.unfreeze_weights()

