import torch
import torch.nn as nn


class XRayNetEnsemble(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)

    def forward(self, x):
        output_x_list = []
        output_c_list = []
        for model in self.model_list:
            output_x, output_c = model(x)
            output_x_list.append(output_x)
            output_c_list.append(output_c)
        output_x_list = torch.stack(output_x_list)
        output_c_list = torch.stack(output_c_list)
        print(output_c_list)

        output_x_mean = torch.mean(output_x_list, dim=0)
        output_c_mean = torch.mean(output_c_list, dim=0)
        # output_c_entropy = torch.distributions.Categorical(probs=output_c_list).entropy()
        output_c_variance = 1 - torch.var(output_c_list, dim=0)
        return output_x_mean, output_c_variance
