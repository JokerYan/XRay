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

        output_x_mean = torch.mean(output_x_list, dim=0)
        output_c_mean = torch.mean(output_c_list, dim=0)

        entropy_threshold = 0.6853
        output_c_entropy = entropy_threshold - torch.distributions.Categorical(probs=output_c_list.reshape(-1)).entropy().reshape([1]) + 0.5

        variance_threshold = 0.03
        output_c_variance = torch.var(output_c_list, dim=0) + (0.5 - variance_threshold)

        # output_c_final = torch.max(torch.stack([output_c_mean, output_c_variance]), dim=0).values
        output_c_final = torch.max(torch.stack([output_c_mean, output_c_entropy]), dim=0).values

        print(output_c_list.reshape(-1))
        print(output_c_mean)
        # print(output_c_variance)
        print(output_c_entropy)

        return output_x_mean, output_c_final
