import json
import torch
import torchvision.transforms as transforms

from train import construct_model
from utils.xray_dataset import XRayDataset
from lib.models.cw_inf_att import CWInfAttack
import utils.image_transforms as custom_transforms

config_path = 'experiments/cw_inf_att.json'
# to change target model, both the 'model_path' in the above config
# and '--cfg' path in attack.sh needs to be changed


def main():
    config_json = json.load(open(config_path))
    model, target_config = load_target_model(config_json['model_path'])
    attack_model = CWInfAttack(
        model,
        config_json['c'],
        config_json['lr'],
        config_json['momentum'],
        config_json['steps'],
    )
    attack_model = attack_model.cuda()

    # image normalized right before passing to model
    # original image needed for attack
    valid_dataset = XRayDataset(
        './data/val_image_selected.csv',
        # normalized transform removed because it is called separately in attack model
        transforms.Compose([
             # TODO: Change Random Crop to Centre Crop
             custom_transforms.ImageToOne(),
             custom_transforms.MaskToXray(),
             custom_transforms.ToTensor(cuda=True),
             custom_transforms.Rescale(int(target_config.MODEL.IMAGE_SIZE[0])),
             # custom_transforms.Grayscale(enabled=config.GRAYSCALE),
             # custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                             std=[0.229, 0.224, 0.225]),
        ]),
        target_class=1,  # only load fake ones
    )

    gpus = list(target_config.GPUS)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config_json['batch_size']*len(gpus),
        shuffle=False,
        num_workers=target_config.WORKERS,
        pin_memory=False
    )

    best_acc_list = []
    best_delta_list = []
    for i, data in enumerate(valid_loader):
        input = data['video_frame']
        target_x = data['mask_frame']
        target_c = data['is_fake']
        best_adv_images, best_acc, best_delta = attack_model(input, target_c)
        best_acc_list.append(best_acc)
        best_delta_list.append(best_delta)
    print('===== Attack finished =====')
    print('Avg Acc: {}\tAvg Delta: {}'.format(
        sum(best_acc_list) / len(best_acc_list),
        sum(best_delta_list) / len(best_delta_list)
    ))


def load_target_model(model_path):
    model, target_config = construct_model()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.cuda()
    gpus = list(target_config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    return model, target_config


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
