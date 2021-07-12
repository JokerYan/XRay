import pickle
import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms

from lib.core.evaluate import cal_accuracy
from utils.debug_tools import save_image_stack, clear_debug_image


class CWInfAttack(nn.Module):
    '''
    c:  coefficient of f value to be added to distance.
        Higher the c, higher the success rate and higher the distance
        see fig 2 of paper
    '''
    def __init__(self, model, c, lr, momentum, steps, device='cuda'):
        super(CWInfAttack, self).__init__()

        self.model = model
        self.c = c
        self.lr = lr
        self.steps = steps
        self.device = device
        self.momentum = momentum
        self.Normalize = torch_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        dummy_labels = torch.zeros(images.shape[0]).to(self.device)
        w = self.get_init_w(images).detach()
        w.requires_grad = True
        images.requires_grad = False

        tau = 1

        original_output = None
        best_adv_images = images.clone().detach()
        best_output_x = None
        best_acc = 0
        best_delta = 1

        optimizer = torch.optim.SGD([w], lr=self.lr, momentum=self.momentum)

        for step in range(self.steps):
            adv_images = self.w_to_adv_images(w)
            output_x, output_c = self.model(self.Normalize(adv_images))
            if original_output is None:
                original_output = output_x

            f_value = self.c * self.get_f_value(output_c)
            delta = self.w_to_delta(w, images)
            distance = self.inf_distance(delta, tau)
            loss = f_value + distance

            # update tau
            if torch.max(delta) < tau:
                tau = 0.9 * tau

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # print out results
            acc = cal_accuracy(output_c, dummy_labels)
            avg_delta = torch.mean(delta)
            print('Acc: {}\tDelta: {}'.format(acc, avg_delta))
            if acc > best_acc:
                best_adv_images = adv_images
                best_output_x = output_x
                best_acc = acc
                best_delta = avg_delta
            if acc == best_acc and avg_delta < best_delta:
                best_adv_images = adv_images
                best_output_x = output_x
                best_acc = acc
                best_delta = avg_delta
            if acc == 1:
                break
        print('Batch finished: Acc: {}\tDelta: {}'.format(best_acc, best_delta))
        print('>>>>>')
        pickle.dump(best_adv_images, open('adv_images_batch.pkl', 'wb'))
        clear_debug_image()
        save_image_stack(images, 'original input')
        save_image_stack(best_adv_images, 'adversarial input')
        save_image_stack(original_output, 'original output')
        save_image_stack(best_output_x, 'adversarial output')
        # delta_image = torch.abs(best_adv_images - images)
        # print(torch.max(delta_image))
        # adjusted_delta = delta_image / torch.max(delta_image)
        # save_image_stack(adjusted_delta, 'adjusted delta')

        return best_adv_images, best_acc, best_delta

    def get_f_value(self, outputs):
        src_p = outputs[:]  # class 1
        target_p = 1 - outputs[:]  # class 0
        f6 = torch.relu(src_p - target_p)
        return f6

    def inf_distance(self, delta, tau):
        dist_vec = torch.relu(delta - tau)
        return torch.sum(dist_vec)

    def w_to_adv_images(self, w):
        return 1/2 * (torch.tanh(w) + 1)

    def w_to_delta(self, w, x):
        return torch.abs(self.w_to_adv_images(w) - x)

    def get_init_w(self, x):
        return torch.atanh(2 * x - 1)
