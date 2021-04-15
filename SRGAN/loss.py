
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

gen_loss = nn.BCELoss()
vgg_loss = nn.MSELoss()
mse_loss = nn.MSELoss()


class TVLoss(nn.Module):
    # Total variation loss
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__():
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial loss: min log(1 - D(G(z))) -> max log(D(G(z)))
        adversarial_loss = torch.mean(1 - out_labels) #adv_loss = gen_loss(gen_out, torch.ones_like(gen_out))
        # Content loss
        content_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * content_loss + 2e-8 * tv_loss