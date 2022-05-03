
import torch
import torch.nn as nn
from torchsummary import summary


class IdentityBlock(nn.Module):
    def __init__(self, in_channels=64, filters=64, kernel_size=3):
        super(IdentityBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            padding="same"
        )
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            padding="same"
        )
        self.bn2 = nn.BatchNorm2d(filters)

        self.act = nn.ReLU()

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x += input_tensor
        x = self.act(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            padding="same"
        )
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3)

        self.idla = IdentityBlock(in_channels=64, filters=64, kernel_size=3)
        self.idlb = IdentityBlock(in_channels=64, filters=64, kernel_size=3)

        self.global_pool = nn.AvgPool2d(kernel_size=74)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.idla(x)
        x = self.idlb(x)

        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    img_size = 224
    num_classes = 10
    x = torch.randn((1, 3, img_size, img_size))
    print(f"\nInput shape: {x.shape}")
    model = ResNet(num_classes=num_classes)
    y = model(x)
    print(f"Output shape: {y.shape}\n")
    print(summary(model, (3, 224, 224)))
