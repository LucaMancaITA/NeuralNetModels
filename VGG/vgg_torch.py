
import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    def __init__(
        self, in_channels, filters, kernel_size, repetitions,
        pool_size=2, strides=2
    ):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        vars(self)[f"conv2D_{0}"] = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding="same"
            )
        for i in range(1, self.repetitions):
            vars(self)[f"conv2D_{i}"] = nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding="same"
            )

        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=strides
        )

    def forward(self, input_tensor):
        conv2D_0 = vars(self)["conv2D_0"]
        x = conv2D_0(input_tensor)
        x = self.activation(x)

        for i in range(1, self.repetitions):
            conv2D_i = vars(self)[f"conv2D_{i}"]
            x = conv2D_i(x)
            x = self.activation(x)


        x = self.max_pool(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(VGG, self).__init__()
        self.block_a = Block(
            in_channels=num_channels,
            filters=64,
            kernel_size=3,
            repetitions=2
        )
        self.block_b = Block(
            in_channels=64,
            filters=128,
            kernel_size=3,
            repetitions=2
        )
        self.block_c = Block(
            in_channels=128,
            filters=256,
            kernel_size=3,
            repetitions=3
        )
        self.block_d = Block(
            in_channels=256,
            filters=512,
            kernel_size=3,
            repetitions=3
        )
        self.block_e = Block(
            in_channels=512,
            filters=512,
            kernel_size=3,
            repetitions=3
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        x = self.block_a(input_tensor)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":

    img_size = 224
    num_channels = 3
    num_classes = 1000
    x = torch.randn((1, num_channels, img_size, img_size))
    print(f"\nInput shape: {x.shape}")
    model = VGG(num_classes=num_classes, num_channels=num_channels)
    y = model(x)
    print(f"Output shape: {y.shape}\n")
    print(summary(model, (3, 224, 224)))
