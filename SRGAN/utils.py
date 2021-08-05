
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, CenterCrop

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])