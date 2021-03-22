import os.path as osp
import numpy as np
from PIL import Image
from common.nets.resnet import ResNetBackbone


def load_images():
    assets_path = osp.join('..', 'assets', 'input.jpg')

    img = Image.open(assets_path).convert('RGB').resize((500, 500), Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32)
    return img


if __name__ == '__main__':
    img = load_images()

    model = ResNetBackbone(50)
    output = model(img[None, :, :], is_training=True)
    print(output)
