import numpy as np
from skimage import transform
from torchvision.transforms import functional as F
from PIL import Image

def label_data(coco, category='person'):
    """
    returns:
        list[file_name, 0/1]
    """
    output = []

    cat_id = coco.getCatIds(catNms=[category])[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    yes_imgs = coco.loadImgs(ids=img_ids)
    output += [(im["file_name"], 1) for im in yes_imgs]

    no_target_ids_set = set(coco.getImgIds()) - set(img_ids)
    no_imgs = coco.loadImgs(ids=list(no_target_ids_set))
    output += [(im["file_name"], 0) for im in no_imgs]

    return output


class ResizeAndPad:
    def __init__(self, target_size=500):
        self.target_size = target_size

    def __call__(self, image: Image.Image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = F.resize(image, [new_h, new_w])
        # tensor = F.to_tensor(image)

        pad_right = self.target_size - new_w
        pad_bottom = self.target_size - new_h

        padded = F.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
        return padded
