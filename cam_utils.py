from torchvision.transforms import functional as transforms_F
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

def label_data(coco: COCO, category='person'):
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

        image = transforms_F.resize(image, [new_h, new_w])

        pad_right = self.target_size - new_w
        pad_bottom = self.target_size - new_h

        padded = transforms_F.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
        return padded
    
class ImageListDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None):
        self.data_list = data_list
        self.transform = transform or ToTensor()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name, label = self.data_list[idx]
        image = Image.open(f'{self.data_dir}/{file_name}').convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def overlay_cam_with_centroid(image, cam, center, alpha=0.5):
    """
    image: numpy array [H, W, 3], values in [0, 1]
    cam: numpy array [H, W], values in [0, 1]
    center: tuple (x, y) in pixel coordinates
    alpha: transparency of heatmap
    """
    fig, ax = plt.subplots()

    # Show the original image
    ax.imshow(image)

    cam = cam.unsqueeze(0).unsqueeze(0)  # e.g., [1, 1, 14, 14]
    cam_resized = F.interpolate(cam, size=image.size, mode='bilinear', align_corners=False)
    cam_resized = cam_resized.squeeze()  # back to [H, W]

    cam_detached = cam_resized.cpu().detach().numpy()

    # Overlay the CAM
    heatmap = ax.imshow(cam_detached.T, cmap='jet', alpha=alpha)

    # Add colorbar if you want
    plt.colorbar(heatmap, ax=ax)

    # Mark the centroid
    x_com, y_com = center
    ax.plot(x_com, y_com, 'wo')  # white circle
    ax.plot(x_com, y_com, 'r+', markersize=12, markeredgewidth=2)  # red cross

    ax.set_title("CAM Overlay with Centroid")
    ax.axis('off')
    plt.show()
