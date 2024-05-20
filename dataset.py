from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class Compose(object):
    def __init__(self, base_size, crop_size, resize_scale_range, flip_ratio=0.5):
        self.base_size = base_size
        self.crop_size = crop_size
        self.resize_scale_range = resize_scale_range
        self.flip_ratio = flip_ratio

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        short_size = random.randint(int(self.base_size * self.resize_scale_range[0]),
                                    int(self.base_size * self.resize_scale_range[1]))
        ow, oh = short_size, short_size
        img, mask = img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        img = np.array(img)
        mask = np.array(mask)
        num_crop = 0
        while num_crop < 5:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            endx = x + self.crop_size
            endy = y + self.crop_size
            patch = img[y:endy, x:endx]
            if (patch == 0).all():
                continue
            else:
                break
        img = img[y:endy, x:endx]
        mask = mask[y:endy, x:endx]
        img, mask = Image.fromarray(img), Image.fromarray(mask)
        if random.random() < self.flip_ratio:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask



class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

mask_transform = MaskToTensor()

resize_scale_range = [float(scale) for scale in [0.5,2]]
sync_transforms = Compose(512, 512, resize_scale_range)

class RSDataset(Dataset):
    def __init__(self, class_name,root=None, mode=None, img_transform=img_transform, mask_transform=mask_transform, sync_transforms=sync_transforms):
        # 数据相关
        self.class_names = class_name
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        img_dir = os.path.join(root, 'rgb')
        mask_dir = os.path.join(root, 'label')

        for img_filename in os.listdir(img_dir):
            img_mask_pair = (os.path.join(img_dir, img_filename),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
        random.shuffle(self.sync_img_mask)

    def __getitem__(self, index):
        img_path, mask_path = self.sync_img_mask[index]
        with Image.open(img_path).convert('RGB') as img, Image.open(mask_path).convert('L') as mask:
            if self.mode == 'train':
                img, mask = self.sync_transform(img, mask)
            if self.img_transform:
                img = self.img_transform(img)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.sync_img_mask)