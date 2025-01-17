import paddle
from paddle.io import Dataset
from PIL import Image
import os
import numpy as np
import random
import albumentations as A

SIZE = 256
def read_path_test(path, name: str):
    folder_key = {
        'ECSSD': ['Imgs', 'Gt'], 'SOD': ['images', 'gt'],
        'PASCAL-S': ['Imgs', 'Gt'], 'HKU-IS': ['imgs', 'gt'],
        'DUT-OMRON': ['Imgs', 'Gt'], 'DUTS-TE': ['DUTS-TE-Image', 'DUTS-TE-Mask']
    }

    image_root = os.path.join(path, folder_key[name][0])
    mask_root = os.path.join(path, folder_key[name][1])

    image_names = os.listdir(image_root)
    image_paths = [os.path.join(image_root, n) for n in image_names]
    mask_paths = [os.path.join(mask_root, n.split('.')[0]+'.png') for n in image_names]

    return image_paths, mask_paths

def random_clip_with_bbox(img: np.ndarray, mask: np.ndarray, bbox, offset=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    if offset != 0:
        x1, y1, x2, y2 = max(0, x1-offset), max(0, y1-offset), \
            min(w, x2+offset), min(h, y2+offset)

    x1rd, y1rd = random.randint(0, x1), random.randint(0, y1)
    x2rd, y2rd = random.randint(x2, w), random.randint(y2, h)

    _img = img[y1rd:y2rd, x1rd:x2rd, :]
    _mask = mask[y1rd:y2rd, x1rd:x2rd]
    # _bbox = (bbox[0]-x1rd, bbox[1]-y1rd, bbox[2]-x1rd, bbox[3]-y1rd)

    return _img, _mask

class DUTS_TR(Dataset):
    def __init__(self, path="./dataset/DUTS"):
        super().__init__()

        self.path = path
        self.data = [line.strip().split(",") for line in open(os.path.join(path, 'DUTS-TR', 'bboxes.txt'), 'r').readlines()]
    
        self.transforms = A.Compose([
            A.HorizontalFlip(),
            # A.Rotate(15),
            # A.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # A.Sharpen(),
            A.Resize(SIZE, SIZE)
        ])

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, 'DUTS-TR', 'DUTS-TR-Image', self.data[idx][0]+".jpg")
        mask_path = os.path.join(self.path, 'DUTS-TR', 'DUTS-TR-Mask', self.data[idx][0]+".png")

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        bbox = [int(c) for c in self.data[idx][1:]]

        if len(mask.shape) != 2:
            mask = mask[:, :, 0]

        _img, _mask = img, mask
        # _img, _mask = random_clip_with_bbox(_img, _mask, bbox)
        augmented = self.transforms(image=_img, mask=_mask)
        img, mask = augmented['image'], augmented['mask']

        # to numpy array and normalize
        img_arr = np.array(img).transpose(2, 0, 1) / 255
        img_arr = (img_arr - 0.5) / 0.5
        gt_arr = np.array(mask) / 255
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.data)


class PreLoadedDUTS_TR(Dataset):
    def __init__(self, path="./dataset/DUTS"):
        super().__init__()

        self.path = path
        self.data = [line.strip().split(",") for line in open(os.path.join(path, 'DUTS-TR', 'bboxes.txt'), 'r').readlines()]
        self.imgs, self.labels = self._load_all_images_and_labels()
        
        self.transforms = A.Compose([
            A.HorizontalFlip(),
            # A.Rotate(15),
            # A.ColorJitter(0.3, 0.3, 0.3, 0.3),
            # A.Sharpen()
        ])
        
    def _load_all_images_and_labels(self):
        imgs, labels = [], []
        print("-- loading images and labels ...")
        for i in range(len(self.data)):
            img_path = os.path.join(self.path, 'DUTS-TR', 'DUTS-TR-Image', self.data[i][0]+".jpg")
            mask_path = os.path.join(self.path, 'DUTS-TR', 'DUTS-TR-Mask', self.data[i][0]+".png")
            img = np.array(Image.open(img_path).resize((SIZE, SIZE)), dtype=np.uint8)
            mask = np.array(Image.open(mask_path).resize((SIZE, SIZE)), dtype=np.uint8)
            if len(mask.shape) != 2:
                mask = mask[:, :, 0]
            imgs.append(img)
            labels.append(labels)
        print("-- loading complete")
        return imgs, labels
            
    def __getitem__(self, idx):
        img, mask = self.imgs[idx], self.labels[idx]
        bbox = [int(c) for c in self.data[idx][1:]]

        _img, _mask = img, mask
        # _img, _mask = random_clip_with_bbox(_img, _mask, bbox)
        augmented = self.transforms(image=_img, mask=_mask)
        img, mask = augmented['image'], augmented['mask']

        # to numpy array and normalize
        img_arr = np.array(img).transpose(2, 0, 1) / 255
        img_arr = (img_arr - 0.5) / 0.5
        gt_arr = np.array(mask) / 255
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.data)


class DATASET_TEST(Dataset):
    def __init__(self, path, name):
        super().__init__()
        self.path = path
        self.name = name
        self.image_list, self.mask_list = read_path_test(path, name)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        gt = Image.open(self.mask_list[idx])
        h, w = img.size[1], img.size[0]
        img = img.resize((SIZE, SIZE))
        # to numpy array and normalize
        if len(np.array(img).shape) == 3:
            img_arr = np.array(img).transpose(2, 0, 1) / 255
        else:
            _img_arr = np.array(img).reshape((1, SIZE, SIZE))
            img_arr = np.concatenate([_img_arr, _img_arr, _img_arr], axis=0) / 255
        img_arr = (img_arr-0.5) / 0.5
        gt_arr = np.array(gt) / 255
        
        if len(gt_arr.shape) != 2:
            gt_arr = gt_arr[:, :, 0]
        
        # to paddle tensor
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        # img, label, h, w
        return img_tensor, gt_tensor, h, w

    def __len__(self):
        return len(self.image_list)
