import glob
import os
from collections import namedtuple

import PIL
import numpy
from PIL import Image
from paddle.io import Dataset
from paddle.vision.transforms import Compose, ColorJitter, Resize, CenterCrop, RandomHorizontalFlip, ToTensor, \
    RandomResizedCrop, Normalize

import paddle

from config import cfg

Datasets = namedtuple('Dataset', ['classes', 'mean', 'std',
                                  'eigval', 'eigvec', 'name'])

imagenet = Datasets(name='imagenet',
                    classes=1000,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    eigval=[55.46, 4.794, 1.148],
                    eigvec=[[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]])


def get_data(data_name):
    try:
        return globals()[data_name]
    except KeyError:
        return None


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = numpy.array(eigval)
        self.eigvec = numpy.array(eigvec)

    def __call__(self, image, *args):
        if self.alphastd == 0:
            return (image, *args)
        alpha = numpy.random.randn(3) * self.alphastd
        rgb = (self.eigvec @ numpy.diag(alpha * self.eigval)).sum(axis=1). \
            round().astype(numpy.int32)
        image = numpy.asarray(image)
        image_type = image.dtype
        image = Image.fromarray(
            numpy.clip(image.astype(numpy.int32) + rgb, 0, 255).astype(image_type))
        return (image, *args)


class ImageNet(Dataset):
    def __init__(self, cfg, train=True):
        super(ImageNet, self).__init__()
        self.cfg = cfg
        self.train = train

        self.data_infor = get_data(cfg.Data.dataset)
        self.traindir = os.path.join(cfg.Data.Dir, 'train')
        self.valdir = os.path.join(cfg.Data.Dir, 'val')
        self.catedict = dict(zip(sorted(os.listdir(self.valdir)[:1000]), range(1000)))

        # transform
        # assured inumpyut is CHW
        self.normalize = Normalize(mean=self.data_infor.mean, std=self.data_infor.std, data_format='CHW')
        self.transform_train = [RandomResizedCrop(cfg.Trans.crop_size,
                                                  scale=(cfg.Trans.min_area_ratio, 1.0),
                                                  ratio=(3. / 4, cfg.Trans.aspect_ratio))]

        if self.data_infor.eigval is not None and self.data_infor.eigvec is not None \
                and cfg.Trans.random_color:
            lighting = Lighting(0.1, self.data_infor.eigval, self.data_infor.eigvec)
            jitter = ColorJitter(0.4, 0.4, 0.4)
            self.transform_train.extend([jitter, lighting])
        self.transform_train.extend([RandomHorizontalFlip(),
                                     ToTensor(),
                                     self.normalize])
        self.transform_train = Compose(self.transform_train)
        self.transform_val = Compose([Resize(cfg.Trans.scale_size), CenterCrop(cfg.Trans.crop_size),
                                      ToTensor(), self.normalize])

        self.file_list = self.get_samples()

    def get_samples(self):
        if self.train:
            txtpath = os.path.join(self.cfg.Data.Dir, "train.txt")
            self.dir = self.traindir
        else:
            txtpath = os.path.join(self.cfg.Data.Dir, "val.txt")
            self.dir = self.valdir

        file_list = []
        if not os.path.exists(txtpath):
            with open(txtpath, 'w') as f:
                files = glob.glob(self.dir + '/**/*.JPEG',
                                  recursive=True)  # glob.glob('/root/tpp/paddle/data/train/**/*.jpeg', recursive=True)
                for ele in files:
                    f.write(ele + '\n')
                    file_list.append((ele, self.catedict[ele.split('/')[-2]]))
        else:
            with open(txtpath, 'r') as f:
                for line in f:
                    image_path = line.strip()
                    label = self.catedict[image_path.split('/')[-2]]
                    file_list.append((image_path, label))

        return file_list

    def __getitem__(self, index):
        image_path, label = self.file_list[index]
        image = PIL.Image.open(image_path).convert('RGB')
        # print('image.size', image.size, image_path)

        if self.train:
            image = self.transform_train(image)
        else:
            image = self.transform_val(image)

        return image, label

    def __len__(self):
        return len(self.file_list)


def get_dataloader(cfg):
    train_ds = ImageNet(cfg, train=True)
    val_ds = ImageNet(cfg, train=False)

    train_loader = paddle.io.DataLoader(train_ds, batch_size=cfg.Train.batch_size,
                                        shuffle=True, num_workers=cfg.Train.num_workers)
    val_loader = paddle.io.DataLoader(val_ds, batch_size=cfg.Train.batch_size,
                                      shuffle=False, num_workers=cfg.Train.num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    tl, vl = get_dataloader(cfg)

    # for data, label in ts:
    #     print('ts', data, label, data.shape)
    #     break

    for data, label in tl:
        print('tl', data, label, data.shape)
        break

    for data, label in vl:
        print('vl', data, label, data.shape)
        break

# def get_model_url(data, name):
#     return join(WEB_ROOT, data.name,
#                 '{}-{}.pth'.format(name, data.model_hash[name]))
# http://dl.yf.io/dla/models/imagenet/dla60/24839fc4
