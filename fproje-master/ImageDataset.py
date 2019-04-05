from __future__ import print_function, division
from PIL import Image
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import argparse
import imutils
import cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_map, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.documents_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        #self.labels = {u'receipt' : 0 , u'invoice' : 1, u'inforeceipt' : 2, u'background' : 3, u'fisandslip' : 4, u'slip' : 5, u'hack' : 6, u'discard' : 7, u'multi' : 8}
        #self.label_map=dict()
        #for i ,label in enumerate(self.documents_frame['annotation_value'].unique()):
        #    self.label_map[label]=np.array(i)

    def __len__(self):
        return len(self.documents_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.documents_frame.iloc[idx, 0])
        image = io.imread(img_name)
        documents = np.array(self.label_map[self.documents_frame.iloc[idx]['annotation_value']])
        #documents = self.documents_frame.iloc[idx, 1]
        sample = {'image': image, 'documents': documents}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, documents = sample['image'], sample['documents']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively


        return {'image': img, 'documents': documents}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, documents = sample['image'], sample['documents']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'documents': documents}


class ToTensor(object):

    def __call__(self, sample):
        image, documents = sample['image'], sample['documents']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = color.rgb2gray(image)
        image = color.gray2rgb(image)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'documents': torch.from_numpy(documents)}


class RandomRotate(object):
    def __call__(self, sample):

        image, documents = sample['image'], sample['documents']
        image = imutils.rotate(image, 90*(np.random.randint(0, 4)))

        return {'image': image, 'documents': documents}
