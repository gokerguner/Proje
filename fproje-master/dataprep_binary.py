import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from skimage import io, transform
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import ImageDataset as ds
import os
import torchvision
import warnings
warnings.filterwarnings("ignore")


def load_from_csv():
    annots = pd.read_csv("annotations.csv")
    images = pd.read_csv("images.csv")
    users = pd.read_csv("users.csv")
    return annots, images, users

def all_labels(annots):
    labels = annots.annotation_value.unique()
    label_map = {label : np.array(i) for i, label in enumerate(labels)}
    return labels, label_map

def create_sample(annots, images, label_map,dir_):
    data = annots[annots['annotation_user_id'] > 6]
    #print(data['annotation_value'].value_counts())
    ims_with_annots = images.merge(data, on='id')[['file_name','annotation_value']]
    strip = lambda x : x.lstrip(" u'").rstrip("'")
    ims_with_annots['annotation_value'] = ims_with_annots['annotation_value'].map(strip)
    sample_names = os.listdir(dir_)
    sample_df = ims_with_annots[ims_with_annots['file_name'].isin(sample_names)]

    #print(sample_df['annotation_value'].value_counts())
    sample_df.to_csv('loader.csv',index=None)
    sample = ds.ImageDataset('loader.csv','images',label_map = label_map)

    return sample

def load_transformed_data(label_map, dir_):
    transformed_dataset = ds.ImageDataset(csv_file='loader.csv', root_dir=dir_, label_map = label_map,
                                           transform=transforms.Compose([ds.RandomCrop(224), ds.RandomRotate(), ds.ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=64,shuffle=True, num_workers=4)
    return dataloader

def show_documents_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, documents_batch = sample_batched['image'], sample_batched['documents']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def selected_labels():
    labels = ['receipt','invoice','inforeceipt','fisandslip','slip']
    label_map = {label : np.array(i) for i, label in enumerate(labels)}
    return labels, label_map

def selected_labels_bin():
    labels_bin = ['receipt','inforeceipt']
    label_map_bin = {label : np.array(i) for i, label in enumerate(labels_bin)}
    return labels_bin, label_map_bin

def selected_dataset(labels,label_map,labels_bin, labels_map_bin, dir_,train_percent,val_percent,test_percent):
    #bin_labels = ['receipt','inforeceipt']
    sampleset = pd.read_csv('loader.csv')
    selected_samples = sampleset[sampleset['annotation_value'].isin(labels)]
    selected_samples.to_csv('selected.csv',index=None)

    selected_data = pd.read_csv("selecteds.csv")
    trains = selected_data[0:int(len(selected_data) * train_percent)]
    trains.to_csv("trains.csv",index=False)
    train_read_bin = pd.read_csv("trains.csv")
    selected_train_bin = train_read_bin[train_read_bin['annotation_value'].isin(labels_bin)]
    selected_train_bin.to_csv('trains_binary.csv',index=None)


    vals = selected_data[int(len(selected_data) * train_percent):int(len(selected_data) * train_percent)+int(len(selected_data) * val_percent)]
    vals.to_csv("vals.csv",index=False)
    vals_read_bin = pd.read_csv('vals.csv')
    selected_vals_bin = vals_read_bin[vals_read_bin['annotation_value'].isin(labels_bin)]
    selected_vals_bin.to_csv('vals_binary.csv',index=None)

    tests = selected_data[int(len(selected_data) * train_percent)+int(len(selected_data) * val_percent) :
                            int(len(selected_data) * train_percent)+int(len(selected_data) * val_percent) + int(len(selected_data) * test_percent)]
    tests.to_csv("tests.csv",index=False)
    tests_read_bin = pd.read_csv('tests.csv')
    selected_tests_bin = tests_read_bin[tests_read_bin['annotation_value'].isin(labels_bin)]
    selected_tests_bin.to_csv('tests_binary.csv',index=None)


    transform = transforms.Compose([ds.RandomCrop(224), ds.RandomRotate(),ds.ToTensor()])
    image_datasets = {x[0]: ds.ImageDataset(x[1],dir_, labels_map_bin, transform=transform)
                        for x in [['train','trains_binary.csv'],['val','vals_binary.csv'],['test','tests_binary.csv']]}
    dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=64,shuffle=True,num_workers=4)
                      for x in ['train','val','test']}
    dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val','test']}
    print(dataset_sizes)
    return dataloaders, dataset_sizes
