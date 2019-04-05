import dataprep as dp
import matplotlib.pyplot as plt
import torch
import train_resnet50 as t3
import predict_test as p
DIR = "/mnt/data/data/summer_2018/resized_target_files"
annots, images, users = dp.load_from_csv()
labels, label_map = dp.all_labels(annots)
sample = dp.create_sample(annots, images, label_map, DIR)
dataloader = dp.load_transformed_data(label_map, DIR)

device = torch.device('cuda')
print("*************************************************************")
labels, label_map = dp.selected_labels()
#print(label_map)
print(labels)
dataloaders, dataset_sizes = dp.selected_dataset(labels, label_map,DIR,0.8,0.1,0.1)
model = t3.create_model()
trained_model_weights = torch.load('/mnt/data/data/summer_2018/model_resnet50.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(trained_model_weights)
p.predict_rotated_test2(dataloaders, dataset_sizes, model,device)
