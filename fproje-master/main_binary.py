import dataprep_binary as dpb
import matplotlib.pyplot as plt
import torch
import train_binary as tb
import predict_test_binary as ptb
DIR = "/mnt/data/data/summer_2018/resized_target_files/"
annots, images, users = dpb.load_from_csv()
labels, label_map = dpb.all_labels(annots)
sample = dpb.create_sample(annots, images, label_map, DIR)
dataloader = dpb.load_transformed_data(label_map, DIR)

device = torch.device('cuda')
print("*************************************************************")
labels, label_map = dpb.selected_labels()
labels_bin, labels_map_bin =dpb.selected_labels_bin()
#print(label_map)
print(labels)
dataloaders, dataset_sizes = dpb.selected_dataset(labels, label_map,labels_bin,labels_map_bin,DIR,0.8,0.1,0.1)

#print(dataset_sizes)
print(dataloaders)
model_b = tb.create_model()
model_b = tb.model_training(model_b, device,dataloaders, dataset_sizes)
#ptb.predict_rotate_test2(dataloaders, dataset_sizes, model_b,device)
