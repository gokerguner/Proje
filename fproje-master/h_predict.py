import dataprep as dp
import dataprep_binary as dp2
import matplotlib.pyplot as plt
import torch
import train_dense as t
import train_resnet50 as t3
import train_binary as tb
import predict_test as p
DIR = "/mnt/data/data/summer_2018/resized_target_files"
annots, images, users = dp.load_from_csv()
labels, label_map = dp.all_labels(annots)
sample = dp.create_sample(annots, images, label_map, DIR)
dataloader = dp.load_transformed_data(label_map, DIR)


#labels2, label_map2 = dp2.all_labels(annots)
#sample2 = dp2.create_sample(annots, images, label_map, DIR)
#dataloader = dp2.load_transformed_data(label_map, DIR)

device = torch.device('cuda')
print("*************************************************************")
labels, label_map = dp.selected_labels()
#labels2, label_map2 = dp2.selected_labels()

#print(label_map)
#print(labels)
dataloaders, dataset_sizes = dp.selected_dataset(labels, label_map,DIR,0.8,0.1,0.1)
#dataloaders2, dataset_sizes2 = dp2.selected_dataset(labels, label_map,DIR,0.8,0.1,0.1)

model = t.create_model()
trained_model_weights = torch.load('/mnt/data/data/summer_2018/model_dense_modified.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(trained_model_weights)

model2 = t3.create_model()
trained_model_weights2 = torch.load('/mnt/data/data/summer_2018/model_resnet50.pth', map_location=lambda storage, loc: storage)
model2.load_state_dict(trained_model_weights2)


model_b = tb.create_model()
trained_model_weights_b = torch.load('/mnt/data/data/summer_2018/model_resnet_binary.pth', map_location=lambda storage, loc: storage)
model_b.load_state_dict(trained_model_weights_b)

p.predict_rotated_test2_multimodel2(dataloaders, dataset_sizes, model,model2,model_b,device)
