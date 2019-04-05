from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import sampler
import os
import time
import copy
import torch

def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    fo = open("acc.txt", "a")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        fo.write('Epoch ' + str(epoch) + "/" + str(num_epochs - 1) + "\n")
        fo.write('----------' + "\n")
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n = 0
            for batch in dataloaders[phase]:
                inputs = batch['image'].float().to(device)
                labels = batch['documents'].to(device)
                #print(labels)
                print("batch", n)
                n = n + 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print("out", outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            fo.write(str(phase) + " Loss: " + str(epoch_loss) + "  Acc: "+ str(epoch_acc.item()) + "\n")
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
		torch.save(best_model_wts, '/mnt/data/data/summer_2018/model_resnet50.pth')
		print("model has been saved")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    fo.write('Training complete in ' + str(time_elapsed // 60) + 'm ' +  str(time_elapsed % 60) + 's' + "\n")
    fo.write('Best val Acc: ' + str(best_acc) + "\n")
    # load best model weights
    model.load_state_dict(best_model_wts)
    fo.close()
    return model

def create_model():
    model_ft = models.resnet50(pretrained=True)
    #num_ftrs = model_ft.classifier.in_features
    #model_ft.classifier = nn.Linear(num_ftrs,5)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5)

    return model_ft

def model_training(model_ft, device, dataloaders, dataset_sizes):
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss(weight = torch.cuda.FloatTensor([0.02, 0.037, 0.1, 0.24, 0.38]))
    #criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=100)
    return model_ft
