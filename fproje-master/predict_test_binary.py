from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, utils
import ImageDataset as ds
import torch
import numpy as np
import imutils
import cv2

def predict_test(dataloaders,dataset_sizes, model,device):
    model = model.to(device)
    model.eval()
    predicted_labels = []
    true_labels = []
    # cm = np.zeros((5,5))
    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            labels = batch['documents'].to(device)
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)


        predicted_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    #print(predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
    # pre = precision_score(true_labels, predicted_labels)
    # rec = recall_score(true_labels, predicted_labels)
    # f1 = f1_score(true_labels, predicted_labels)
    print(cm)
    print("acc",acc)
    # print("pre",pre)
    # print("rec",rec)
    # print("f1", f1)
    fo = open("acc_binary.txt", "a")

    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()

def predict_rotated_test(dataloaders,dataset_sizes, model,device):
    model = model.to(device)
    model.eval()
    predicted_labels = []
    true_labels = []
    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            outputs = model(inputs)
            for i in range(3):

                inputs = inputs.cpu().numpy()
                inputs = rotate_images(inputs)
                inputs = torch.from_numpy(inputs).to(device)
                output = model(inputs)
                outputs = np.hstack((outputs, output))
            #print(outputs.shape)
            outputs = torch.from_numpy(outputs).to(device)

            labels = batch['documents'].to(device)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        preds = preds%2
        #print(preds)


        predicted_labels.extend(preds)
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)


def predict_rotated_test2(dataloaders,dataset_sizes, model,device):
    model = model.to(device)
    model.eval()
    predicted_labels = []
    true_labels = []
    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            outputs = model(inputs)
            for i in range(3):

                inputs = inputs.cpu().numpy()
                inputs = rotate_images(inputs)
                inputs = torch.from_numpy(inputs).to(device)
                output = model(inputs)
                outputs += output
            #print(type(outputs))
            #outputs = torch.from_numpy(outputs).to(device)

            labels = batch['documents'].to(device)
        _, preds = torch.max(outputs, 1)

        predicted_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)

    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()

def rotate_images(inputs):
    images = inputs.transpose((0, 2, 3, 1))
    for i in range(images.shape[0]):
        images[i,:,:,:] = imutils.rotate(images[i,:,:,:], 90)
    images = images.transpose((0,3 ,1 ,2 ))
    return images
