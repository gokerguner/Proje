from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, utils
import ImageDataset as ds
import torch
import numpy as np
import imutils
import cv2
import torch.nn.functional as F

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
    fo = open("acc.txt", "a")

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
        preds = preds%5
        #print(preds)


        predicted_labels.extend(preds)
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)
    fo = open("acc.txt", "a")
    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()


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

    fo = open("acc.txt", "a")
    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()




def predict_rotated_test2_multimodel(dataloaders,dataset_sizes, model,model2,device):
    model = model.to(device)
    model.eval()

    model2 = model2.to(device)
    model2.eval()

    predicted_labels = []
    true_labels = []

    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            outputs = model(inputs)
            outputs2 = model2(inputs)

            for i in range(3):

                inputs = inputs.cpu().numpy()
                inputs = rotate_images(inputs)
                inputs = torch.from_numpy(inputs).to(device)

                output = model(inputs)
                output = F.softmax(output, dim=1)
                outputs += output

                output2 = model2(inputs)
                output2 = F.softmax(output2, dim=1)

                outputs2 += output2

                #print(output)
            outputs = outputs.cpu().numpy()
            #print(outputs.shape)
            outputs2 = outputs2.cpu().numpy()

            #outputs = outputs/np.sum(outputs, axis = 1).reshape(outputs.shape[0],1)
            #outputs2 = outputs2/np.sum(outputs2, axis = 1).reshape(outputs.shape[0],1)

            outputs = outputs + outputs2
            outputs = torch.from_numpy(outputs).to(device)


            #print(outputs)
            labels = batch['documents'].to(device)

        _, preds = torch.max(outputs, 1)

        predicted_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)

    fo = open("acc.txt", "a")
    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()

    grp = open("grp.txt", "a")
    grp.write(str(np.hstack((true_labels, predicted_labels))))
    #grp.write(str(true_labels) + "\n")
    #grp.write(str(predicted_labels))
    grp.close()

def predict_rotated_test_multimodel(dataloaders,dataset_sizes, model,model2,device):
    model = model.to(device)
    model.eval()

    model2 = model2.to(device)
    model2.eval()

    predicted_labels = []
    true_labels = []

    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            outputs = model(inputs)
            outputs2 = model2(inputs)

            for i in range(3):

                inputs = inputs.cpu().numpy()
                inputs = rotate_images(inputs)
                inputs = torch.from_numpy(inputs).to(device)

                output = model(inputs)
                output = F.softmax(output, dim=1)
                outputs += output

                output2 = model2(inputs)
                output2 = F.softmax(output2, dim=1)

                outputs2 += output2

                #print(output)
            outputs = outputs.cpu().numpy()
            #print(outputs.shape)
            outputs2 = outputs2.cpu().numpy()

            #outputs = outputs/np.sum(outputs, axis = 1).reshape(outputs.shape[0],1)
            #outputs2 = outputs2/np.sum(outputs2, axis = 1).reshape(outputs.shape[0],1)

            #outputs = outputs + outputs2
            outputs = np.hstack((outputs, outputs2))

            outputs = torch.from_numpy(outputs).to(device)


            #print(outputs)
            labels = batch['documents'].to(device)

        _, preds = torch.max(outputs, 1)

        preds = preds.cpu().numpy()
        preds = preds%5
                #print(preds)
        predicted_labels.extend(preds)
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)

    fo = open("acc.txt", "a")

    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()
    grp = open("grp.txt",'a')
    grp.write(str(true_labels) + "\n")
    grp.write(str(predicted_labels) + "\n")
    grp.close()  

def predict_rotated_test2_multimodel2(dataloaders,dataset_sizes, model,model2,model3,device):
    model = model.to(device)
    model.eval()

    model2 = model2.to(device)
    model2.eval()

    model3 = model3.to(device)
    model3.eval()

    predicted_labels = []
    true_labels = []

    for batch in dataloaders['test']:
        with torch.no_grad():
            inputs = batch['image'].float().to(device)
            outputs = model(inputs)
            outputs2 = model2(inputs)
            outputs3 = model3(inputs)

            for i in range(3):

                inputs = inputs.cpu().numpy()
                inputs = rotate_images(inputs)
                inputs = torch.from_numpy(inputs).to(device)

                output = model(inputs)
                output = F.softmax(output, dim=1)
                outputs += output

                output2 = model2(inputs)
                output2 = F.softmax(output2, dim=1)
                outputs2 += output2

                output3 = model3(inputs)
                output3 = F.softmax(output3, dim=1)
                outputs3 += output3


            outputs = outputs.cpu().numpy()
            outputs2 = outputs2.cpu().numpy()

            outputs = outputs + outputs2
            outputs = torch.from_numpy(outputs).to(device)


            #print(outputs)
            labels = batch['documents'].to(device)

        _, preds = torch.max(outputs, 1)
        _, preds_b = torch.max(outputs3, 1)

        preds = preds.cpu().numpy()
        preds_b = preds_b.cpu().numpy()*2

        #print(preds)
        mask1 = np.array(preds == 0)
        mask2 = np.array(preds == 2)
        #print("MASK1:", mask1)
        #print("MASK2:", mask2)
        mask12 = np.bitwise_or(mask1, mask2)
        #print(mask12.shape)
        #print(mask12)
        #break
        notmask12 = np.bitwise_not(mask12)

        preds = preds*notmask12 + preds_b*mask12


        predicted_labels.extend(preds)
        true_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)

    print(cm)
    print("acc",acc)

    fo = open("acc.txt", "a")
    fo.write(str(cm) + "\n")
    fo.write("Test accuracy:" + str(acc) + "\n")
    fo.close()


def rotate_images(inputs):
    images = inputs.transpose((0, 2, 3, 1))
    for i in range(images.shape[0]):
        images[i,:,:,:] = imutils.rotate(images[i,:,:,:], 90)
    images = images.transpose((0,3 ,1 ,2 ))
    return images
