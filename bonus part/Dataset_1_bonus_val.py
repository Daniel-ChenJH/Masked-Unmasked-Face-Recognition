#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/07/23 11:23:39
@Author  :   SWS SUMMERWORKSHOP GROUP 9
@Version :   1.0
@Description :   Bonus part of project 'Masked Unmasked Face Recognition'
'''
# Headers to be included:
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import transforms
import torchvision
import time
from torch.optim.lr_scheduler import ExponentialLR
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from skimage.feature import hog


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# read datasets;
def readfile(dataset_path):
    x = np.zeros((750, 150, 150, 3), dtype=np.uint8)
    y = np.zeros(750, dtype=np.uint8)
    j=0
    for subject_name in os.listdir(os.path.join(os.getcwd(),dataset_path)):
        subject_images_dir = os.path.join(dataset_path,subject_name)
        image_dir = sorted(os.listdir(subject_images_dir))
        for i, file_name in enumerate(image_dir):
            img = cv2.imread(os.path.join(subject_images_dir, file_name))
            x[j, :, :] = cv2.resize(img, (150, 150))
            y[j] = int(subject_name.replace("s",''))-1
            j+=1
    return x, y                

def read_test_file(transform,dataset_path,labels=[]):
    j=0
    f2=open('clf_classifier.model','rb')
    s2=f2.read()
    model1=pickle.loads(s2)
    num=len(os.listdir(dataset_path))
    x = np.zeros((num, 150, 150, 3), dtype=np.uint8)
    y=[]
    predict_class=[]
    if len(labels):
        for each_name in labels:
            y.append(int(each_name.replace('s',''))-1)
    else:
        for i in range(num):
            y.append(-1)
    for each_img in sorted(os.listdir(dataset_path)):
        img = cv2.imread(os.path.join(dataset_path,each_img))
        x[j, :, :] = cv2.resize(img, (150, 150))
        test_features=hog(img, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, multichannel=True)
        predict_class.append(model1.predict(test_features.reshape(1,-1)))
        j+=1
    
    image_set = ImgDataset(x, y, transform)
    val_loader = DataLoader(image_set,pin_memory=True)

    return predict_class,val_loader             

class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        # label is required to be a LongTensor
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        Y = self.y[index]
        return X, Y


if __name__ == "__main__":




    batch_size = 16    
    photosize=150

    # set up random seed
    setup_seed(1896)

    # Set up the dataset processing mode
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(photosize),  # resize
        transforms.CenterCrop(photosize), # crop
        transforms.ToTensor(), # to tensor
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  # do normalization to (-1,1), with mean and varience fixed
    ])



    # Import the preprocessed datasets
    print("Reading data")
    # processed_x, processed_y = readfile(os.path.join(workspace_dir, "processed_img"))
    # # masked_x, masked_y = readfile(os.path.join(workspace_dir, "masked_img"))
    # processed_X_train, processed_X_test, processed_y_train, processed_y_test = train_test_split(processed_x, processed_y, test_size=0.2, shuffle=True)
    # train_set = ImgDataset(processed_X_train, processed_y_train, transform)
    # val_set = ImgDataset(processed_X_test, processed_y_test, transform)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,pin_memory=True)

    # # train_set=datasets.CIFAR10('data/',download=True, train=True, transform=transform)
    # # val_set=datasets.CIFAR10('data/',download=True, train=False, transform=transform)
    
    # # train_set, val_set,leaveout= torch.utils.data.random_split(dataset= val_set, lengths=[2000, 500, 7500])
    # print("Size of training data = {}".format(len(processed_X_train)))
    # print("Size of validation data = {}".format(len(processed_X_test)))
    # print('Dataset import is completed.')

    # labels=['s04','s04','s04','s08','s08','s08','s08']
    # val_loader=read_test_file(transform,'try',labels=labels)
    predict_class,val_loader=read_test_file(transform,'try')

    # model:mainly using resnet18;could be changed to alexnet,vgg,etc. similarly
    model = torchvision.models.resnet18(pretrained=True)
    num_fc = model.fc.in_features

    # change the output to 50 dimention
    model.fc = nn.Linear(num_fc,50)
    # turn on the gradient calculation
    for param in model.parameters():    param.requires_grad = True
    print('Model import is completed')

    # model=model.cuda()
    # use GPU or CPU to run the program
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    print('GPU device count: ',torch.cuda.device_count())
    # if there are more than one GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1,2])
        model=model.cuda()
        torch.backends.cudnn.benchmark = True

    start_time = time.time()
    print('Predict Results:')

    # 加载参数
    for j in range(len(predict_class)):
        model.zero_grad()
        result='unmasked'
        if predict_class[j][0]==1: 
            result='masked'
            state_dict=model.load_state_dict(torch.load('masked_resnet18_model_best.pth.tar', map_location='cpu'))
        else:state_dict=model.load_state_dict(torch.load('processed_resnet18_model_best.pth.tar', map_location='cpu'))
        val_acc = 0.0
        predict_results=[]
        with_labels=False
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                val_acc += np.sum(
                    np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                predict_results.append(np.argmax(val_pred.cpu().data.numpy(), axis=1).tolist()[0]+1)
                if -1 not in data[1].numpy():with_labels=True

        for i in range(len(predict_results)):
            if predict_results[i]<10:predict_results[i]='s0'+str(predict_results[i])
            else:predict_results[i]='s'+str(predict_results[i])
        val_acc /= val_loader.__len__()
        print(sorted(os.listdir('try'))[j]+'  '+ result+'  '+str(predict_results[j]))
        if with_labels: 
            print('Ground Truth: '+str(labels))
            print('Accuracy on validation set: %3.6f' % val_acc)