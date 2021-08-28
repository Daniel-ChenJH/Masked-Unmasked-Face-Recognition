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
    for subject_name in os.listdir(dataset_path):
        subject_images_dir = os.path.join(dataset_path,subject_name)
        image_dir = sorted(os.listdir(subject_images_dir))
        for i, file_name in enumerate(image_dir):
            img = cv2.imread(os.path.join(subject_images_dir, file_name))
            
            x[j, :, :] = cv2.resize(img, (150, 150))
            y[j] = int(subject_name.replace("s",''))-1
            j+=1
    return x, y                

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
    workspace_dir = 'D:\\Python-Programming\\nus\\Proj 2\\nus_project'

    print("Reading data")
    processed_x, processed_y = readfile(os.path.join(workspace_dir, "processed_img"))
    # masked_x, masked_y = readfile(os.path.join(workspace_dir, "masked_img"))
    processed_X_train, processed_X_test, processed_y_train, processed_y_test = train_test_split(processed_x, processed_y, test_size=0.2, shuffle=True)
    train_set = ImgDataset(processed_X_train, processed_y_train, transform)
    val_set = ImgDataset(processed_X_test, processed_y_test, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,pin_memory=True)

    # train_set=datasets.CIFAR10('data/',download=True, train=True, transform=transform)
    # val_set=datasets.CIFAR10('data/',download=True, train=False, transform=transform)
    
    # train_set, val_set,leaveout= torch.utils.data.random_split(dataset= val_set, lengths=[2000, 500, 7500])
    print("Size of training data = {}".format(len(processed_X_train)))
    print("Size of validation data = {}".format(len(processed_X_test)))
    print('Dataset import is completed.')

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

    # use CrossEntropyLoss as loss function
    # build optimizer;
    # epoch times num_epoch;
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 

    # Set the learning rate to exponentially decay
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    print("Initial learning rate：", optimizer.defaults['lr'])
    num_epoch = 15
    val_acc_best = 0.0

    # start training and print the results
    for epoch in range(num_epoch):
        print("Learning rate of epoch %d ：%f" % (epoch+1, optimizer.param_groups[0]['lr']))

        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() 
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  
            # data=Image.fromarray(data)
            train_pred = model(data[0].to(device))  # do Forward propagation
            batch_loss = loss(train_pred, data[1].to(device))  # compute loss
            batch_loss.backward()  # do Backward propagation
            optimizer.step()  # use gradient to refresh parameters

            train_acc += np.sum(
                np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                batch_loss = loss(val_pred, data[1].to(device))

                val_acc += np.sum(
                    np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                    data[1].numpy())
                val_loss += batch_loss.item()

        train_acc /= train_set.__len__()
        train_loss /= train_set.__len__()
        val_acc /= val_set.__len__()
        val_loss /= val_set.__len__()

        # print out the results
        print(
            '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f'
            % (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc,
            train_loss, val_acc, val_loss))

        # save the best model
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(model.state_dict(), 'processed_resnet18_model_best.pth.tar')
            print('Save model')

        # decay the learning rate exponentially
        scheduler.step()

    print('Best accuracy on validation set: %3.6f' % val_acc_best)
    endtime=time.time()
    print('Total time for training：',endtime-start_time,' seconds')