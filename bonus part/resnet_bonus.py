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
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision
import time
from torch.optim.lr_scheduler import ExponentialLR

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    batch_size = 16
    photosize=128

    # 设置随机数种子
    setup_seed(1896)

    # 定义数据集处理方式
    transform=transforms.Compose([
        transforms.Resize(photosize),  #缩放图片（Image）,保持长宽比不变，最短边为128像素
        transforms.CenterCrop(photosize), #从图片中间裁剪出128*128的图片
        transforms.ToTensor(), #将图片Image转换成Tensor，归一化至【0,1】
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  #标准化至【-1,1】，规定均值和方差
    ])

    #导入cifar10数据集
    train_set=datasets.CIFAR10('data/',download=True, train=True, transform=transform)
    val_set=datasets.CIFAR10('data/',download=True, train=False, transform=transform)
    
    # train_set, val_set,leaveout= torch.utils.data.random_split(dataset= val_set, lengths=[2000, 500, 7500])
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=16)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=16)
    print("Size of training data = {}".format(len(train_set)))
    print("Size of validation data = {}".format(len(val_set)))
    print('数据集导入完成')

    # 搭建模型 model，采用resnet18为主体结构；此处可改为alexnet、vgg等结构，相应地改变其最后全连接层即可。对alexnet、vgg为model.classifier[6]
    model = torchvision.models.resnet18(pretrained=True)
    num_fc = model.fc.out_features
    # 微调网络结构，将输出改为10维
    model.fc = nn.Sequential(model.fc, nn.ReLU(), nn.Dropout(0.4), nn.Linear(num_fc,10))
    # 开启全网络的梯度流运算
    for param in model.parameters():    param.requires_grad = True
    print('模型导入完成')

    model=model.cuda()
    # 采用GPU运行程序
    device = torch.device('cuda:0')
    print(device)
    print('GPU device count: ',torch.cuda.device_count())
    # 如果有多张显卡
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1,2])
        model=model.cuda()
        torch.backends.cudnn.benchmark = True

    start_time = time.time()

    # 构造损失函数 loss，采用交叉熵函数;
    # 构造优化器 optimizer;
    # 设定训练次数 num_epoch;
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 

    # 设定学习率指数衰减
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    print("初始化的学习率：", optimizer.defaults['lr'])
    num_epoch = 1
    val_acc_best = 0.0

    # 训练 并print每个epoch的结果;
    for epoch in range(num_epoch):
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))

        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() 
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 用 optimizer 将 model 参数的 gradient 归零
            train_pred = model(data[0].to(device))  # 调用 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].to(device))  # 计算 loss
            batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
            optimizer.step()  # 以 optimizer 用 gradient 更新参数值

            train_acc += np.sum(
                np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            train_loss += batch_loss.item()
            break

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                print('11111111111111111111')
                print(val_pred)
                print(data[1])
                batch_loss = loss(val_pred, data[1].to(device))
                break

                val_acc += np.sum(
                    np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                    data[1].numpy())
                val_loss += batch_loss.item()

        train_acc /= train_set.__len__()
        train_loss /= train_set.__len__()
        val_acc /= val_set.__len__()
        val_loss /= val_set.__len__()

        # 将结果 print 出来
        print(
            '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f'
            % (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc,
            train_loss, val_acc, val_loss))

        # 记录最好的结果 并保存模型
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(model.state_dict(), 'resnet18_model_best.pth.tar')
            print('Save model')

        # 学习率指数衰减一次，衰减因子为0.9
        scheduler.step()

    print('Best accuracy on validation set: %3.6f' % val_acc_best)
    endtime=time.time()
    print('模型训练总用时：',endtime-start_time,'秒')