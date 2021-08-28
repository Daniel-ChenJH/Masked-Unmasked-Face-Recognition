import os
from types import prepare_class
import cv2
import numpy as np
import dlib
from skimage import io,color,transform
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn import svm
import face_recognition
import sys
import copy
# import more libraries as you need

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


if not os.path.exists('celeba/processed_img'):
    os.mkdir('celeba/processed_img')
if not os.path.exists('celeba/masked_img'):
    os.mkdir('celeba/masked_img')

with open('celeba/identity_CelebA.txt','r') as f:
    lines = f.readlines()
check_list=[]
for line in lines:
    check_list.append(line.split(' ')[0].replace('.jpg', '')+'_'+line.split(' ')[1].strip()+'.jpg')

# T1  start _______________________________________________________________________________
# Read in Dataset
dataset_path = "D:\\Python-Programming\\nus\\Proj 2\\nus_project\\celeba\\images"
processed_saved_path="D:\\Python-Programming\\nus\\Proj 2\\nus_project\\celeba\\processed_img"
masked_saved_path="D:\\Python-Programming\\nus\\Proj 2\\nus_project\\celeba\\masked_img"
i=0
wrong_name=[]
for img_name in sorted(os.listdir(dataset_path)):
    try:
        if int(check_list[i].split('.')[0].split('_')[1].strip())>100:
            i+=1
            continue
        img = io.imread(os.path.join(dataset_path, img_name),plugin='matplotlib')
        img = (img*255).astype(np.uint8)
        dets=detector(img,1)
        if len(dets):
            for k,d in enumerate(dets):
                pos_start=tuple([d.left(),d.top()])
                pos_end=tuple([d.right(),d.bottom()])
                height=d.bottom()-d.top()
                width=d.right()-d.left()
                img_blank=np.zeros((height,width,3),np.uint8)
                for a in range(height):
                    for b in range(width):
                        img_blank[a][b]=img[d.top()+a][d.left()+b]
                processed_img=cv2.resize(img_blank,(150,150))
                gaussian=cv2.GaussianBlur(processed_img,(5,5),1.5)
            temp=copy.deepcopy(gaussian)

            dets=detector(gaussian,1)
            if len(dets):
                for(a,det) in enumerate(dets):
                    shape=predictor(gaussian,det)
                mask_area=np.array([[shape.part(2).x,shape.part(2).y],[shape.part(3).x,shape.part(3).y],[shape.part(4).x,shape.part(4).y],[shape.part(5).x,shape.part(5).y],[shape.part(6).x,shape.part(6).y],[shape.part(7).x,shape.part(7).y],[shape.part(8).x,shape.part(8).y],[shape.part(9).x,shape.part(9).y],[shape.part(10).x,shape.part(10).y],[shape.part(11).x,shape.part(11).y],[shape.part(12).x,shape.part(12).y],[shape.part(13).x,shape.part(13).y],[shape.part(14).x,shape.part(14).y],[shape.part(15).x,shape.part(15).y],[shape.part(29).x,shape.part(29).y]])
                cv2.fillPoly(gaussian,[mask_area],(255,255,255))
                io.imsave(masked_saved_path+'\\'+check_list[i], gaussian)
                io.imsave(processed_saved_path+'\\'+check_list[i], temp)
            else:
                wrong_name.append(i)
        else:
            wrong_name.append(i)
        if i%200==0:print(check_list[i])
        i+=1
    except KeyboardInterrupt:
        sys.exit()
    except:
        wrong_name.append(i)
        i+=1
print(wrong_name)
print(len(wrong_name))