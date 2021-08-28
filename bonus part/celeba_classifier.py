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
import pickle



def read_file(dataset_path,label):
    model_features=[]
    labels=[]
    for img_name in sorted(os.listdir(dataset_path)):
        if int(img_name.split('_')[1].replace('.jpg','').strip())<=100:
            img = io.imread(os.path.join(dataset_path, img_name),plugin='matplotlib')
            x_feature = hog(img, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, multichannel=True)
            model_features.append(x_feature)
            labels.append(label)
    return model_features,labels

model_features,labels=read_file(os.path.join(os.getcwd(),'celeba/masked_img'),1)
model_features2,labels2=read_file(os.path.join(os.getcwd(),'celeba/processed_img'),0)
model_features.extend(model_features2)
labels.extend(labels2)

X_train,X_test,y_train,y_test=train_test_split(model_features,labels,test_size=0.2,shuffle=True)
# write code to split the dataset into train-set and test-set
clf_classifier=svm.SVC()
clf_classifier.fit(X_train,y_train)
print("测试集准确率")
print (clf_classifier.score(X_test,y_test))


s=pickle.dumps(clf_classifier)
f=open('celeba_classifier.model', "wb+")
f.write(s)
f.close()
print ("Done\n")