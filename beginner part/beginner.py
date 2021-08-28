import os
import cv2 as cv
import dlib
import sys
import copy
from skimage import io,transform,filters
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage.util import img_as_ubyte
# import more libraries as you need
from face_landmark_detection import getface

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "D:\\Python-Programming\\nus\\Proj 2\\Dataset_1"
X = []
y = []
gaussion_sigma=10

print(dlib.DLIB_USE_CUDA)

print(dlib.cuda.get_num_devices())

for subject_name in os.listdir(dataset_path):
    y.append(subject_name)
    subject_images_dir = os.path.join(dataset_path,subject_name )

    temp_x_list = []

    for img_name in os.listdir(subject_images_dir):
        # write code to read each 'img'
        # img=cv.imread(img_name)
        img=io.imread(os.path.join(subject_images_dir,img_name))
        # img = img.astype(np.uint8)

        #if use gaussian we need to multiply it by 255
        img=filters.gaussian(img,sigma=gaussion_sigma)
        img=img*255

        img = img.astype(np.uint8)
        temp_x_list.append(img)
        # io.imshow(img)
        # plt.show()        
        # add the img to temp_x_list
    X.append(temp_x_list)
    # add the temp_x_list to X

# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []
X_masked = []

if not os.path.exists('Processed_Dataset_1'):
    os.mkdir('Processed_Dataset_1')
    for i in range(50):
        if i<9:os.mkdir('Processed_Dataset_1\\s0'+str(i+1))
        else:os.mkdir('Processed_Dataset_1\\s'+str(i+1))

if not os.path.exists('Masked_Dataset_1'):
    os.mkdir('Masked_Dataset_1')
    for i in range(50):
        if i<9:os.mkdir('Masked_Dataset_1\\s0'+str(i+1))
        else:os.mkdir('Masked_Dataset_1\\s'+str(i+1))
i=0

non_detected=0
for x_list in X:
    i+=1
    j=0
    temp_X_processed = []
    temp_X_masked = []
    for x in x_list:
        j+=1
        # write the code to detect face in the image (x) using dlib facedetection library
        try:left,top,right,bottom,maskregion=getface('shape_predictor_68_face_landmarks.dat',x)
        except UnboundLocalError:
            non_detected+=1
            continue
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
        ori_x=copy.deepcopy(x)
        # print(maskregion)

        cv.fillPoly(x, maskregion, (255,255,255))
        cropped_mask_x=x[top-100:bottom+50,left-50:right+50,:]
        gray_mask_x=rgb2gray(cropped_mask_x)
        gray_mask_x=transform.resize(gray_mask_x, (150,150))

        cropped_x=ori_x[top-100:bottom+50,left-50:right+50,:]
        gray_x=rgb2gray(cropped_x)
        gray_x=transform.resize(gray_x, (150,150))

        # io.imshow(x)
        # plt.show()
        # write the code to convert the image (x) to grayscale
        temp_X_processed.append(gray_x)
        temp_X_masked.append(gray_mask_x)

        # if i<10 and j<10:
        #     io.imsave('Processed_Dataset_1\\s0'+str(i)+'\\0'+str(j)+'.jpg', gray_x)
        #     io.imsave('Masked_Dataset_1\\s0'+str(i)+'\\0'+str(j)+'.jpg', gray_mask_x)
        # if i<10 and j>=10:
        #     io.imsave('Processed_Dataset_1\\s0'+str(i)+'\\'+str(j)+'.jpg', gray_x)
        #     io.imsave('Masked_Dataset_1\\s0'+str(i)+'\\'+str(j)+'.jpg', gray_mask_x)
        # if i>=10 and j<10:
        #     io.imsave('Processed_Dataset_1\\s'+str(i)+'\\0'+str(j)+'.jpg', gray_x)
        #     io.imsave('Masked_Dataset_1\\s'+str(i)+'\\0'+str(j)+'.jpg', gray_mask_x)
        # if i>=10 and j>=10:
        #     io.imsave('Processed_Dataset_1\\s'+str(i)+'\\'+str(j)+'.jpg', gray_x)
        #     io.imsave('Masked_Dataset_1\\s'+str(i)+'\\'+str(j)+'.jpg', gray_mask_x)

    print(i)
        
        # io.imshow(gray_mask_x)
        # plt.show()
        # append the converted image into temp_X_processed
    X_processed.append(temp_X_processed)
    X_masked.append(temp_X_masked)
    # append temp_X_processed into  X_processed

# T2 end ____________________________________________________________________________________


# # T3 start __________________________________________________________________________________
# # Create masked face dataset
# X_masked = []
# for x_list in X_processed:
#     temp_X_masked = []
#     for x in x_list:
#         # write the code to detect face in the image (x) using dlib facedetection library        
#         cv.fillPoly(x, maskregion, (255, 255, 255))
#         # write the code to add synthetic mask as shown in the project problem description
#         io.imshow(x)
#         plt.show()
#         # append the converted image into temp_X_masked

#     # append temp_X_masked into  X_masked

# # T3 end ____________________________________________________________________________________


# T4 start __________________________________________________________________________________
# Build a detector that can detect presence of facemask given an input image


# dataset_path = "D:\\Python-Programming\\nus\\Proj 2\\Masked_Dataset_1"
# X_masked = []
# y = []

# for subject_name in os.listdir(dataset_path):
#     y.append(subject_name)
#     subject_images_dir = os.path.join(dataset_path,subject_name )

#     temp_x_list = []

#     for img_name in os.listdir(subject_images_dir):
#         # write code to read each 'img'
#         # img=cv.imread(img_name)
#         img=io.imread(os.path.join(subject_images_dir,img_name),0)
#         img = img.astype(np.uint8)
#         temp_x_list.append(img)
#         # add the img to temp_x_list
#     X_masked.append(temp_x_list)
#     # add the temp_x_list to X


# dataset_path = "D:\\Python-Programming\\nus\\Proj 2\\Processed_Dataset_1"
# X_processed = []
# y = []

# for subject_name in os.listdir(dataset_path):
#     y.append(subject_name)
#     subject_images_dir = os.path.join(dataset_path,subject_name )

#     temp_x_list = []

#     for img_name in os.listdir(subject_images_dir):
#         # write code to read each 'img'
#         # img=cv.imread(img_name)
#         img=io.imread(os.path.join(subject_images_dir,img_name),0)
#         img = img.astype(np.uint8)
#         temp_x_list.append(img)
#         # add the img to temp_x_list
#     X_processed.append(temp_x_list)
#     # add the temp_x_list to X

X_features = []
X_extracted=[]
y = []

for x_list in X_masked:
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, multichannel=False)
        temp_X_features.append(x_feature)
        X_extracted.append(x_feature)
        y.append(1)
    X_features.append(temp_X_features)

for x_list in X_processed:
    temp_X_features = []
    for x in x_list:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, multichannel=False)
        temp_X_features.append(x_feature)
        X_extracted.append(x_feature)
        y.append(0)
    X_features.append(temp_X_features)

# print(X_extracted)
# print(y)
print('Nums of non face detected after gaussion blur with σ== '+str(gaussion_sigma)+' : '+str(non_detected)+' within 750 photos, percentage == '+str(non_detected/750))
X_train, X_test, y_train, y_test = train_test_split(X_extracted, y, test_size=0.2, shuffle=True)

# write code to train and test the SVM classifier as the facemask presence detector
print("start SVM training.")
clf = svm.SVC()
clf.fit(X_train, y_train)

print("SVM training completed.")
print(clf.score(X_test, y_test))
# T4 end ____________________________________________________________________________________
