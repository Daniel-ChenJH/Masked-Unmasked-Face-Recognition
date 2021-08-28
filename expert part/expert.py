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
# import more libraries as you need


# T1  start _______________________________________________________________________________
# Read in Dataset
# dataset_path = "D:\\Python-Programming\\nus\\Proj 2\\nus_project\\Dataset_1"
# processed_saved_path="D:\\Python-Programming\\nus\\Proj 2\\nus_project\\processed_img"
# masked_saved_path="D:\\Python-Programming\\nus\\Proj 2\\nus_project\\masked_img"
# X = []
# y = []
# X_masked=[]
# X_processed=[]
# for subject_name in os.listdir(dataset_path):
#     y.append(subject_name)
#     subject_images_dir = os.path.join(dataset_path,subject_name )

#     temp_x_list = []
#     for img_name in os.listdir(subject_images_dir):
#         img = io.imread(os.path.join(subject_images_dir, img_name),plugin='matplotlib')
#         temp_x_list.append(img)
#     X.append(temp_x_list)
# # change the dataset path here according to your folder structure
#     # add the temp_x_list to X
# # T1 end ____________________________________________________________________________________

# # T2 start __________________________________________________________________________________
# # # # Preprocessing
# # X_processed = []
# # X_masked = []
# # i=0
# # model_processed=[]
# # model_masked=[]
# # detector=dlib.get_frontal_face_detector()
# # predictor=dlib.shape_predictor("G:\\SJTU\\2021-7\\nus_project\\shape_predictor_68_face_landmarks.dat")
# # for x_list in X:
# #     temp_X_processed = []
# #     temp_X_masked = []
# #     temp_model_processed=[]
# #     temp_model_masked=[]
# #     j=0
# #     i=i+1
# #     for x in x_list:
# #         j=j+1
# #         dets=detector(x,1)
# #         # write the code to detect face in the image (x) using dlib facedetection library
# #         for k,d in enumerate(dets):
# #             pos_start=tuple([d.left(),d.top()])
# #             pos_end=tuple([d.right(),d.bottom()])
# #             height=d.bottom()-d.top()
# #             width=d.right()-d.left()
# #             img_blank=np.zeros((height,width,3),np.uint8)
# #             for a in range(height):
# #                 for b in range(width):
# #                     img_blank[a][b]=x[d.top()+a][d.left()+b]
# #             processed_img=cv2.resize(img_blank,(150,150))
# #         # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
# #             #gray=cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
# #         # write the code to convert the image (x) to grayscale
# #             gaussian=cv2.GaussianBlur(processed_img,(5,5),1.5)
# #         temp_X_processed.append(gaussian)
# #         #processed_img_model_training
# #         if i<=30:
# #             face_encoding=face_recognition.face_encodings(gaussian)
# #             if len(face_encoding):
# #                 temp_model_processed.append(face_encoding)
# #         # if i<10 and j<10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\processed_img\\s0'+str(i)+'\\0'+str(j)+'.jpg', gaussian)
# #         # if i<10 and j>=10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\processed_img\\s0'+str(i)+'\\'+str(j)+'.jpg', gaussian)
# #         # if i>=10 and j<10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\processed_img\\s'+str(i)+'\\0'+str(j)+'.jpg', gaussian)
# #         # if i>=10 and j>=10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\processed_img\\s'+str(i)+'\\'+str(j)+'.jpg', gaussian)
# #         dets=detector(gaussian,1)
# #         for(a,det) in enumerate(dets):
# #             shape=predictor(gaussian,det)
# #         # write the code to add synthetic mask as shown in the project problem description
# #         mask_area=np.array([[shape.part(2).x,shape.part(2).y],[shape.part(3).x,shape.part(3).y],[shape.part(4).x,shape.part(4).y],[shape.part(5).x,shape.part(5).y],[shape.part(6).x,shape.part(6).y],[shape.part(7).x,shape.part(7).y],[shape.part(8).x,shape.part(8).y],[shape.part(9).x,shape.part(9).y],[shape.part(10).x,shape.part(10).y],[shape.part(11).x,shape.part(11).y],[shape.part(12).x,shape.part(12).y],[shape.part(13).x,shape.part(13).y],[shape.part(14).x,shape.part(14).y],[shape.part(15).x,shape.part(15).y],[shape.part(29).x,shape.part(29).y]])
# #         #print (mask_area)
# #         cv2.fillPoly(gaussian,[mask_area],(255,255,255))
# #         # append the converted image into temp_X_masked
# #         temp_X_masked.append(gaussian)
# #         # if i<10 and j<10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\masked_img\\s0'+str(i)+'\\0'+str(j)+'.jpg', gaussian)
# #         # if i<10 and j>=10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\masked_img\\s0'+str(i)+'\\'+str(j)+'.jpg', gaussian)
# #         # if i>=10 and j<10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\masked_img\\s'+str(i)+'\\0'+str(j)+'.jpg', gaussian)
# #         # if i>=10 and j>=10:
# #         #     io.imsave('G:\\SJTU\\2021-7\\nus_project\\masked_img\\s'+str(i)+'\\'+str(j)+'.jpg', gaussian)
# #         #io.imsave("G:\\SJTU\\2021-7\\nus_project\\masked\\2.jpg",x)
# #     X_masked.append(temp_X_masked)
# #     model_processed.append(temp_model_processed)
# #     X_processed.append(temp_X_processed)
# # #     # append temp_X_processed into  X_processed

# # Create masked face dataset
# # i=0
# # for x_list in X_processed:
# #     j=0
# #     i+=1
# #     for x in x_list:
# #         # write the code to detect face in the image (x) using dlib facedetection library
# #         dets=detector(x,1)
# #         for(a,det) in enumerate(dets):
# #             shape=predictor(x,det)
# #         # write the code to add synthetic mask as shown in the project problem description
# #         mask_area=np.array([[shape.part(2).x,shape.part(2).y],[shape.part(3).x,shape.part(3).y],[shape.part(4).x,shape.part(4).y],[shape.part(5).x,shape.part(5).y],[shape.part(6).x,shape.part(6).y],[shape.part(7).x,shape.part(7).y],[shape.part(8).x,shape.part(8).y],[shape.part(9).x,shape.part(9).y],[shape.part(10).x,shape.part(10).y],[shape.part(11).x,shape.part(11).y],[shape.part(12).x,shape.part(12).y],[shape.part(13).x,shape.part(13).y],[shape.part(14).x,shape.part(14).y],[shape.part(15).x,shape.part(15).y],[shape.part(29).x,shape.part(29).y]])
# #         #print (mask_area)
# #         cv2.fillPoly(x,[mask_area],(255,255,255))
# #         # append the converted image into temp_X_masked
# #         temp_X_masked.append(x)
# #     append temp_X_masked into  X_masked
# # T3 end ____________________________________________________________________________________


# # T4 start __________________________________________________________________________________
# # Build a detector that can detect presence of facemask given an input image

# #         write code to read each 'img'

# #         add the img to temp_x_list
# i=0
# model_processed=[]
# model_masked=[]
# for subject_name in os.listdir(processed_saved_path):
#     y.append(subject_name)
#     j=0
#     i+=1
#     subject_images_dir = os.path.join(processed_saved_path,subject_name )

#     temp_x_list = []
#     temp_processed=[]
#     for img_name in os.listdir(subject_images_dir):
#         j+=1
#         img = io.imread(os.path.join(subject_images_dir, img_name),plugin='matplotlib')
#         temp_x_list.append(img)
#         if(i<=30):
#             face_encoding=face_recognition.face_encodings(img)
#             if len(face_encoding):
#                 temp_processed.append(face_encoding)


#         # write code to read each 'img'

#         # add the img to temp_x_list
#     X_processed.append(temp_x_list)
#     model_processed.append(temp_processed)
# i=0
# for subject_name in os.listdir(masked_saved_path):
#     y.append(subject_name)
#     i+=1
#     subject_images_dir = os.path.join(masked_saved_path,subject_name )
#     temp_masked=[]
#     temp_x_list = []
#     for img_name in os.listdir(subject_images_dir):
#         img = io.imread(os.path.join(subject_images_dir, img_name),plugin='matplotlib')
#         temp_x_list.append(img)
#         # write code to read each 'img'

#         # add the img to temp_x_list
#     X_masked.append(temp_x_list)


# # test the model
# # test_img=io.imread("G:\\SJTU\\2021-7\\nus_project\\processed_img\\s10\\01.jpg")
# # test_img_encoding=face_recognition.face_encodings(test_img)
# # for x_list in model_processed:
# #     for x in x_list:
# #         results=face_recognition.compare_faces(np.array(x),np.array(test_img_encoding))
# #         if(results[0]==True):
# #             print("accept")
# #             break
# #     if(results[0]==True):
# #         break
# # if(results[0]==False):
# #     print("reject")

# # train the masked model by SVM since masked people cannot be encoded
# y=[]
# model_features=[]
# i=0
# for x_list in X_masked:
#     i+=1
#     for x in x_list:
#         if(i<=30):
#             x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
#                         cells_per_block=(1, 1), visualize=False, multichannel=True)
#             model_features.append(x_feature)
#             y.append(1)
#         else:
#             x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
#                         cells_per_block=(1, 1), visualize=False, multichannel=True)
#             model_features.append(x_feature)
#             y.append(0)
# X_train,X_test,y_train,y_test=train_test_split(model_features,y,test_size=0.2,shuffle=True)
# clf_masked=svm.SVC(C=2500,gamma=0.02)
# clf_masked.fit(X_train,y_train)

# # classifier
# y=[]
# features=[]
# X_features = []
# for x_list in X_masked:
#     temp_X_features = []
#     for x in x_list:
#         x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
#                         cells_per_block=(1, 1), visualize=False, multichannel=True)
#         temp_X_features.append(x_feature)
#         features.append(x_feature)
#         y.append(1)
#     X_features.append(temp_X_features)
# for x_list in X_processed:
#     temp_X_features=[]
#     for x in x_list:
#         x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
#                         cells_per_block=(1, 1), visualize=False, multichannel=True)
#         temp_X_features.append(x_feature)
#         features.append(x_feature)
#         y.append(0)
#     X_features.append(temp_X_features)

# X_train,X_test,y_train,y_test=train_test_split(features,y,test_size=0.2,shuffle=True)
# # write code to split the dataset into train-set and test-set
# clf_classifier=svm.SVC()
# clf_classifier.fit(X_train,y_train)
# print("测试集准确率")
# print (clf_classifier.score(X_test,y_test))
# # write code to train and test the SVM classifier as the facemask presence detector

import pickle

f2=open('clf_classifier.model','rb')
s2=f2.read()
clf_classifier=pickle.loads(s2)

f2=open('clf_masked.model','rb')
s2=f2.read()
clf_masked=pickle.loads(s2)

test_img=io.imread("D:\\Python-Programming\\nus\\Proj 2\\nus_project\\masked_img\\s10\\01.jpg")
test_features=hog(test_img, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False, multichannel=True)


predict_class=clf_classifier.predict(test_features)
if(predict_class==1):
    if(clf_masked.predict(test_features.reshape(1,-1))==1):
        print("masked Accept")
    else:
        print("masked Reject")
if(predict_class==0):
    test_img_encoding=face_recognition.face_encodings(test_img)
    for x_list in model_processed:
        for x in x_list:
            results=face_recognition.compare_faces(np.array(x),np.array(test_img_encoding))
            if(results[0]==True):
                print("unmasked Accept")
                break
        if(results[0]==True):
            break
    if(results[0]==False):
        print("unmasked Reject")

