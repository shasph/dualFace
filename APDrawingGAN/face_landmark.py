from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import csv

def getfeats_dlib_fromImg(fname):
    image = face_recognition.load_image_file(fname)
    face_landmarks_list = face_recognition.face_landmarks(image)
    return getfeats_dlib(face_landmarks_list[0])
     
def getfeats_dlib(face_landmarks):
        #print(face_landmarks)
        trans_points = np.empty([5,2],dtype=np.int64) 
        
        arr0 = np.array(face_landmarks['left_eye'][0],dtype=np.int64)#left_eye
        arr1 = np.array(face_landmarks['left_eye'][3],dtype=np.int64)
        arr1=(arr0+arr1)/2
        trans_points[0,:] = arr1
        
        arr0 = np.array(face_landmarks['right_eye'][0],dtype=np.int64)#left_eye
        arr1 = np.array(face_landmarks['right_eye'][3],dtype=np.int64)
        arr1=(arr0+arr1)/2
        trans_points[1,:] = arr1
        
        arr1= np.array(face_landmarks['nose_bridge'][3],dtype=np.int64)
        trans_points[2,:] = arr1
        
        arr1= np.array(face_landmarks['top_lip'][0],dtype=np.int64)#0,7
        trans_points[3,:] = arr1
        arr1= np.array(face_landmarks['top_lip'][7],dtype=np.int64)#0,7
        trans_points[4,:] = arr1
        return trans_points
        
