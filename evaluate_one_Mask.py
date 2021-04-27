#!/usr/bin/python
# -*- encoding: utf-8 -*-
try:
    from logger import setup_logger
    from model import BiSeNet
except:
    from faceParsing.logger import setup_logger
    from faceParsing.model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from stroke_managers import remake_mask

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
maskMap=[0,5,6,3,4,2,7,8,14,1,9,10,11,16,15,17,12,13]
drawList_atts =[0,1,2,3,4,5,6,7,8,9,10,11,12,16]
def creat_image(img_size):
    img_mat = np.ones(img_size) * 255
    return img_mat
def MaskToColor(parsing_anno, stride=1):
    # Colors for all 20 parts

    part_colors = [ [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [51, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) #+ 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi-1]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_parsing_anno_color=cv2.cvtColor(vis_parsing_anno_color, cv2.COLOR_RGB2BGR)

    return vis_parsing_anno_color
def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    vis_parsing_anno[vis_parsing_anno > 0 ]=255

    
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    img2 = transforms.ToPILImage()(vis_parsing_anno)#vis_parsing_anno vis_im

    return vis_im

def vis_parsing_maps2(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    # Save result or not
    #vis_parsing_anno[vis_parsing_anno > 0 ]=255
    vis_parsing_anno[vis_parsing_anno == 0 ]=255

    img2 = transforms.ToPILImage()(vis_parsing_anno)#vis_parsing_anno vis_im
    #print(img2)
    #img2.show()
    
    return img2
    #return vis_parsing_anno
    
def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            #print(parsing)
            #print(parsing.shape)#512,512
            #print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))



def MaskIndexConvert0(idx):
    if idx>0 and idx<=len(maskMap):
        idx=maskMap[idx-1]+1
    else:
        idx=0
    return idx
    
def MaskIndexConvert(Img_mat):
    tmp_im = Img_mat.copy().astype(np.uint8)
    sub=np.ones(tmp_im.shape, dtype=np.uint8)*100
    for i in np.unique(Img_mat):
        if(i==0):
            continue
        idx=maskMap[i-1]+1 
        tmp_im[tmp_im==i]=100+idx
    tmp_im[tmp_im==0]=100
    tmp_im=tmp_im-sub
    return tmp_im

    
from matplotlib import pyplot as plt
    
    
def GetMaskContours(Img_mat,showPreview=False): 
    #ContourList=[]
    ContourList={}
    #item = {'A': A, 'A_paths': A_path}
    for i in range(18):
        label_idx=i-1
        if(i==0):
            continue
        if(not any([ label_idx in drawList_atts])):
            continue
        
        #print(label_idx)
        tmp_im= Img_mat.copy().astype(np.uint8)
        tmp_im[tmp_im==i]=255
        if(label_idx==0):
            #labelset=[]
            tmp_im[(0+1 < tmp_im) & (tmp_im < 12+1)&(tmp_im != 7+1)&(tmp_im != 8+1)]=255
        tmp_im[tmp_im!=255]=0
        #pimg=transforms.ToPILImage()(tmp_im)
        gray = np.array(tmp_im)#cv2.cvtColor(np.array(pimg),cv2.COLOR_BGR2GRAY)
        
        #gray = cv2.cvtColor(tmp_im,cv2.COLOR_GRAY2BGR)
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
        if(showPreview):
            plt.imshow(gray,'gray')
            plt.show()
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #ContourList.append(contours)
        ContourList[label_list[label_idx]]=contours
        #print('GetMaskContours=',contours)
    #print('GetMaskContours=',ContourList)
    return ContourList
        
def DrawMaskContours(Img_mat,contourList,line_width=3): 
    
    
    cv_image =creat_image(Img_mat.shape)
    #for idx in drawList_atts:
    #for name,contours in contourList:
    for name,contours in contourList.items():
        #contours=contourList[idx]
        #print('DrawMaskContours',name)
        if contours is not None:
            cv2.drawContours(cv_image,contours,-1,(0,0,0),line_width)
    #cv2.imshow("img", cv_image)  
    #cv2.waitKey(0)
    return cv_image
def MaskDraw(Img_mat):
    #print(np.unique(Img_mat))
    for i in np.unique(Img_mat):
        if(i==0):
            continue
        idx=maskMap[i-1]
        print(i-1,label_list[idx])
        tmp_im = Img_mat.copy().astype(np.uint8)
        tmp_im[tmp_im==i]=255
        img2 = transforms.ToPILImage()(tmp_im)
        img2.show()
def save_temp_mask(img_path,img):
    temp_path='temp'
    save_path=osp.join(temp_path, img_path)
    save_name=save_path[:-4] +'.png'
    #my_file = Path(save_name)
    if os.path.exists(save_name):
        print(save_name,'existed')
    else:
        cv2.imwrite(save_name, img.copy().astype(np.uint8))

def get_mask_net(cp='model_final_diss.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('faceParsing/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    return net
def evaluate_one(net,strokes,img_path='./data'):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #for image_path in os.listdir(dspth):
            image_path=img_path
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            parsing=MaskIndexConvert(parsing)# idx is different between MASKGAN and parsing_face net
            ori_mask=parsing.copy().astype(np.uint8)
            parsing,colored_mask=remake_mask(strokes,parsing)

            
            Contours=GetMaskContours(parsing)
            maskMono=DrawMaskContours(parsing,Contours)
            maskMono=cv2.cvtColor(maskMono.astype('uint8'),cv2.COLOR_GRAY2RGB)
            
            maskMono_PIL = Image.fromarray(maskMono)                                    #gray cv2img to pil

            maskPIL=transforms.ToPILImage()(parsing.copy().astype(np.uint8))
            
            parsing[parsing > 0 ]=255
            bgPIL=transforms.ToPILImage()(parsing.copy().astype(np.uint8))

            return maskPIL,bgPIL,maskMono_PIL,Contours,ori_mask,colored_mask



if __name__ == "__main__":
    pass


