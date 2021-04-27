from RamerDouglasPeucker import rdp
import pickle
from matplotlib import pyplot as plt
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
color_list_p = [ [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [51, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
maskMap=[0,5,6,3,4,2,7,8,14,1,9,10,11,16,15,17,12,13]
drawList_atts =[0,1,2,3,4,5,6,7,8,9,10,11,12,16]
def strokes_load():
    with open("strokes_save.txt", "rb") as fp:   #Unpickling
        strokes = pickle.load(fp)      
        return strokes

def ConverColorToMark(cv_image):
    
    img_mask=np.copy(cv_image)
    for i in range(len(color_list_p)):
        color=np.array(color_list_p[i])
        pcolor=np.array([color[2],color[1],color[0]])

        mk=i+1
        mark=[mk,mk,mk]
        mask_i=cv2.inRange(cv_image,pcolor,pcolor)
        img_mask[mask_i!=0]=mark
    
    pcolor=np.array([255,255,255])
    mark=[0,0,0]
    mask_i=cv2.inRange(cv_image,pcolor,pcolor)
    img_mask[mask_i!=0]=mark
    #cv2.imwrite('cout.png',img_mask)
    img_mask = cv2.cvtColor(np.array(img_mask),cv2.COLOR_BGR2GRAY)
    ##cv2.imwrite('cout.png',img_mask)
    img_mask=np.array(img_mask)
    return img_mask
def DrawMaskContours(Img_mat,contourList,line_width=-1): 
    cv_image =np.zeros((512, 512, 3), np.uint8)#creat_image(Img_mat.shape)
    for name,contours in contourList.items():
        point_color = color_list_p[label_list.index(name)]#RGB
        point_color=(point_color[2],point_color[1],point_color[0])
        #print('DrawMaskContours',name)
        if contours is not None:
            cv2.drawContours(cv_image,contours,-1,point_color,line_width)

    return cv_image
    
def SkinArea(contourList): 
    for name,contours in contourList.items():
        if (name==label_list[0]):
            area = cv2.contourArea(contours[0])
            print('SkinArea=',area)
            return area
    return 0.0   
def SkinAreaStroke(strokes,marks):
    maxArea=0.0   
    maxIdx=0
    for i in range(len(strokes)):
        mark=marks[i]
        if (mark==label_list[0]):
            stroke=strokes[i]
            contour=np.array(stroke).reshape((-1,1,2)).astype(np.int32)
            area = cv2.contourArea(contour)
            if(maxArea<area):
                maxArea=area
                maxIdx=i
    print(maxIdx,' owns maxArea=',maxArea)
    return maxArea,maxIdx   
def drawOriSkinCheck(contourList,strokes,marks):
    a1=SkinArea(contourList)
    a2,_=SkinAreaStroke(strokes,marks)
    alpha=0.6
    return (a1*alpha > a2)
def np_list_move(lst, k):
    #return lst[k:] + lst[:k]   
    return np.concatenate((lst[k:], lst[:k]), axis=0)
def list_move(lst, k):
    return lst[k:] + lst[:k]  
def month_split(mouth_stroke):
    ss=np.array(mouth_stroke)
    #print(ss.shape,ss)
    xlist=ss[:,0]
    ylist=ss[:,1]
    #print(xlist)

    xmin=xlist.argmin()
    xmax=xlist.argmax()


    if(xmin>xmax):
        xmin=xmin-len(xlist)
    #print(xmin,xmax,xmax-xmin)
    x2=np_list_move(xlist,xmin)
    y2=np_list_move(ylist,xmin)
    ss2=list(zip(x2,y2))
    a=np.split(ss2,[xmax-xmin])
    #print(a)
    y0=a[0][:,1]
    y1=a[1][:,1]
    mu1=np.mean(y1)
    mu2=np.mean(y2)
    if(mu1>mu2):
        return a[0],a[1]
    return a[1],a[0]
    cv_image =np.zeros((512, 512, 3), np.uint8)
    c1=np.array(a[0]).reshape((-1,1,2)).astype(np.int32)
    c2=np.array(a[1]).reshape((-1,1,2)).astype(np.int32)
    #c3=np.array(a[2]).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(cv_image,[c1],-1,(255,0,0),2)  
    cv2.drawContours(cv_image,[c2],-1,(0,255,0),2)
    cv2.imshow("month", cv_image)  
    cv2.waitKey(0)

def StrokeMerge(strokes):
    new_strokes=[]
    for s in strokes:
        for p in s:
            new_strokes.append(p)
    
    return new_strokes
def ReorderContoursWithMarks(strokes,marks,contourList):
    new_strokes=[]
    new_marks=[]
    mouth_strokes=[]
    SkipMouth=False
    for i in range(len(strokes)):
        mark=marks[i]
        if (mark==label_list[9]):
            SkipMouth=True
            return new_strokes,new_marks,SkipMouth
        if (mark==label_list[9] or mark==label_list[10] or mark==label_list[11]):#9,10,11
            mouth_strokes.append(strokes[i])
        else:
            new_strokes.append(strokes[i])
            new_marks.append(mark)
            
    unique_marks, mark_cnt = np.unique(new_marks, return_counts=True)
    tmp_strokes=[]
    tmp_marks=[]
    for j in range(len(unique_marks)):
        if(mark_cnt[j]>0):
            tmp=[]
            t_mark=unique_marks[j]
            mk_idx_list= [i for i,x in enumerate(new_marks) if x==t_mark]
            merged_stroke_mk = new_strokes[mk_idx_list[0]]
            if len(mk_idx_list)>1:
                for k in mk_idx_list:
                    tmp.append(new_strokes[k])
                merged_stroke_mk = StrokeMerge(tmp)
            tmp_strokes.append(merged_stroke_mk)
            tmp_marks.append(t_mark)

    new_strokes = tmp_strokes          
    new_marks  =  tmp_marks    
    if(len(mouth_strokes)==1):        
        mouth_stroke = mouth_strokes[0]
        ulip,llip=month_split(mouth_stroke)
        mark_ulip=label_list[10]
        mark_llip=label_list[11]
        new_strokes.append(ulip)
        new_strokes.append(llip)
        new_marks.append(mark_ulip)
        new_marks.append(mark_llip)
        return new_strokes,new_marks,SkipMouth
        
    if(len(mouth_strokes)>1):
        mouth_stroke=StrokeMerge(mouth_strokes)
        ulip,llip=month_split(mouth_stroke)
        mark_ulip=label_list[10]
        mark_llip=label_list[11]
        new_strokes.append(ulip)
        new_strokes.append(llip)
        new_marks.append(mark_ulip)
        new_marks.append(mark_llip)
        return new_strokes,new_marks,SkipMouth
        
    return strokes,marks,SkipMouth
def ReorderContoursWithMarks_hair(strokes,marks):
    new_strokes=[]
    new_marks=[]
    skin_strokes=[]
    skin_area,skin_no=SkinAreaStroke(strokes,marks)
    for i in range(len(strokes)):
        mark=marks[i]
        if (mark==label_list[0]):
            skin_strokes.append(strokes[i])
        else:
            new_strokes.append(strokes[i])
            new_marks.append(mark)
    
    if(len(skin_strokes)>1):
        for j in range(len(skin_strokes)):
            if(j==skin_no):
                new_marks.append(label_list[0])
            else:
                new_marks.append(label_list[12])
            new_strokes.append(skin_strokes[j])
        return new_strokes,new_marks
        
    return strokes,marks     
def DrawMaskContoursWithMarks(Img_mat,contourList,strokes,marks,line_width=-1): 
    SkipMouth=False
    strokes,marks,SkipMouth=ReorderContoursWithMarks(strokes,marks,contourList)
    #strokes,marks=ReorderContoursWithMarks_hair(strokes,marks)
    cv_image =np.zeros((512, 512, 3), np.uint8)#creat_image(Img_mat.shape)
    #drawOrder=[12,0,1,3,4,5,6,7,8,10,11]
    drawOrder=[12,0,1,2,3,4,5,6,7,8,9,10,11]
    drawOriSkin=True
    
    for odr in drawOrder:
        for name,contours in contourList.items():
            if(name != label_list[odr]):
                continue
            point_color = color_list_p[label_list.index(name)]#RGB
            point_color=(point_color[2],point_color[1],point_color[0])
            if (name==label_list[0]):
                drawOriSkin=True#drawOriSkinCheck(contourList,strokes,marks)# True
                print('drawOriSkin=',drawOriSkin)
            if (drawOriSkin and name==label_list[0]):
                cv2.drawContours(cv_image,contours,-1,point_color,line_width)
                drawOriSkin = False
                continue
            if(name in marks):
                if(SkipMouth):
                    if not (name in [9,10,11]):
                        continue
                else:
                    continue
            #print('DrawMaskContours',name)
            if contours is not None:
                cv2.drawContours(cv_image,contours,-1,point_color,line_width)
        for i in range(len(strokes)):
            mark=marks[i]
            if(label_list[odr]!=mark):
                continue
            if (drawOriSkin and mark==label_list[0]):
                continue
            stroke=strokes[i]
            point_color = color_list_p[label_list.index(mark)]#RGB
            point_color=(point_color[2],point_color[1],point_color[0])
            #contours=[stroke]
            contours=np.array(stroke).reshape((-1,1,2)).astype(np.int32)#np.array(stroke)
            #print(contours)
            #cv_image=
            cv2.drawContours(cv_image,[contours],-1,point_color,line_width)  # img为三通道才能显示轮廓
       
    return cv_image
def drawStrokes(strokes,thickness=3):
    img = np.ones((512, 512, 3), np.uint8)*255#np.zeros((512, 512, 3), np.uint8)
    ptStart=None
    ptEnd=None
    point_color = (0, 0, 0) # BGR
    lineType = 4
    for stroke in strokes:
        for point in stroke:
            point_int=(int(point[0]),int(point[1]))
            if(ptStart is None):
                ptStart=point_int
                continue
            ptEnd=point_int
            cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
            ptStart=ptEnd
        ptStart=None
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return img
def drawStrokesWithMarks(strokes,marks,thickness=3):
    img = np.ones((512, 512, 3), np.uint8)*255
    ptStart=None
    ptEnd=None
    #point_color = (0, 255, 0) # BGR
    lineType = 4
    for i in range(len(strokes)):
    #for stroke in strokes:
        mark=marks[i]
        stroke=strokes[i]
        point_color = color_list_p[label_list.index(mark)]#RGB
        point_color=(point_color[2],point_color[1],point_color[0])
        #img=cv2.drawContours(img,contours,-1,(0,255,0),5)  # img为三通道才能显示轮廓
        for point in stroke:
            point_int=(int(point[0]),int(point[1]))
            if(ptStart is None):
                ptStart=point_int
                continue
            ptEnd=point_int
            cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
            ptStart=ptEnd
        ptStart=None
    cv2.imshow('image', img)
    cv2.waitKey(0)
def drawStrokesWithMarks2(strokes,marks,thickness=3):
    img = np.ones((512, 512, 3), np.uint8)*0#255
    #point_color = (0, 255, 0) # BGR
    lineType = 4
    drawOrder=[12,0,1,3,4,5,6,10,11]
    for odr in drawOrder:
        for i in range(len(strokes)):
            mark=marks[i]
            if(label_list[odr]!=mark):
                continue
            stroke=strokes[i]
            point_color = color_list_p[label_list.index(mark)]#RGB
            point_color=(point_color[2],point_color[1],point_color[0])
            #contours=[stroke]
            contours=np.array(stroke).reshape((-1,1,2)).astype(np.int32)#np.array(stroke)
            #print(contours)
            img=cv2.drawContours(img,[contours],-1,point_color,thickness)  # img为三通道才能显示轮廓
            
    cv2.imshow('drawStrokesWithMarks2', img)
    cv2.waitKey(0)
    return img

def GetMinMax(contours):
    if(len(contours)==0):
        return 0,0,0,0
    contour=contours[0]   
    ss=np.array(contour)
    #print(ss[:,0][:,0])
    xlist=ss[:,0][:,0]
    ylist=ss[:,0][:,1]
    xmin=xlist[xlist.argmin()]#ss[np.argmin(ss, axis=0)][0]#ss[ss[:,0].argmin()]
    ymin=ylist[ylist.argmin()]#ss[ss[:,1].argmin()]
    xmax=xlist[xlist.argmax()]#ss[ss[:,0].argmax()]
    ymax=ylist[ylist.argmax()]#ss[ss[:,1].argmax()]
    return xmin,ymin,xmax-xmin+1,ymax-ymin+1
def IsInRegion(x0, y0, w0, h0,point):
    #x0, y0, w0, h0=GetMinMax(contours)
    x=point[0]
    y=point[1]
    if x>=x0 and x<=(x0+w0) and y>=y0 and y<=(y0+h0):
        return True
    else:
        return False
def IsInContourRegion(contours,point):
    #x0, y0, w0, h0=GetMinMax(contours)
    if(len(contours)==0):
        return False
    x=point[0]
    y=point[1]
    dist = cv2.pointPolygonTest(contours[0], point, measureDist=True)
    #if x>=x0 and x<=(x0+w0) and y>=y0 and y<=(y0+h0):
    if abs(dist)<=5.1:
        return True
    else:
        return False
def VoteInRegion(contours,stroke,Speedup=True):
    x0, y0, w0, h0=GetMinMax(contours)
    cnt=0
    if Speedup:
        for point in stroke:
            if(IsInRegion(x0, y0, w0, h0,point)):
                cnt+=1
    else:
        for point in stroke:
            if(IsInContourRegion(contours,point)):
                cnt+=1
    return cnt
def GetVoteInRegions(contours_list,stroke):
    voteList=[]
    contours_index_list=[0,1,3,4,5,6,10,11,12]#[1,3,4,5,6,10,11]
    
    for idx in contours_index_list:
        name=label_list[idx]
        contours=contours_list[name]
        if idx!=0 and idx!=12:
            vote=VoteInRegion(contours,stroke)
        else:
            vote=VoteInRegion(contours,stroke,False)
        voteList.append(vote)
    #print(voteList)
    voteArray=np.array(voteList)
    label_idx=contours_index_list[voteArray.argmax()]
    label_name=label_list[label_idx]
    #print(label_name)
    return label_name
def GetMaskContours(Img_mat,showPreview=False): 
    #ContourList=[]
    ContourList={}

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

        gray = np.array(tmp_im)

        if(showPreview):
            x, y, w, h = cv2.boundingRect(gray)
            print('showPreview',x, y, w, h)
            color_img = cv2.cvtColor(np.array(gray),cv2.COLOR_GRAY2BGR)
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.imshow(color_img,'gray')
            plt.show()
            
        ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if(showPreview):
            x0, y0, w0, h0=GetMinMax(contours)
            print('GetMinMax',x0, y0, w0, h0)


        ContourList[label_list[label_idx]]=contours

    return ContourList

def point_contour_dist(img, hull, point, text, measure_dist=True):
    dist = cv2.pointPolygonTest(hull, point, measure_dist)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, point, font, 1, (0, 255, 0), 3)
    print("dist%s=" % text, dist)
    return dist   
def remake_mask(strokes,ori_mask):

    mat_img=ori_mask.copy().astype(np.uint8)
    cts=GetMaskContours(mat_img,False)
    DrawMaskContours(mat_img,cts)
    marks=[]
    for stroke in strokes:
        #print(stroke)
        simple_stroke=rdp(stroke, 1.0)
        #print(simple_stroke)
        ss=np.array(simple_stroke)
        mark=GetVoteInRegions(cts,simple_stroke)
        marks.append(mark)
        
    outimg=DrawMaskContoursWithMarks(mat_img,cts,strokes,marks)#-1 ==cv2.FILLED
    return ConverColorToMark(outimg),outimg#,ori_mask
