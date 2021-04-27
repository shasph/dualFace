# -*- coding: utf-8 -*-
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import os
import socket
import sys
import time
from ui.util2 import *

import pickle#load,save list

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(204, 0, 204), QColor(51, 51, 255), QColor(0, 255, 255), QColor(51, 255, 255), QColor(255, 0, 0), QColor(102, 51, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

class GraphicsScene(QGraphicsScene):
    def __init__(self, mode, size, parent=None):
        QGraphicsScene.__init__(self, parent)
        #print(self.size()) # outputs QSize(512, 512)
        #self.setSceneRect(20,120,512,512)
        self.mono_mask = None
        self.mono_mask_show=False
        self.mask_img_base='imgdata/mask'#'
        self.query_shadow=True#True\False
        self.graphicsView = None
        self.PosX=20
        self.PosY=120
        self.mode = mode
        self.size = size#stroke size
        #self.mouse_clicked = False
        self.drawCnt = 0
        self.prev_pt = None
        self.closest_face_fname = None
        self.closer_face_fname_list = []

        self.this_mouse_stroke=[]
        self.this_mouse_strokes=[]
        self.this_mouse_strokes_mark=[]
        self.line_size=[]#.append(self.size)
        self.line_color_list=[]
        self.current_color=(0, 0, 0)
        # self.masked_image = None
        # self.cv_image = None
        self.shadow_limit = 3
        self.shadow_pil = None
        self.shadow_show = True
        self.global_stage = True
        # save the points
        self.mask_points = []
        for i in range(len(color_list)):
            self.mask_points.append([])

        # save the size of points
        self.size_points = []
        for i in range(len(color_list)):
            self.size_points.append([])

        # save the history of edit
        self.history = []
        
        lineItem = QGraphicsLineItem(0,0,0,0)
        self.addItem(lineItem)
        self.removeItem(lineItem)
        
        self.strokes_path="temp/strokes_save"#
        self.strokes_width_path="temp/strokes_width_save"#
        self.strokes_color_path="temp/strokes_color_save"
        self.callbk= None
        self.sse_flist=self.read_sse_list()
        self.mouse_left=False
        self.mouse_right=False
        
        
    def reset(self):
        # save the points
        self.mask_points = []
        for i in range(len(color_list)):
            self.mask_points.append([])
        # save the size of points
        self.size_points = []
        for i in range(len(color_list)):
            self.size_points.append([])
        # save the history of edit
        self.history = []

        self.mode = 0
        self.prev_pt = None
        
        self.this_mouse_stroke=[]
        self.this_mouse_strokes=[]
        self.line_size=[]
        self.line_color_list=[]
        self.current_color=(0, 0, 0)
        self.shadow_pil = None
        self.mono_mask_show =False
        self.query_shadow = True
    #def changed(self, event):
    #    print("Scene Update!")
    def mousePressEvent(self, event):
        #self.mouse_clicked = True
        self.this_mouse_stroke=[]
        if(event.buttons() == Qt.LeftButton):
            self.mouse_left=True
        elif(event.buttons() == Qt.RightButton):
            self.mouse_right=True
            self.stroke_del(event)
        #if self.prev_pt is None:
        #   self.prev_pt = event.scenePos()
            #print(self.prev_pt)
    def setShadowImage(self):
        if(self.shadow_pil is None):
            return
        im = self.shadow_pil
        data = im.tobytes('raw', 'RGB')
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim)
        pix=self.set_pixmap_alpha(pix)
        tst=self.addPixmap(pix)
        tst.setPos(self.PosX, self.PosY)  
        #self.setShadowImage_toRed()
    def setShadowImage_toRed(self,colorConvert=True):
        img_pil=self.mono_mask
        if(img_pil is None):
            return
        colorConvert=self.shadow_show
        im = img_pil#.copy()
        if colorConvert:
            data = np.array(im)
            red, green, blue = data.T
            black_areas = (red == 0) & (blue == 0) & (green == 0)
            data[black_areas.T] = (255, 0, 0) # Transpose back needed
            im = Image.fromarray(data)
        data = im.tobytes('raw', 'RGB')
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim)
        pix=self.set_pixmap_alpha(pix)
        tst=self.addPixmap(pix)
        tst.setPos(self.PosX, self.PosY)  
    def queryShadow(self):
        if self.query_shadow:
            path=os.path.join(os.getcwd(),'temp','query_img.png')
            qstr=self.query(path)
            self.closest_face_fname=qstr[0]
            self.closer_face_fname_list=qstr
            self.shadow_pil =self.getShadow(qstr)
                    
            for item in self.items():
                obj_type = type(item)
                #print(obj_type)
                #if(obj_type==QPixmap):
                if isinstance(item, QGraphicsPixmapItem):
                    self.removeItem(item)
                    print("remove QGraphicsPixmapItem")
            #qim = ImageQt(self.shadow_pil
            """
            im = self.shadow_pil
            data = im.tobytes('raw', 'RGB')
            qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
            
            pix = QPixmap.fromImage(qim)
            #pix.save("dddd1.png");
            pix=self.set_pixmap_alpha(pix)
            
            tst=self.addPixmap(pix)
            #tst=self.addPixmap(QPixmap('\capture.png'))
            tst.setPos(20, 120)
            """

        self.Refresh()
        if self.shadow_show:
            self.setShadowImage() 
        if self.mono_mask_show:
            self.setShadowImage_toRed()     
    #def keyPressEvent(self, event):
    #    self.undo()
    def stroke_del(self,event):
        #print("stroke_del")
        #only for global
        #drawSingleStroke
        if(self.global_stage):
            refresh_flag=False
            this_pt=event.scenePos()
            point=(this_pt.x()-20.0,this_pt.y()-120.0)
            print("stroke_del",point)
            this_mouse_strokes=[]
            line_size=[]
            line_color_list=[]
            for i in range(len(self.this_mouse_strokes)):
                cv_mat=self.drawSingleStroke(self.this_mouse_strokes,self.line_size,self.line_color_list,i)
                color=cv_mat[int(point[1]), int(point[0])]
                if(all(color==(0,0,0))):
                    print(i,"th stroke is deleted")
                    refresh_flag=True
                else:
                    this_mouse_strokes.append(self.this_mouse_strokes[i])
                    line_size.append(self.line_size[i])
                    line_color_list.append(self.line_color_list[i])
            if refresh_flag:
                self.this_mouse_strokes=this_mouse_strokes
                self.line_size=line_size
                self.line_color_list=line_color_list
                self.Refresh()
                self.save_tmp_img(path=os.path.join(os.getcwd(),'temp','query_img.png'))
                self.queryShadow()
            #print("RGB: ",all(color==(0,0,0)) )
    def mouseReleaseEvent(self, event):
        if self.mouse_right:
            self.mouse_right=False
            return
        if not self.mouse_left:
            return
            
        self.mouse_left=False
        
        self.prev_pt = None
        #self.mouse_clicked = False
        if(len(self.this_mouse_stroke)>0):
            print("strokes",self.this_mouse_stroke)
            self.this_mouse_strokes.append(self.this_mouse_stroke)
            self.line_size.append(self.size)
            self.line_color_list.append(self.current_color)
            self.this_mouse_stroke=[]
            self.save_tmp_img(path=os.path.join(os.getcwd(),'temp','query_img.png'))
            self.queryShadow()
        
        if self.callbk is not None:
            self.caller(self.callbk)
    #part1:get mark of this_mouse_strokes
    def ClassifyByPos(self,Contours,pointlist):
        for name,contours in Contours.items():
            
            return name
    def MarkStrokes(self,mask):
        marks=[]
        mask_mat=mask
        Contours=GetMaskContours(mask_mat)
        if(self.this_mouse_strokes is not None):
            for stroke in self.this_mouse_strokes:
                res=ClassifyByPos(Contours,stroke)
                marks.append(res)
        
        self.this_mouse_strokes_mark=marks
        return
    #part1-end:get mark of this_mouse_strokes
    #part2:mask merge
    #dis(start,end)<0.1 or overlay in last 1/2parts,closed
    #if stroke closed, true->instead false->merge convexHull
    
    #part2-end:mask merge
    #self.this_mouse_strokes
    def strokes_save(self):
        with open(self.strokes_path, "wb") as fp:   #Pickling
            pickle.dump(self.this_mouse_strokes, fp)
        with open(self.strokes_width_path, "wb") as fp:   #Pickling
            pickle.dump(self.line_size, fp)
        with open(self.strokes_color_path, "wb") as fp:   
            pickle.dump(self.line_color_list, fp)
        self.save_tmp_img(path=os.path.join(os.getcwd(),'temp','user_sketch.png'))
    def strokes_load1(self):
        with open(self.strokes_path+'_auto', "rb") as fp:   #Unpickling
             self.this_mouse_strokes = pickle.load(fp)      
        with open(self.strokes_width_path+'_auto', "rb") as fp:   #Unpickling
             self.line_size = pickle.load(fp)   
        with open(self.strokes_color_path+'_auto', "rb") as fp:   #Unpickling
             self.line_color_list = pickle.load(fp) 
            
        self.Refresh()
        self.setShadowImage() 
    def strokes_load(self):
        with open(self.strokes_path, "rb") as fp:   #Unpickling
             self.this_mouse_strokes = pickle.load(fp)      
        with open(self.strokes_width_path, "rb") as fp:   #Unpickling
             self.line_size = pickle.load(fp)   
        with open(self.strokes_color_path, "rb") as fp:   #Unpickling
             self.line_color_list = pickle.load(fp) 
        self.Refresh()
        self.setShadowImage()
    def set_pixmap_alpha(self, p1, alpha=0.2):#p2, mode=QtGui.QPainter.CompositionMode_SourceOver):
        #s = p1.size().expandedTo(p2.size())
        s = p1.size()
        result =  QPixmap(s)
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setOpacity(alpha)
        #painter.setRenderHint(QPainter.Antialiasing)
        #painter.drawPixmap(result.rect(), p1)
        painter.drawPixmap(0, 0, p1);
        #painter.setCompositionMode(mode)
        #painter.drawPixmap(result.rect(), p2, p2.rect())
        painter.end()
        #result.save("dddd.png");
        return result
    def getShadow(self, qstr,limit=3):
        limit=self.shadow_limit
        #self.mask_img_base='imgdata/mask'#'H:/re/code/opensse-master/bin/tmp/data'
        i=0
        for q in qstr:
            path=os.path.join(self.mask_img_base,'{0:05d}.jpg'.format(int(q)))
            print(path)
            im_tensor =image_load_toTensor(path).unsqueeze(0)
            if i==0:
                shadow = im_tensor
            elif(i>=limit):
               break
            else:
                shadow =  torch.cat((shadow,im_tensor),0)
            i+=1
        #print(shadow.shape)
        results_shadow=torch.mean(shadow,0).squeeze(0)
        #results_shadow=np.transpose(results_shadow, (1,2,0))
        
        pil_img=tensor_to_PIL(results_shadow)
        #pil_img.show()

        return pil_img
        
    def mouseMoveEvent(self, event): # drawing
        #print('mouseMoveEvent')
        if self.mouse_left:#self.mouse_clicked:
            this_pt=event.scenePos()
            #print(this_pt)
            point=(this_pt.x()-20.0,this_pt.y()-120.0)
            #print(point)
            self.this_mouse_stroke.append(point)
            if self.prev_pt:
                
                #print('drawMask')
                #self.drawMask(self.prev_pt, event.scenePos(), color_list[self.mode], self.size)
                #self.drawMask(self.prev_pt, this_pt, color_list[0], self.size)
                self.drawMask(self.prev_pt, this_pt, QColor(self.current_color[0],self.current_color[1],self.current_color[2]), self.size)
                pts = {}
                pts['prev'] = (int(self.prev_pt.x()),int(self.prev_pt.y()))
                pts['curr'] = (int(this_pt.x()),int(this_pt.y()))
        
                self.size_points[self.mode].append(self.size)
                self.mask_points[self.mode].append(pts)
                self.history.append(self.mode)
                self.prev_pt = this_pt
            else:
                self.prev_pt = this_pt

    def drawMask(self, prev_pt, curr_pt, color, size):
        self.drawCnt+=1
        if(False):
            return
        else:
            #print(self.prev_pt,curr_pt)
            #lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
            lineItem = QGraphicsLineItem()
            lineItem.setFlag(QGraphicsItem.ItemIsMovable)
            lineItem.setPen(QPen(color, size, Qt.SolidLine)) 
            lineItem.setLine(prev_pt.x(),prev_pt.y(),curr_pt.x(),curr_pt.y()) # rect

            self.addItem(lineItem)
            #I don't know why this will happen #setSceneRect!!!!
            if(self.drawCnt<0):
                self.undo()
                if(self.drawCnt==1):
                    #lineItem2 = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
                    lineItem.setPen(QPen(QColor(255, 255, 255), size, Qt.SolidLine))
                    self.addItem(lineItem)
                #self.removeItem(lineItem)
        self.update()
        #self.save_tmp_img()

    def erase_prev_pt(self):
        self.prev_pt = None

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)
        
    def undo0(self):
        if len(self.items())>0:
            if len(self.items())>=9:
                for i in range(len(self.items())):
                #for i in range(8):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == self.mode:
                        print(self.mask_points[self.mode][-1])
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()
            else:
                for i in range(len(self.items())-1):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == self.mode:
                        print('k',self.mask_points[self.mode][-1])
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()
                        
    def drawStrokes(self,strokes,thicknessList,colorList):
        img = np.ones((512, 512, 3), np.uint8)*255#np.zeros((512, 512, 3), np.uint8)
        ptStart=None
        ptEnd=None
        point_color = (0, 0, 0) # BGR
        point_color2 = (255, 255, 255)
        lineType = 4
        idx=0
        for stroke in strokes:
            thickness=thicknessList[idx]
            point_color=colorList[idx]
            for point in stroke:
                point_int=(int(point[0]),int(point[1]))
                if(ptStart is None):
                    ptStart=point_int
                    continue
                ptEnd=point_int
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                ptStart=ptEnd
            ptStart=None
            idx+=1
            
        return img
    def drawSingleStroke(self,strokes,thicknessList,colorList,index):
        img = np.ones((512, 512, 3), np.uint8)*255#np.zeros((512, 512, 3), np.uint8)
        ptStart=None
        ptEnd=None
        point_color = (0, 0, 0) # BGR
        point_color2 = (255, 255, 255)
        lineType = 4
        idx=0
        for stroke in strokes:
            if(index==idx):
                thickness=thicknessList[idx]
                point_color=colorList[idx]
                for point in stroke:
                    point_int=(int(point[0]),int(point[1]))
                    if(ptStart is None):
                        ptStart=point_int
                        continue
                    ptEnd=point_int
                    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                    ptStart=ptEnd
                ptStart=None
            idx+=1
            
        return img    
    def drawStrokes0(self,strokes,thicknessList):
        img = np.ones((512, 512, 3), np.uint8)*255#np.zeros((512, 512, 3), np.uint8)
        ptStart=None
        ptEnd=None
        point_color = (0, 0, 0) # BGR
        point_color2 = (255, 255, 255)
        lineType = 4
        idx=0
        for stroke in strokes:
            thickness=thicknessList[idx]
            for point in stroke:
                point_int=(int(point[0]),int(point[1]))
                if(ptStart is None):
                    ptStart=point_int
                    continue
                ptEnd=point_int
                cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
                ptStart=ptEnd
            ptStart=None
            idx+=1
            
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        return img
    
    def undo(self):
        if len(self.items())>0:
            for i in range(len(self.items())):
                item = self.items()[0]
                self.removeItem(item)
                if len(self.history)>0:
                    if self.history[-1] == self.mode:
                        #print(self.mask_points[self.mode][-1])
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()         
                    
        if len(self.this_mouse_strokes)>0:
            self.this_mouse_strokes.pop()
            self.line_color_list.pop()
            self.line_size.pop()
            self.redraw()
    def Refresh(self):
        if len(self.items())>0:
            for i in range(len(self.items())):
                item = self.items()[0]
                self.removeItem(item)
                if len(self.history)>0:
                    if self.history[-1] == self.mode:
                        #print(self.mask_points[self.mode][-1])
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()         
        if len(self.this_mouse_strokes)>0:
            self.redraw()
    def redraw(self):
        cv_mat=self.drawStrokes(self.this_mouse_strokes,self.line_size,self.line_color_list)
        mat_img = cv2.cvtColor(np.array(cv_mat),cv2.COLOR_RGB2BGR)
        Qimage_new= QImage(mat_img, 512, 512, QImage.Format_RGB888)
        pix = QPixmap.fromImage(Qimage_new)
        tst=self.addPixmap(pix)
        tst.setPos(self.PosX, self.PosY)   
    def caller(self, func):
        path=os.path.join(os.getcwd(),'temp','query_img.png')
        qstr=self.query(path)
        func(qstr)
    def save_tmp_img(self,path): 
        self.Refresh()
        # Create a QImage to render to and fix up a QPainter for it.
        area= QRectF(0.0, 0.0, 512.0, 512.0)
        area2= QRectF(20.0, 120.0, 512.0, 512.0)
        image = QImage(area.width(), area.height(), QImage.Format_ARGB32_Premultiplied)

        painter = QPainter(image)

        # Render the region of interest to the QImage.
        self.render(painter, area, area2)
        painter.end()

        # Save the image to a file.
        #path=os.path.join(os.getcwd(),'temp','query_img.png')
        image.save(path)
        #image.save("capture.png")
    def save_tmp_img2(self):
        # Get region of scene to capture from somewhere.
        view = QGraphicsView(self)
        pixmap = QPixmap(QRect(20, 120, 512, 512).size())#QPixmap(view.viewport().size())
        pixmap.fill()
        area = view.viewport().rect()#self.sceneRect()#QRect(0,0,512,512)#get_QRect_to_capture_from_somewhere()
        view.viewport().render(pixmap)
        
        #print(area)
        # Create a QImage to render to and fix up a QPainter for it.
        #image = QImage(area.size(), QImage.Format_RGB888)
        #painter = QPainter(image)

        # Render the region of interest to the QImage.
        #self.render(painter, image.rect(), area)
        #painter.end()
        
        path=os.path.join(os.getcwd(),'temp','query_img.png')
        # Save the image to a file.
        #image.save(path)
        pixmap.save(path)
    def read_sse_list(self):
        f = open(r"sse\filelist","r")   
        lines = f.readlines()
        f.close()
        #print(lines)
        print('Load sse list with length ',len(lines))
        return lines
    def query(self,fname):
        
        HOST, PORT = "localhost", 50007
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        
        start_t = time.time()
        
        s.send(fname.encode())
        resStr=str(s.recv(1000).decode())
        splitRes=resStr.split(',')
        
        end_t = time.time()
        exlist=[]#
        new_splitRes=[]
        for i in splitRes:
            fname=self.sse_flist[int(i)]
            fidx=int(fname[:-5])
            if(fidx in exlist):
                continue
            else:
                new_splitRes.append(fidx)
      
        print('global query time : {}'.format(end_t-start_t))
        print('querylist=',new_splitRes)
        return new_splitRes