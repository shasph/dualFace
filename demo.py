import os
import sys
import cv2
import time
import random
import numpy as np
from PIL import Image,ImageQt

import torch
from torchvision.utils import save_image

from evaluate_one_Mask import evaluate_one,get_mask_net
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from APDrawing import CallAPdrawModel,GetAPOption,GetAPdrawModel,GetUpdatedAPdrawDataset
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]
color_list_p = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [51, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

class Ex(QWidget, Ui_Form):
    def __init__(self, model,opt,AP_model,AP_opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.opt = opt
        
        self.mask_net=get_mask_net(cp='79999_iter.pth')
        self.AP_model = AP_model
        self.AP_opt = AP_opt
        self.output_img = None
        self.CheckMask=True #
        self.NoIcon=False

        #if self.CheckMask:
        #    self.resize(1260, 660)
        self.mat_img = None

        self.mode = 0
        self.shadow_no=0
        self.size = self.sld.value()
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.GotDetails = False
        self.UseBGMask = True
        self.shadow_mode = 0
        
        self.scene = GraphicsScene(self.mode, self.size)
        self.scene.setSceneRect(20,120,512,512)
        self.scene.graphicsView = self.graphicsView
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.ref_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
 
        self.result_scene = QGraphicsScene()#GraphicsScene(self.mode, self.size)#QGraphicsScene()
        #self.result_scene.query_shadow=False
        #self.result_scene.setSceneRect(1260,120,512,512)
        #self.result_scene.PosX=1260
        #self.result_scene.PosY=120
        #self.result_scene.graphicsView = self.graphicsView_3
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None
        
        self.IconPen= QPixmap("icon/pen.png")
        self.IconEr= QPixmap("icon/eraser.png")
        self.IconSave= QPixmap("icon/save.png")
        self.IconLoad= QPixmap("icon/load.png")
        self.IconUndo= QPixmap("icon/undo.png")
        self.IconGenDetails= QPixmap("icon/gen-details.png")
        self.IconSwitch= QPixmap("icon/switch.png")
        
        if not self.NoIcon:
            self.pushButton.setText("")#GenDetails
            self.pushButton_4.setText("")#Load
            self.pushButton_5.setText("")#Undo
            self.pushButton_6.setText("")#Save      
             
            self.pushButton.setIcon(QIcon(self.IconGenDetails))#self.pushButton.setIcon(QIcon(self.IconSwitch))
            self.pushButton.setIconSize(self.pushButton.rect().size())       
            self.pushButton_4.setIcon(QIcon(self.IconLoad))
            self.pushButton_4.setIconSize(self.pushButton_4.rect().size())
            self.pushButton_5.setIcon(QIcon(self.IconUndo))
            self.pushButton_5.setIconSize(self.pushButton_5.rect().size())
            self.pushButton_6.setIcon(QIcon(self.IconSave))
            self.pushButton_6.setIconSize(self.pushButton_6.rect().size())
        
        self.IsEraser=False
        self.pushButton_7.setIcon(QIcon(self.IconPen))
        self.pushButton_7.setIconSize(self.pushButton_7.rect().size())
        self.pushButton_7.setVisible(False)
        
        self.pushButton_8.setVisible(False)
        #self.pushButton_7.setFlat(True)
        self.undoAct= QAction('t',self)
        self.undoAct.setShortcut('Ctrl+z')
        self.undoAct.setStatusTip('Undo')
        self.undoAct.triggered.connect(self.undo)
        self.addAction(self.undoAct)
        self.scene.query_shadow = True#False
        self.img_base='imgdata/CelebAMask-HQ'
        #self.scene.shadow_pil=Image.open('10003-mask-mono.png').resize((512,512))
        #self.scene.setShadowImage()
    def changePen(self,value):
        if self.IsEraser:
            self.pushButton_7.setIcon(QIcon(self.IconPen))
            self.pushButton_7.setIconSize(self.pushButton_7.rect().size())
            self.scene.current_color=(0,0,0)
        else:
            self.pushButton_7.setIcon(QIcon(self.IconEr))
            self.pushButton_7.setIconSize(self.pushButton_7.rect().size())
            self.scene.current_color=(255,255,255)
        #print(self.IsEraser)
        self.IsEraser =bool(1-self.IsEraser)
    def changeSilderValue(self,value):
        self.sld_label.setText("Stroke Width: "+str(value))
        self.scene.size=value
        #self.result_scene.size=value
        #print('sld=',value)
    def changeShadowLimitSilderValue(self,value):
        self.sld_labe2.setText("Shadow Number: "+str(value))
        print('changeSilderValue=',value)
        self.scene.shadow_limit=value

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).resize((512,512))
            fileName='temp/input_image.png'
            mat_img.save(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            #self.ref_scene.addPixmap(image)
 
            
            #=evaluate_one()
            #including mask idx convert
            pimg,bgPIL,maskPIL,maskContours,cv_mat_mask,_=evaluate_one(self.mask_net,img_path=fileName)
            self.get_mask(cv_mat_mask)
            #maskPIL.show()
            #apd_pil=APdraw(fileName,bgPIL)
            datasetAP=GetUpdatedAPdrawDataset(self.AP_opt,fileName,bgPIL)
            apd_pil=CallAPdrawModel(self.AP_model,datasetAP)
            #maskPIL.show()
            apd_qt = ImageQt.toqpixmap(apd_pil)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(apd_qt)
            #pimg = transforms.ToPILImage()(pimg)
            #pass

            #pimg.show() Mask_img
            mat_img = cv2.cvtColor(np.array(pimg),cv2.COLOR_RGB2BGR)
            #cv2.imshow("OpenCV",mat_img)
            #cv2.waitKey(0)
            self.mask = mat_img.copy()
            self.mask_m = mat_img
            image_mask = QImage(mat_img, 512, 512, QImage.Format_RGB888)
            
            #mask color convert
            if True:
                for i in range(512):
                    for j in range(512):
                        r, g, b, a = image_mask.pixelColor(i, j).getRgb()
                        image_mask.setPixel(i, j, color_list[r].rgb()) 
            
            #qimage = ImageQt.ImageQt(maskPIL)
            pixmap = QPixmap()
            #pixmap.convertFromImage(qimage)  
            pixmap.convertFromImage(image_mask) 
            pixmap=ImageQt.toqpixmap(maskPIL)            
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            tst=self.scene.addPixmap(self.image)
            tst.setPos(20, 120)
            
    def Mask_InferenceUI2(self,idx=0):       
        result_name='temp/input_image.png'
        if(idx>0):
            result_name='temp/input_image_{}.png'.format(idx)  
        return result_name    
    def Mask_Inference(self,idx=0):
              
        #set self.mask,self.mask_m
        params = get_params(self.opt, (512,512))
        transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
        transform_image = get_transform(self.opt, params)
        
        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask = transform_mask(Image.fromarray(np.uint8(mask))) 
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(self.img)
    
        start_t = time.time()
        generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
        end_t = time.time()
        #print('inference time : {}'.format(end_t-start_t))
        #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()#.cpu().numpy()#
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        
        #qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)
        qim = QImage(bytes(result.data), result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)

        #if len(self.ref_scene.items())>0:
        #    self.ref_scene.removeItem(self.ref_scene.items()[-1])
        #    self.ref_scene.addPixmap(qim)
          
        result_name='temp/ref_result.png'
        if(idx>0):
            result_name='temp/ref_result_%d.png'%idx
        img_pil = ImageQt.fromqpixmap(qim)
        img_pil.save(result_name)
        
        return result_name
    def mask_convert_to_3channel(self,mask):
        a=np.array(mask)
        a=a[:, : ,np.newaxis]
        b=np.concatenate((a,a,a),axis=2)
        return b
        
    def switch_shadow(self):
        self.shadow_no+=1
        self.scene.Refresh()
        if self.shadow_no % 3 ==0:
            self.shadow_no=0
        self.switch_shadowPIL(self.shadow_no)
        
    def switch_shadowPIL(self,idx):
        self.shadow_mode=0
        self.scene.shadow_show = True
        self.scene.mono_mask_show =True
        
        monomask_path='temp/autocompelete_mono_mask_{}.png'.format(idx)
        shadow_path='temp/output_image_final_{}.png'.format(idx)
        if(idx==0):
            monomask_path='temp/autocompelete_mono_mask.png'
            shadow_path='temp/output_image_final.png'
        if not os.path.exists(monomask_path):
            return
        if not os.path.exists(shadow_path):
            return
        self.scene.shadow_pil=Image.open(shadow_path)#apd_pil
        self.scene.query_shadow=False
        self.scene.mono_mask=Image.open(monomask_path)#maskPIL
        self.scene.setShadowImage_toRed()
        self.scene.setShadowImage()
        
        return
    def switch_mode(self):
        self.shadow_mode+=1
        self.scene.Refresh()
        if self.shadow_mode % 3 ==0:
            self.shadow_mode=0
            self.scene.shadow_show = True
            self.scene.mono_mask_show =True
            self.scene.setShadowImage() 
            self.scene.setShadowImage_toRed() 
        if self.shadow_mode % 3 ==1:
            self.scene.shadow_show =True
            self.scene.mono_mask_show =False
            self.scene.setShadowImage() 
        if self.shadow_mode % 3 ==2:
            self.scene.shadow_show =False
            self.scene.mono_mask_show =True
            self.scene.setShadowImage_toRed() 
    def ap_reset(self):
        button = QMessageBox.question(self,"Warning","Back to global stage will clear up current data. Are you sure?",QMessageBox.Yes | QMessageBox.No)
        if button == QMessageBox.Yes:
            print('reset')
            self.GotDetails=False
            self.scene.reset()
            self.scene.reset_items()
            self.scene.Refresh()
            if self.NoIcon:
                self.pushButton.setText("genDetails")
                self.pushButton.setGeometry(QRect(250, 40, 81, 27))
            else:
                self.pushButton.setIcon(QIcon(self.IconGenDetails))
                self.pushButton.setIconSize(self.pushButton.rect().size())
            
            self.pushButton_7.setVisible(False)
            self.IsEraser=False
            self.pushButton_7.setIcon(QIcon(self.IconPen))
            self.pushButton_7.setIconSize(self.pushButton_7.rect().size())
            
            self.pushButton_8.setVisible(False)
            
        else:
            self.rButton2.setChecked(True)
        return
    def redio1_clicked(self):
        if self.rButton1.isChecked():
            self.ap_reset()
    def redio2_clicked(self):
        if self.rButton2.isChecked():
            print('genDetails')
            self.genDetails()
    def redio_toggled(self):
        if self.rButton1.isChecked():
            self.ap_reset()
        if self.rButton2.isChecked():
            print('genDetails')
            self.genDetails()
    def genDetailInList(self,limit=2):
        length=len(self.scene.closer_face_fname_list)
        for i in range(1,length,1):
            if(i>limit):
                return
            fname=os.path.join(self.img_base,'{0}.jpg'.format(int(self.scene.closer_face_fname_list[i])))
            print('genOneDetail',fname,i)
            self.genOneDetail(fname,i)
                
    def genOneDetail(self,fileName,idx):
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).resize((512,512))
            fileName='temp/input_image_{}.png'.format(idx)
            mat_img.save(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            pimg,bgPIL,maskPIL,maskContours,cv_mat_mask,color_modmask_cvmat=evaluate_one(self.mask_net,self.scene.this_mouse_strokes,img_path=fileName)
            maskPIL.save('temp/autocompelete_mono_mask_{}.png'.format(idx))
            cv2.imwrite('temp/modmask_{}.png'.format(idx),color_modmask_cvmat)
            
            mskBG=np.array(bgPIL).astype(np.uint8)
            self.mask = self.mask_convert_to_3channel(cv_mat_mask)
            self.mask_m = self.mask_convert_to_3channel(pimg)

            fileName=self.Mask_Inference(idx)#save ref_result
            datasetAP=GetUpdatedAPdrawDataset(self.AP_opt,fileName,bgPIL)
            apd_pil=CallAPdrawModel(self.AP_model,datasetAP)
            #maskPIL.show()
            #apd_pil.save('temp/output_image_{}.png' % idx)
            if self.UseBGMask:
                ap_cv_img = cv2.cvtColor(np.asarray(apd_pil),cv2.COLOR_RGB2BGR)  
                res = cv2.bitwise_and(ap_cv_img,ap_cv_img,mask=mskBG)
                res[mskBG==0,:]=255
                apd_pil = Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGR2RGB)) 
            apd_pil.save('temp/output_image_final_{}.png'.format(idx))
    def genDetails(self):
        if(self.GotDetails):
            #self.switch_shadow()
            self.switch_mode()
            return
        #self.img_base='imgdata/CelebAMask-HQ'#'H:/re/code/opensse-master/bin/tmp/data'
        if self.scene.closest_face_fname:
            fileName=os.path.join(self.img_base,'{0}.jpg'.format(int(self.scene.closest_face_fname)))
        else:
            fileName=os.path.join(self.img_base,'{0}.jpg'.format(random.randint(0,20)))#'temp/10003.jpg'
        print('base-img',fileName)
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).resize((512,512))
            fileName='temp/input_image.png'
            mat_img.save(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
                
            self.scene.strokes_save()  
            pimg,bgPIL,maskPIL,maskContours,cv_mat_mask,color_modmask_cvmat=evaluate_one(self.mask_net,self.scene.this_mouse_strokes,img_path=fileName)
            cv2.imwrite('temp/modmask.png',color_modmask_cvmat)
            #cv2.imwrite('temp/original_mask.png',cv_mat_mask)
            #pil_modmask=Image.open('temp/modmask.png')autocompelete_mono_mask
            pil_modmask=maskPIL#Image.open('temp/autocompelete_mono_mask.png')
            pixmap_modmask = ImageQt.toqpixmap(pil_modmask)
            self.ref_scene.addPixmap(pixmap_modmask)
            if self.CheckMask:
                self.resize(1260, 660)
                button = QMessageBox.question(self,"Warning","Do you think this mask is reasonable?",QMessageBox.Yes | QMessageBox.No)
                if button == QMessageBox.No:
                    self.resize(560, 660)
                    self.rButton1.setChecked(True)
                    return
            self.resize(560, 660)
            self.GotDetails=True
            self.rButton2.setChecked(True)
            mskBG=np.array(bgPIL).astype(np.uint8)
            #print(mskBG.shape) (512,512)
            #bgPIL.show()
            self.mask = self.mask_convert_to_3channel(cv_mat_mask)
            self.mask_m = self.mask_convert_to_3channel(pimg)
            #print(np.array(self.mask).shape,np.array(self.mask_m).shape)
            start_t0 = time.time()
            fileName=self.Mask_Inference()

            #pimg,bgPIL,maskPIL,maskContours,cv_mat_mask=evaluate_one(self.mask_net,self.scene.this_mouse_strokes,img_path=fileName)

            datasetAP=GetUpdatedAPdrawDataset(self.AP_opt,fileName,bgPIL)
            apd_pil=CallAPdrawModel(self.AP_model,datasetAP)
            end_t0 = time.time()
            #maskPIL.show()
            apd_pil.save('temp/output_image.png')
            if self.UseBGMask:
                ap_cv_img = cv2.cvtColor(np.asarray(apd_pil),cv2.COLOR_RGB2BGR)  
                res = cv2.bitwise_and(ap_cv_img,ap_cv_img,mask=mskBG)
                res[mskBG==0,:]=255
                #cv2.imshow("OpenCV",res)  
                #cv2.waitKey(0)
                apd_pil = Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGR2RGB)) 
            apd_pil.save('temp/output_image_final.png')
            apd_qt = ImageQt.toqpixmap(apd_pil)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(apd_qt)
            #pimg = transforms.ToPILImage()(pimg)
            #pass

            #pimg.show() Mask_img
            mat_img = cv2.cvtColor(np.array(pimg),cv2.COLOR_RGB2BGR)
            #cv2.imshow("OpenCV",mat_img)
            #cv2.waitKey(0)
            self.mask = mat_img.copy()
            self.mask_m = mat_img
            image_mask = QImage(mat_img, 512, 512, QImage.Format_RGB888)
            
            #mask color convert
            if True:
                for i in range(512):
                    for j in range(512):
                        r, g, b, a = image_mask.pixelColor(i, j).getRgb()
                        image_mask.setPixel(i, j, color_list[r].rgb()) 
            
            #qimage = ImageQt.ImageQt(maskPIL)
            pixmap = QPixmap()
            #pixmap.convertFromImage(qimage)  
            pixmap.convertFromImage(image_mask) 
            pixmap=ImageQt.toqpixmap(maskPIL)   
            maskPIL.save('temp/autocompelete_mono_mask.png')            
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
                    
            self.scene.strokes_save()
            self.scene.strokes_path="temp/strokes_save_auto"
            self.scene.strokes_width_path="temp/strokes_width_save_auto"
            self.scene.strokes_color_path="temp/strokes_color_save_auto"
            self.scene.strokes_save()
            self.scene.strokes_path="temp/strokes_save"
            self.scene.strokes_width_path="temp/strokes_width_save"
            self.scene.strokes_color_path="temp/strokes_color_save"
            start_t1 = time.time()
            self.genDetailInList()
            end_t1 = time.time()
            t0=end_t0-start_t0
            t1=end_t1-start_t1
            print('local time : {}+{}={}'.format(t0,t1,t0+t1))
            self.scene.reset()
            self.scene.reset_items()
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(self.image)
            #if len(self.scene.items())>0:
            #    self.scene.reset_items() 
            #tst=self.scene.addPixmap(self.image)
            #tst.setPos(20, 120)
            
            
            #set shadow for local stage
            self.scene.shadow_pil=apd_pil
            self.scene.query_shadow=False
            self.scene.mono_mask=maskPIL
            self.scene.setShadowImage_toRed()
            self.scene.setShadowImage()
            if self.NoIcon:
                self.pushButton.setText("Switch")
                self.pushButton.setGeometry(QRect(250, 40, 61, 27))
            else:
                self.pushButton.setIcon(QIcon(self.IconSwitch))
                self.pushButton.setIconSize(self.pushButton.rect().size()) 
            self.pushButton_7.setVisible(True)
            
            
            self.pushButton_8.setVisible(True)
            
    def get_mask(self,cvmat_img,setToRef=True):
        #cv2.imwrite('temp_mask0' +'.png', cvmat_img)
        #vis_parsing_anno=cvmat_img
        #vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(cvmat_img)

        #for pi in range(1, num_of_class + 1):
        #    index = np.where(vis_parsing_anno == pi)
        #   vis_parsing_anno_color[index[0], index[1], :] = color_list_p[pi]
            
        #cv2.imwrite('temp_mask1' +'.png', vis_parsing_anno_color)
        image = QImage(cvmat_img, 512, 512, QImage.Format_RGB888)
        
        for pi in range(1, num_of_class + 1):
            index = np.where(cvmat_img == pi)
            #print(index)
            for ind in range(len(index[0])):
                #print(ind)
                image.setPixel(int(index[1][ind]), int(index[0][ind]), color_list[pi].rgb())

                    
        pixmap = QPixmap()        
        pixmap.convertFromImage(image)           

        if setToRef:
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(pixmap)
        return pixmap
    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:    
            mat_img = cv2.imread(fileName)
            self.mask = mat_img.copy()
            self.mask_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 512, 512, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(512):
                for j in range(512):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)

    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 
    """      
    def genDetails1(self):
        #self.mask_m, self.mask = remake_mask(self.scene.this_mouse_strokes,self.scene.closest_face_fname)
        #self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)
        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask = transform_mask(Image.fromarray(np.uint8(mask))) 
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(self.img

        #start_t = time.time()
        generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
        #end_t = time.time()
        #print('inference time : {}'.format(end_t-start_t))
        #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()#.cpu().numpy()#
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)

        if len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(QPixmap.fromImage(qim))
    """
    '''def update_result_scene(self,fileName):
        pimg,bgPIL=evaluate_one(img_path=fileName, cp='79999_iter.pth')
        apd_pil=APdraw(fileName,bgPIL)
        apd_qt = ImageQt.toqpixmap(apd_pil)
        if len(self.result_scene.items())>0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(apd_qt)'''
       
    def edit(self):
        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        params = get_params(self.opt, (512,512))
        transform_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
        transform_image = get_transform(self.opt, params)

        mask = self.mask.copy()
        mask_m = self.mask_m.copy()

        mask = transform_mask(Image.fromarray(np.uint8(mask))) 
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(self.img)
    
        start_t = time.time()
        generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()#.cpu().numpy()#
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        
        qim = QImage(result.data, result.shape[1], result.shape[0], result.strides[0], QImage.Format_RGB888)

        #if len(self.ref_scene.items())>0:
        #    self.ref_scene.removeItem(self.ref_scene.items()[-1])
        #    self.ref_scene.addPixmap(qim)
          
        result_name='temp/ref_result.png'
        img_pil = ImageQt.fromqpixmap(qim)
        img_pil.save(result_name)
        #img_pil.show()
        
        pimg,bgPIL,_,_=evaluate_one(self.mask_net,img_path=result_name)
        #apd_pil=APdraw(result_name,bgPIL)
        datasetAP=GetUpdatedAPdrawDataset(self.AP_opt,result_name,bgPIL)
        apd_pil=CallAPdrawModel(self.AP_model,datasetAP)
        apd_qt = ImageQt.toqpixmap(apd_pil)
        if len(self.result_scene.items())>0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(apd_qt)
        
        img_pixmap = ImageQt.toqpixmap(img_pil)
        if len(self.ref_scene.items())>0:
            self.ref_scene.removeItem(self.ref_scene.items()[-1])
        self.ref_scene.addPixmap(img_pixmap)
        #update_result_scene(result_name)
        #for i in range(512):
        #    for j in range(512):
        #       r, g, b, a = image.pixelColor(i, j).getRgb()
        #       image.setPixel(i, j, color_list[r].rgb()) 
        
        #if len(self.result_scene.items())>0: 
        #    self.result_scene.removeItem(self.result_scene.items()[-1])
        #    self.result_scene.addPixmap(QPixmap.fromImage(qim))

    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if False:
            if type(self.output_img):
                fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                        QDir.currentPath())
                cv2.imwrite(fileName+'.jpg',self.output_img)
        else:
            self.scene.strokes_save()

    def undo(self):
        self.scene.undo()
        self.scene.setShadowImage()
        
    def load_strokes(self):
        self.scene.strokes_load()
    def clear(self):
        if(self.mask!=None):
            self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        #if type(self.image):
        if self.image is not None:
            tst=self.scene.addPixmap(self.image)
            tst.setPos(20, 120)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    #model = Model(config)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    model = create_model(opt)   
    APopt=GetAPOption()
    APmodel=GetAPdrawModel(APopt)
    app = QApplication(sys.argv)
    ex = Ex(model,opt,APmodel,APopt)
    sys.exit(app.exec_())
