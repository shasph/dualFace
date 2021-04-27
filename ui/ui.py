from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        #Form.resize(1800, 660)
        Form.resize(560, 660)
        bt_height=0#27
        bt_width=0#97
        
        init_x=30
        init_y=10
        
        pushx=50#81
        pushy=50#27

        gapx=10
        gapy=20

        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sld.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.sld.setGeometry(init_x+100, init_y+70, 100, 20)
        self.sld.setTickInterval(1)
        self.sld.setMinimum(1)
        self.sld.setMaximum(6)
        self.sld.setValue(6)
        self.sld.setTickPosition(QtWidgets.QSlider.TicksAbove)
        
        self.sld_shadowlimit = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.sld_shadowlimit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sld_shadowlimit.setGeometry(init_x+390, init_y+70, 100, 20)
        self.sld_shadowlimit.setTickInterval(1)
        self.sld_shadowlimit.setMinimum(1)
        self.sld_shadowlimit.setMaximum(6)
        self.sld_shadowlimit.setValue(3)
        self.sld_shadowlimit.setTickPosition(QtWidgets.QSlider.TicksAbove)
        
        self.sld_label = QtWidgets.QLabel(self) 
        self.sld_label.setGeometry(init_x, init_y+70, 100, 20)
        self.sld_label.setAlignment(QtCore.Qt.AlignCenter)
                
        self.sld_labe2 = QtWidgets.QLabel(self) 
        self.sld_labe2.setGeometry(init_x+280, init_y+70, 100, 20)
        self.sld_labe2.setAlignment(QtCore.Qt.AlignCenter)
        
        self.rButton1 = QtWidgets.QRadioButton(Form) 
        self.rButton1.setObjectName("rButton1")
        self.rButton1.setGeometry(QtCore.QRect(400, 10, 100, 27))
        self.rButton1.setChecked(True)
        self.rButton1.clicked.connect(Form.redio1_clicked)#toggled
        self.rButton2 = QtWidgets.QRadioButton(Form) 
        self.rButton2.setObjectName("rButton2")
        self.rButton2.setGeometry(QtCore.QRect(400, 40, 100, 27))
        self.rButton2.clicked.connect(Form.redio2_clicked)#redio_toggled
        
        self.pushButton = QtWidgets.QPushButton(Form)
        #self.pushButton.setGeometry(QtCore.QRect(1160, 360, 81, 27))
        self.pushButton.setGeometry(QtCore.QRect(30+pushx*3+3+gapx*3, 10, pushx, pushy))#Form.genDetails =>self.pushButton.setGeometry(QtCore.QRect(250, 40, 81, 27))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 10, bt_width, bt_height))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 40, bt_width, bt_height))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 10, pushx, pushy))#Form.load_strokes
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(30+pushx*2+3+gapx*2, 10, pushx, pushy))#Form.undo
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(30+pushx+3+gapx, 10, pushx, pushy))#Form.save_img #250 10
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(30+pushx*4+3+gapx*4, 10, pushx, pushy))#Form.changePen 335
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(30+pushx*5+3+gapx*5, 10, 20, pushy))#self.pushButton_8.setGeometry(QtCore.QRect(311, 40, 20, pushy))#Form.switch_shadow
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(Form)
        self.pushButton_9.setGeometry(QtCore.QRect(450, 40, bt_width, bt_height))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(Form)
        self.pushButton_10.setGeometry(QtCore.QRect(570, 10, bt_width, bt_height))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(Form)
        self.pushButton_11.setGeometry(QtCore.QRect(570, 40, bt_width, bt_height))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(Form)
        self.pushButton_12.setGeometry(QtCore.QRect(690, 10, bt_width, bt_height))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(Form)
        self.pushButton_13.setGeometry(QtCore.QRect(690, 40, bt_width, bt_height))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(Form)
        self.pushButton_14.setGeometry(QtCore.QRect(810, 10, bt_width, bt_height))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(Form)
        self.pushButton_15.setGeometry(QtCore.QRect(810, 40, bt_width, bt_height))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_16 = QtWidgets.QPushButton(Form)
        self.pushButton_16.setGeometry(QtCore.QRect(930, 10, bt_width, bt_height))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_17 = QtWidgets.QPushButton(Form)
        self.pushButton_17.setGeometry(QtCore.QRect(930, 40, bt_width, bt_height))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(Form)
        self.pushButton_18.setGeometry(QtCore.QRect(1050, 10, bt_width, bt_height))
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_19 = QtWidgets.QPushButton(Form)
        self.pushButton_19.setGeometry(QtCore.QRect(1050, 40, bt_width, bt_height))
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_20 = QtWidgets.QPushButton(Form)
        self.pushButton_20.setGeometry(QtCore.QRect(1170, 10, bt_width, bt_height))
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_21 = QtWidgets.QPushButton(Form)
        self.pushButton_21.setGeometry(QtCore.QRect(1170, 40, bt_width, bt_height))
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_22 = QtWidgets.QPushButton(Form)
        self.pushButton_22.setGeometry(QtCore.QRect(1290, 10, bt_width, bt_height))
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_23 = QtWidgets.QPushButton(Form)
        self.pushButton_23.setGeometry(QtCore.QRect(1290, 40, bt_width, bt_height))
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_24 = QtWidgets.QPushButton(Form)
        self.pushButton_24.setGeometry(QtCore.QRect(1410, 10, bt_width, bt_height))
        self.pushButton_24.setObjectName("pushButton_24")
        self.pushButton_25 = QtWidgets.QPushButton(Form)
        self.pushButton_25.setGeometry(QtCore.QRect(1410, 40, bt_width, bt_height))
        self.pushButton_25.setObjectName("pushButton_25")
        self.pushButton_26 = QtWidgets.QPushButton(Form)
        self.pushButton_26.setGeometry(QtCore.QRect(1530, 10, bt_width, bt_height))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_bt_height = QtWidgets.QPushButton(Form)
        self.pushButton_bt_height.setGeometry(QtCore.QRect(1530, 40, bt_width, bt_height))
        self.pushButton_bt_height.setObjectName("pushButton_bt_height")

        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(20, 120, 512, 512))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(620, 120, 512, 512))
        self.graphicsView_2.setObjectName("graphicsView_2") 
        self.graphicsView_3 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_3.setGeometry(QtCore.QRect(1260, 120, 512, 512))
        self.graphicsView_3.setObjectName("graphicsView_3")
        


        self.retranslateUi(Form)
        self.sld.valueChanged[int].connect(Form.changeSilderValue)
        self.sld_shadowlimit.valueChanged[int].connect(Form.changeShadowLimitSilderValue)

        self.pushButton.clicked.connect(Form.genDetails)
        self.pushButton_2.clicked.connect(Form.open)
        self.pushButton_3.clicked.connect(Form.open_mask)
        self.pushButton_4.clicked.connect(Form.load_strokes)#clear
        self.pushButton_5.clicked.connect(Form.undo)
        self.pushButton_6.clicked.connect(Form.save_img)
        self.pushButton_7.clicked.connect(Form.changePen)#Form.bg_mode
        self.pushButton_8.clicked.connect(Form.switch_shadow)#Form.skin_mode
        self.pushButton_9.clicked.connect(Form.nose_mode)
        self.pushButton_10.clicked.connect(Form.eye_g_mode)
        self.pushButton_11.clicked.connect(Form.l_eye_mode)
        self.pushButton_12.clicked.connect(Form.r_eye_mode)
        self.pushButton_13.clicked.connect(Form.l_brow_mode)
        self.pushButton_14.clicked.connect(Form.r_brow_mode)
        self.pushButton_15.clicked.connect(Form.l_ear_mode)
        self.pushButton_16.clicked.connect(Form.r_ear_mode)
        self.pushButton_17.clicked.connect(Form.mouth_mode)
        self.pushButton_18.clicked.connect(Form.u_lip_mode)
        self.pushButton_19.clicked.connect(Form.l_lip_mode)
        self.pushButton_20.clicked.connect(Form.hair_mode)
        self.pushButton_21.clicked.connect(Form.hat_mode)
        self.pushButton_22.clicked.connect(Form.ear_r_mode)
        self.pushButton_23.clicked.connect(Form.neck_l_mode)
        self.pushButton_24.clicked.connect(Form.neck_mode)
        self.pushButton_25.clicked.connect(Form.cloth_mode)
        self.pushButton_26.clicked.connect(Form.increase)
        self.pushButton_bt_height.clicked.connect(Form.decrease)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        
        Form.setWindowTitle(_translate("Form", "dualFace"))
        self.sld_label.setText(_translate("Form", "Stroke Width: "+str(self.sld.value())))
        self.sld_labe2.setText(_translate("Form", "Shadow Number: "+str(self.sld_shadowlimit.value())))
        self.pushButton.setText(_translate("Form", "Gen Details"))
        self.pushButton_2.setText(_translate("Form", "Open Image"))
        self.pushButton_3.setText(_translate("Form", "Open Mask"))
        self.pushButton_4.setText(_translate("Form", "Load"))#Clear Load Strokes
        self.pushButton_5.setText(_translate("Form", "Undo"))
        self.pushButton_6.setText(_translate("Form", "Save"))#Save Image
        #self.pushButton_7.setText(_translate("Form", "BackGround"))
        self.pushButton_8.setText(_translate("Form", ">>"))#_translate("Form", "Skin")
        self.pushButton_9.setText(_translate("Form", "Nose"))
        self.pushButton_10.setText(_translate("Form", "Eyeglass"))
        self.pushButton_11.setText(_translate("Form", "Left Eye"))
        self.pushButton_12.setText(_translate("Form", "Right Eye"))
        self.pushButton_13.setText(_translate("Form", "Left Eyebrow"))
        self.pushButton_14.setText(_translate("Form", "Right Eyebrow"))
        self.pushButton_15.setText(_translate("Form", "Left ear"))
        self.pushButton_16.setText(_translate("Form", "Right ear"))
        self.pushButton_17.setText(_translate("Form", "Mouth"))
        self.pushButton_18.setText(_translate("Form", "Upper Lip"))
        self.pushButton_19.setText(_translate("Form", "Lower Lip"))
        self.pushButton_20.setText(_translate("Form", "Hair"))
        self.pushButton_21.setText(_translate("Form", "Hat"))
        self.pushButton_22.setText(_translate("Form", "Earring"))
        self.pushButton_23.setText(_translate("Form", "Necklace"))
        self.pushButton_24.setText(_translate("Form", "Neck"))
        self.pushButton_25.setText(_translate("Form", "Cloth"))
        self.pushButton_26.setText(_translate("Form", "+"))
        self.pushButton_bt_height.setText(_translate("Form", "-"))
        self.rButton1.setText(_translate("Form", "Global Stage"))
        self.rButton2.setText(_translate("Form", "Local Stage"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

