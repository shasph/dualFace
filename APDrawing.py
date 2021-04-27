import os
import torch
import torchvision.transforms as transforms
from APDrawingGAN.options.test_options import TestOptions
from APDrawingGAN.data import CreateDataLoader
from APDrawingGAN.models import create_model
from PIL import Image
from APDrawingGAN.data.base_dataset import BaseDataset, get_transform
from APDrawingGAN.util.util import tensor2im,save_image
import APDrawingGAN.data.face_landmark
def SaveImg(visuals):
    for label, im_data in visuals.items():
        image_name = '%s_%s.png' % ('temp/', label)
        print(image_name)
        if label=='fake_B':
            im = tensor2im(im_data)
            image_pil = Image.fromarray(im)
            return image_pil
def GetAPOption():
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    opt.dataroot='APDrawingGAN/dataset/data'#/test_single' 
    opt.name ='formal_author' 
    opt.model ='test'
    opt.dataset_mode ='single'
    opt.norm ='batch'
    #opt.use_local=True
    opt.which_epoch ='300'
    return opt
def GetAPdrawModel(opt):
    model = create_model(opt)
    model.setup(opt)
    return model
def GetUpdatedAPdrawDataset(opt,img_path,img_background):
    opt.im_p=img_path
    opt.img_background=img_background
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset
def CallAPdrawModel(model,dataset):
        for i, data in enumerate(dataset):
            if i>0:
                break
            model.set_input(data)
            
            with torch.no_grad():
                model.test()
                visuals = model.get_current_visuals()#in test the loadSize is set to the same as fineSize
            #print(visuals)
            return SaveImg(visuals)


