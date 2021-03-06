# dualFace
dualFace: Two-Stage Drawing Guidance for Freehand Portrait Sketching (CVMJ)

We provide python implementations for our CVM 2021 paper "dualFace:Two-Stage Drawing Guidance for Freehand Portrait Sketching".
This project provide sketch support for artistic portrait drawings with a two-stage framework.
[[arXiv]](https://arxiv.org/abs/2104.12297)[[PDF]](https://arxiv.org/pdf/2104.12297.pdf)[[Project]](http://www.jaist.ac.jp/~xie/dualface.html)[[Video]](https://youtu.be/29nrIwo1t10)

## User Interface
![image](https://user-images.githubusercontent.com/4180028/116048238-f08a1180-a6af-11eb-9504-8b8f9dd99236.png)

## Network Structure (Global Stage)
![dualface-global](https://user-images.githubusercontent.com/4180028/117390165-9e0ae980-af28-11eb-859a-3d8987d6fbe5.JPG)

## Network Structure (Local Stage)
![dualface-local](https://user-images.githubusercontent.com/4180028/117390179-a2cf9d80-af28-11eb-8ef5-94ad2456552a.JPG)

## Prerequisites
- Window
- Conda (Python 3.6)
- CPU or NVIDIA GPU + CUDA CuDNN
## Getting Started
### Installation
- Install PyTorch 1.3.1 and torchvision 0.4.1 from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by
```bash
bat
call conda remove -n py36df
call conda create -n py36df python=3.6 
call conda activate py36df
call conda install pytorch==1.3.1 -c pytorch
pip install cmake
pip install -r requirements.txt
```

### Quick Start (Apply a Pre-trained Model)
- Download a pre-trained model from (https://drive.google.com/open?id=1cQx9hPOJ18sU5HPGkbRTJ-e6cYqqHUHh)
```bash
cd sse
sse.exe "-i index_file -v vocabulary -f filelist -n 8"
call conda activate py36df
python demo.py
```

## Acknowledgments
Our code has depended on the following opensource codes.
- MaskGAN(https://github.com/switchablenorms/CelebAMask-HQ)
- faceParsing(https://github.com/zllrunning/face-parsing.PyTorch) 
- APDrawingGAN(https://github.com/yiranran/APDrawingGAN)
- OpenSSE(https://github.com/zddhub/opensse)

Please contact xie@jaist.ac.jp for any comments or requests.

## Citation
If you use this code for your research, please cite our paper.
```
@article{dualface2021,
  author    = {Zhengyu Huang and
               Yichen Peng and
               Tomohiro Hibino and
               Chunqi Zhao and
               Haoran Xie and
               Tsukasa Fukusato and
               Kazunori Miyata},
  title     = {dualFace: Two-Stage Drawing Guidance for Freehand Portrait Sketching},
  journal   = {Computational Visual Media},
  volume    = {8},
  pages     = {63???77},
  year      = {2022},
  url       = {https://doi.org/10.1007/s41095-021-0227-7}
}
```
