# dualFace
dualFace: Two-Stage Drawing Guidance for Freehand Portrait Sketching (CVMJ)

We provide python implementations for our CVM 2021 paper "dualFace:Two-Stage Drawing Guidance for Freehand Portrait Sketching".
This project provide sketch support for artistic portrait drawings with a two-stage framework.
[[arXiv]](https://arxiv.org/abs/2104.12297)[[PDF]](https://arxiv.org/pdf/2104.12297.pdf)[[Project]](http://www.jaist.ac.jp/~xie/dualface.html)[[Video]](https://youtu.be/29nrIwo1t10)

In this paper, we propose dualFace, a portrait drawing interface to assist users with different levels of drawing skills to complete recognizable and authentic face sketches. dualFace consists of two-stage drawing assistance to provide global and local visual guidance: global guidance, which helps users draw contour lines of portraits (i.e., geometric structure), and local guidance, which helps users draws details of facial parts (which conform to user-drawn contour lines), inspired by traditional artist workflows in portrait drawing. In the stage of global guidance, the user draws several contour lines, and dualFace then searches several relevant images from an internal database and displays the suggested face contour lines over the background of the canvas. In the stage of local guidance, we synthesize detailed portrait images with a deep generative model from user-drawn contour lines, but use the synthesized results as detailed drawing guidance. We conducted a user study to verify the effectiveness of dualFace, and we confirmed that dualFace significantly helps achieve a detailed portrait sketch.


## User Interface
![image](https://user-images.githubusercontent.com/4180028/116048238-f08a1180-a6af-11eb-9504-8b8f9dd99236.png)

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
@article{huang21dualface,
  title={dualFace: Two-Stage Drawing Guidance for Freehand Portrait Sketching},
  author={Huang, Zhengyu and Peng, Yichen and Hibino, Tomohiro and Zhao, Chunqi Zhao and Xie, Haoran and Fukusato, Tsukasa and Miyata, Kazunori},
  journal={Computational Visual Media},
  year={2021},
  publisher={Springer}
}
```
