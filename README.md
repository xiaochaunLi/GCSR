# GCSR
AIM2020 efficent SR challenge

The code is based on the PyTorch implementation of the EDSR(https://github.com/thstkdgus35/EDSR-PyTorch)
Prerequisites
Linux or macOS
Python 3.6
NVIDIA GPU + CUDA10.1

Getting Started
Installation
pip install PyTorch>=1.1.0 and dependencies from http://pytorch.org
pip install Torchvision>=0.3.0 from the source.
pip install numpy
pip install skimage
pip install imageio
pip install matplotlib
pip install tqdm

Quickstart (Demo)

if you want to test out model,put the image in ./test file and run
cd src
python main.py --data_test Demo --pre_train ../experiment/AIM2020/model_best.pt --test_only --save_results --save AIM2020
and you can see the results in experiment/AIM2020/results_Demo 

if you want to test our model time and parameters,open the MSRResnet,please put the image in the DIV2K file and run :
cd MSRResnet
python test_demo.py
