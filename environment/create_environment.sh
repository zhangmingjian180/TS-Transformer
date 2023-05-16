#!/bin/bash

conda install pip==21.2.4  #include python==3.9.12

#pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html # if no GPU.
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
