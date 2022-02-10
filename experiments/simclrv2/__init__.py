"""
Convert tf models to pytorch, based on the codebase
    https://github.com/Separius/SimCLRv2-Pytorch

Download -> convert -> verify.

Needs Python 3.7 to run tf==1.15.4

python download.py r152_3x_sk1
python convert.py r152_3x_sk1/model.ckpt-250228 --ema
python verify.py r152_3x_sk1.pth
"""
