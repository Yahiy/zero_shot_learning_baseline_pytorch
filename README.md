# zero_shot_learning_baseline_pytorch



In this repository, you can find the code of some baseline mothod of ZSL. Such as sigmod cross-entropy loss, billner, encode-decode. Papers will be push next time.

#### 1. Dependencies

| Package       | Version  |
| :------------ | -------- |
| Python        | 2.7      |
| CUDA          | 9.0      |
| torch         | 0.3.1    |
| torchvision   | 0.2.0    |
| Tensor flow   | 1.6      |
| tensorboardX  | 1.1      |
| scipy         | 1.0.0    |
| opencv-python | 3.4.3.18 |
| pillow        | :5.0.0   |

**conda env **:

```
conda env list
conda create -n YOUR_NAME python=2.7
conda install -n YOUR_NAME python=2.7
source activate YOUR_NAME
```

```
pip install opencv-python
pip install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
pip install tensorboardX
pip install scipy
pip install opencv-python
pip install pillow
pip install sklearn
```

#### 2. Train and test 

```
cd code
python main.py
```

| File             | Algorithm     |
| ---------------- | ------------- |
| train_64.py      | Sig           |
| train_biliner.py | Deep-Rule     |
| train_mult.py    | Must task     |
| train_decode.py  | Encode-Decode |

