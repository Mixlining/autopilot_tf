# Description

本项目是fork自[此项目](https://github.com/SullyChen/Autopilot-TensorFlow),同时也参照了[此项目](https://github.com/nishantml/Self-Driving-Car)

`freeze.yaml`是使用 `conda env export > freeze.yml` 导出的依赖表,可以使用`conda env create -f freeze.yml`创建对应的运行环境

各个`.md`文档是对相应`.py`文件的注释



# Autopilot-TensorFlow

A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes. For a summary of the design process and FAQs, see [this medium article I wrote](https://medium.com/@sullyfchen/how-a-high-school-junior-made-a-self-driving-car-705fa9b6e860).

# IMPORTANT

Absolutely, under NO circumstance, should one ever pilot a car using computer vision software trained with this code (or any home made software for that matter). It is extremely dangerous to use your own self-driving software in a car, even if you think you know what you're doing, not to mention it is quite illegal in most places and any accidents will land you in huge lawsuits.

This code is purely for research and statistics, absolutley NOT for application or testing of any sort.

# How to Use
Download the [dataset](https://github.com/SullyChen/driving-datasets) and extract into the repository folder

Use `python train.py` to train the model

Use `python run.py` to run the model on a live webcam feed. 需要打开摄像头,并给予画面,模型会预测出转向角度

Use `python run_dataset.py` to run the model on the dataset 使用的还是训练数据集的图像,并输出角度 

To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.
