# 3D RU-Net

Code for the paper entitled "3D RoI-aware U-Net for Accurate and Efficient Colorectal Cancer Segmentation"(https://arxiv.org/abs/1806.10342).

The latest codes along with weights and a test fold are now released.

Tips: a recent attempt that transfers training and inferencing to fp16 data format can further enlarge applicable volume sizes.

![Fig.0.](https://github.com/huangyjhust/3D-RU-Net/blob/master/Images/R-UNet.png)

Here are some results of colorectal cancer segmentation, which is the case of the paper; and illustrations of another task, mandible and masseter segmentation, showing the scalability of the proposed method.

![Fig.1.](https://github.com/huangyjhust/3D-RU-Net/blob/master/Images/Results1.png)
![Fig.2.](https://github.com/huangyjhust/3D-RU-Net/blob/master/Images/Results2.png)

Latest experiment: simultaneously segmenting 14 organs from pelvic CTs in ~0.5s (We trained this model with 24 training samples).

![Fig.2.](https://github.com/huangyjhust/3D-RU-Net/blob/master/Images/Results3.png)
