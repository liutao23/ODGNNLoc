#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2024/12/26 10:25
import timm
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class Global_Feature_Extractor(nn.Module):

    def __init__(self,
                 model_name='convnext_base.fb_in22k_ft_in1k_384',
                 pretrained=False,
                 img_size=383, device=None):

        super(Global_Feature_Extractor, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self, ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None):
        # Ensure images are on the correct device
        img1 = img1.to(self.device)
        if img2 is not None:
            img2 = img2.to(self.device)

        if img2 is not None:

            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            return image_features1, image_features2

        else:
            image_features = self.model(img1)

            return image_features

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = Global_Feature_Extractor()
    global_model.load_state_dict(torch.load("/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/template_matching.pth"),
                                 strict=False)
    global_model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    drone_img_path = r"E:\数据集\TTMN\10张图数据集\SUES200mini\300\train\drone\0000\1.jpg"
    satellite_img_path = r"E:\数据集\TTMN\10张图数据集\SUES200mini\300\train\satellite\0000\1.jpg"

    drone_img = transform(Image.open(drone_img_path)).unsqueeze(0).to(device)  # 为图像添加batch维度并转移到设备
    satellite_img = transform(Image.open(satellite_img_path)).unsqueeze(0).to(device)  # 同样处理卫星图像

    with torch.no_grad():
        drone_img_feature, satellite_img_feature = global_model(drone_img, satellite_img)

    print(drone_img_feature.shape)
    print(satellite_img_feature.shape)


