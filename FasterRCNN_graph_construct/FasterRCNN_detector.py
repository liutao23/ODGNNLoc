#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2024/12/10 13:28

import torch
import os
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

class FasterRCNN_detect:
    def __init__(self, model_path=None, num_classes=17, confidence_threshold=0.2):
        """
        特征提取器初始化。

        Args:
            model_path (str): 模型权重文件路径。如果未提供，将抛出错误。
            num_classes (int): 模型的类别数，不包括背景类。
            feature_dim (int): 输出特征维度, 是1024维度。
            device (str): 使用的设备（'cuda' 或 'cpu'）。默认为可用的 GPU 或 CPU。
        """
        super(FasterRCNN_detect, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.confidence_threshold = confidence_threshold
        # 加载基础模型
        self.model = self._initialize_model(num_classes, model_path)

    def _initialize_model(self, num_classes, model_path):
        """
        Returns:
            torch.nn.Module: 加载权重的模型。
        """
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Provided model path '{model_path}' does not exist.")

        model = FasterRCNN(backbone=resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d),
                           num_classes=num_classes+1, rpn_score_thresh=0.5)
        model.load_state_dict(torch.load(model_path, map_location=self.device)["model"])
        model.to(self.device)
        model.eval()
        return model

    def forward(self, x):
        """
        前向传播，获取目标检测结果及特征。
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
        Returns:
            tuple: 包含以下内容的元组：
                - bboxes (torch.Tensor): 检测的边界框，形状为 (N, 4)。
                - confidences (torch.Tensor): 每个边界框的置信度，形状为 (N,)。
                - labels (torch.Tensor): 每个边界框的类别标签，形状为 (N,)。
                - features (torch.Tensor): 每个边界框的特征，形状为 (N, feature_dim)。
        """
        x = x.to(self.device)

        # 获取模型推理结果
        with torch.no_grad():
            results, glocal_feat = self.model(x)

        filtered_results = []

        for result in results:
            # 提取模型输出
            bboxes = result['boxes']  # 检测框
            confidences = result['scores']  # 置信度
            categories = result['labels']  # 类别标签
            features = result['features']  # 特征（模型返回的特征张量）

            # 根据置信度阈值筛选结果
            valid_indices = confidences > self.confidence_threshold
            filtered_bboxes = bboxes[valid_indices]
            filtered_confidences = confidences[valid_indices]
            filtered_categories = categories[valid_indices]
            filtered_features = features[valid_indices]
            filtered_results.append((filtered_bboxes,filtered_features, filtered_categories, filtered_confidences))
        # 返回筛选后的结果和特征
        # return filtered_bboxes, filtered_features, filtered_categories, filtered_confidences
        return filtered_results, glocal_feat



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置模型路径
    drone_img_path = "/media/liutao/A/Datasets/U1652/train/drone/0839/image-02.jpeg"

    satellite_img_path = "/media/liutao/A/Datasets/U1652/train/satellite/0839/image-01.jpg"


    drone_model_path = '/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/share_detect.pth'
    satellite_model_path = '/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/share_detect.pth'

    drone_detector = FasterRCNN_detect(model_path=drone_model_path)
    satellite_detector = FasterRCNN_detect(model_path=satellite_model_path)

    data_transform = transforms.Compose([transforms.ToTensor()])
    drone_imgs = data_transform(Image.open(drone_img_path)).unsqueeze(0)

    satellite_imgs = data_transform(Image.open(satellite_img_path)).unsqueeze(0)

    drone_results, drone_global_feat= drone_detector.forward(drone_imgs)

    satellite_results, satellite_global = satellite_detector.forward(satellite_imgs)

    print(drone_results[0])
    print(satellite_results[0])