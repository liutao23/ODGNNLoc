#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2025/2/11 17:36
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2025/1/9 16:35

from FasterRCNN_detector import FasterRCNN_detect
from Global_Feature_Extractor import Global_Feature_Extractor
from PIL import Image
from torchvision import transforms
import torch
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建节点特征
def build_node_features(bboxes, features, globalfeatures, confidences, img_width, img_height, device=None):
    """
    构建节点特征，包括局部特征、全局特征、归一化边界框信息。
    """
    # 数据有效性检查
    if bboxes.size(1) != 4:
        raise ValueError(f"检测到的边界框形状不对, 是 {bboxes.shape}.")
    if features.size(0) != bboxes.size(0):
        raise ValueError("目标框特征和边界框的个数应该相同.")
    if globalfeatures.size(0) != 1:
        raise ValueError("全局特征向量的维度应该是 (1, D).")

    # 将数据移动到设备并归一化边界框
    bboxes = bboxes.to(device)
    features = features.to(device)
    globalfeatures = globalfeatures.to(device)
    confidences = confidences.to(device)

    # 计算目标框的面积
    box_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # 计算图像的面积
    image_area = img_width * img_height

    # 计算目标框面积与图像面积的比例
    area_ratio = box_area / image_area
    weighted_ratio = area_ratio * confidences

    # 融合局部和全局特征
    # 较大的目标更多依赖局部特征，较小的目标更多依赖全局特征，这在实践中符合感知的逻辑
    globalfeatures_expanded = globalfeatures.expand(features.size(0), -1)

    # 根据面积比例加权融合全局和局部特征
    weighted_ratio_expanded = weighted_ratio.unsqueeze(-1)
    fused_features = weighted_ratio_expanded * features + (1 - weighted_ratio_expanded) * globalfeatures_expanded

    # 拼接特征、置信度和归一化边界框
    bboxes_normalized = bboxes / torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float, device=device)

    # 返回融合后的特征
    return torch.cat([fused_features, bboxes_normalized], dim=-1)

def compute_iou(bboxes1, bboxes2):
    """
    计算两个边界框的交并比 (IoU)，使用torch操作。
    :param bboxes1: 目标框 1 (N, 4)
    :param bboxes2: 目标框 2 (M, 4)
    :return: IoU 矩阵 (N, M)
    """
    x1, y1, x2, y2 = bboxes1.T  # 转置，以便提取坐标
    x1_t, y1_t, x2_t, y2_t = bboxes2.T

    xi1 = torch.max(x1.unsqueeze(1), x1_t.unsqueeze(0))  # N, M
    yi1 = torch.max(y1.unsqueeze(1), y1_t.unsqueeze(0))  # N, M
    xi2 = torch.min(x2.unsqueeze(1), x2_t.unsqueeze(0))  # N, M
    yi2 = torch.min(y2.unsqueeze(1), y2_t.unsqueeze(0))  # N, M

    inter_area = torch.max(xi2 - xi1, torch.tensor(0.0)) * torch.max(yi2 - yi1, torch.tensor(0.0))  # N, M
    area1 = (x2 - x1) * (y2 - y1)  # N
    area2 = (x2_t - x1_t) * (y2_t - y1_t)  # M
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area  # N, M

    return inter_area / union_area  # N, M

def compute_ciou(bboxes1, bboxes2):
    """
    计算两个边界框的CIoU (Complete Intersection over Union)，使用torch操作。
    :param bboxes1: 目标框 1 (N, 4)
    :param bboxes2: 目标框 2 (M, 4)
    :return: CIoU 矩阵 (N, M)
    """
    x1, y1, x2, y2 = bboxes1.T  # 转置，以便提取坐标
    x1_t, y1_t, x2_t, y2_t = bboxes2.T

    # 计算交集的左上角和右下角
    xi1 = torch.max(x1.unsqueeze(1), x1_t.unsqueeze(0))  # N, M
    yi1 = torch.max(y1.unsqueeze(1), y1_t.unsqueeze(0))  # N, M
    xi2 = torch.min(x2.unsqueeze(1), x2_t.unsqueeze(0))  # N, M
    yi2 = torch.min(y2.unsqueeze(1), y2_t.unsqueeze(0))  # N, M

    # 计算交集面积
    inter_area = torch.max(xi2 - xi1, torch.zeros_like(xi1)) * torch.max(yi2 - yi1, torch.zeros_like(yi1))

    # 计算每个框的面积
    area1 = (x2 - x1) * (y2 - y1)  # N
    area2 = (x2_t - x1_t) * (y2_t - y1_t)  # M

    # 计算并集面积
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    # 计算中心点的距离
    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2
    center_x2 = (x1_t + x2_t) / 2
    center_y2 = (y1_t + y2_t) / 2
    center_distance = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2  # N, M

    # 计算最小覆盖框的对角线长度
    c_x1 = torch.min(x1, x1_t)
    c_y1 = torch.min(y1, y1_t)
    c_x2 = torch.max(x2, x2_t)
    c_y2 = torch.max(y2, y2_t)
    c_diagonal = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2  # N, M

    # 计算宽高比的一致性
    aspect_ratio1 = (x2 - x1) / (y2 - y1)  # N
    aspect_ratio2 = (x2_t - x1_t) / (y2_t - y1_t)  # M
    aspect_ratio_diff = (aspect_ratio1.unsqueeze(1) - aspect_ratio2.unsqueeze(0)) ** 2

    # CIoU 计算
    iou = inter_area / union_area
    ciou = iou - (center_distance / c_diagonal) - 0.5 * aspect_ratio_diff

    # 返回 CIoU 值
    return ciou

def build_semantic_edges(classes):
    """
    根据目标的类别相似性构建边。
    :param classes: 每个目标框的类别标签 (N,)
    :return: 邻接矩阵，表示图中基于类别的边
    """
    n = len(classes)
    adjacency_matrix = torch.zeros((n, n), dtype=torch.float)

    # 如果两个目标属于同一类别，则构成边
    adjacency_matrix[classes.unsqueeze(1) == classes.unsqueeze(0)] = 1
    return adjacency_matrix

def build_feature_edges(features):
    """
    根据目标框的特征余弦相似度直接生成边的权重。
    :param features: 目标框的特征向量 (N, D)
    :return: 邻接矩阵，表示目标框之间的余弦相似度权重
    """
    cosine_sim = torch.mm(features, features.T)  # 计算余弦相似度矩阵 (N, N)

    # 标准化相似度矩阵
    norm = torch.norm(features, dim=1, keepdim=True)
    cosine_sim = cosine_sim / (norm * norm.T)  # 归一化为余弦相似度

    return cosine_sim  # 直接返回余弦相似度矩阵作为边的权重

def build_combined_edges(bboxes=None, features=None, classes=None,
    iou_weight=0.6, category_weight=0.2, feature_weight=0.3,
    use_iou=True, use_category=True, use_feature=True):
    """
    综合空间、语义和特征相似性来构建边，直接使用IoU和余弦相似度的实际值作为边权重。
    :param bboxes: 目标框 (N, 4)，如果 use_iou 为 False 或无效可为 None
    :param features: 目标框的特征向量 (N, D)，如果 use_feature 为 False 或无效可为 None
    :param classes: 目标框的类别标签 (N,)，如果 use_category 为 False 或无效可为 None
    :param iou_weight: IoU权重系数
    :param category_weight: 类别权重系数
    :param feature_weight: 特征相似度权重系数
    :param use_iou: 是否使用IoU边
    :param use_category: 是否使用类别边
    :param use_feature: 是否使用特征边
    :return: 邻接矩阵，表示综合关系的图
    """
    possible_sources = [bboxes, features, classes]
    n = next((source.size(0) if source is not None else None for source in possible_sources), None)
    if n is None:
        raise ValueError("必须提供bboxes、features或classes中的至少一个.")

    # 初始化邻接矩阵
    combined_edges = torch.zeros((n, n), dtype=torch.float, device=device)

    # 添加IoU边
    if use_iou and bboxes is not None:
        bboxes = bboxes.to(device)  # 将 bboxes 移动到指定设备
        # iou_edges = compute_iou(bboxes, bboxes).to(device)
        iou_edges = compute_ciou(bboxes, bboxes).to(device)
        combined_edges += iou_weight * iou_edges

    # 添加类别边
    if use_category and classes is not None:
        classes = classes.to(device)  # 将 classes 移动到指定设备
        category_edges = build_semantic_edges(classes).to(device)
        combined_edges += category_weight * category_edges

    # 添加特征边
    if use_feature and features is not None:
        features = features.to(device)  # 将 features 移动到指定设备
        feature_edges = build_feature_edges(features).to(device)
        combined_edges += feature_weight * feature_edges

    # 去除自环边
    combined_edges.fill_diagonal_(0)

    return combined_edges

# 构建一个图
def build_graph_data_dict(node_features, labels, edge_index, edge_attr, combined_edges, bboxes):
    """
    构建一个图数据字典，包括节点特征、标签、边信息等。

    :param node_features: 节点特征，形状为 (N, D)
    :param labels: 节点标签，形状为 (N,)
    :param edge_index: 边索引，形状为 (2, E)
    :param edge_attr: 边的权重，形状为 (E,)
    :param combined_edges: 边的权重矩阵，形状为 (N, N)
    :param bboxes: 目标框，形状为 (N, 4)，每个目标框为 (xmin, ymin, xmax, ymax)

    :return: graph_data_dict: 包含图数据的字典
    """
    graph_data_dict = {
        'node_features': node_features,  # 节点特征
        'labels': labels,  # 节点标签（目标类别）
        'edge_index': edge_index,  # 边索引
        'edge_attr': edge_attr,  # 边的权重
        'combined_edges': combined_edges,  # 边的权重矩阵
        'bboxes': bboxes  # 目标框
    }

    return graph_data_dict

def save_graph(graph_dict, file_path):
    """
    保存图的节点特征、边索引和边特征到文件。
    :param data: 图数据 (Data 对象)
    :param file_path: 保存的文件路径
    """
    with open(file_path, 'wb') as f:
        pickle.dump(graph_dict, f)
    print(f"Graph features saved to {file_path}")

def load_graph_features(file_path):
    """
    从文件加载图的节点特征、边索引和边特征。
    :param file_path: 保存的文件路径
    :return: Graph Data 对象
    """
    with open(file_path, 'rb') as f:
        graph_dict = pickle.load(f)
    print(f"Graph features loaded from {file_path}")
    return graph_dict

if __name__ == "__main__":

    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 载入目标检测模型
    drone_model_path = '/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/share_detect.pth'
    drone_detector = FasterRCNN_detect(model_path=drone_model_path)

    satellite_model_path = '/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/share_detect.pth'
    satellite_detector = FasterRCNN_detect(model_path=satellite_model_path)

    # 载入全局特征提取模型
    global_drone_model_path = "/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/template_matching.pth"
    global_drone_model = Global_Feature_Extractor()
    global_drone_model.load_state_dict(torch.load(global_drone_model_path), strict=False)
    global_drone_model.to(device).eval()

    global_satellite_model_path = "/home/liutao/视频/FasterRCNN_graph_construct/pre_weights/template_matching.pth"
    global_satellite_model = Global_Feature_Extractor()
    global_satellite_model.load_state_dict(torch.load(global_satellite_model_path), strict=False)
    global_satellite_model.to(device).eval()

    # 预处理
    data_transform = transforms.Compose([transforms.ToTensor()])
    drone_miss_path = []
    satellite_miss_path = []


    drone_img_path = r"/home/liutao/视频/FasterRCNN_graph_construct/image-01.jpeg"
    graph_path = drone_img_path[:-4] + 'pkl'
    img = data_transform(Image.open(drone_img_path)).unsqueeze(0)
    _, _, img_height, img_width = img.shape

    bboxes, features, labels, confidences = drone_detector.forward(img)[0][0]
    drone_num = bboxes.size(0)
    global_features = global_drone_model(img)

    # 构建节点特征 (N, 1024+4)
    node_features = build_node_features(bboxes, features, global_features, confidences,
                                        img_width, img_height, device=device)

    # 计算边权重矩阵 (N, N)
    combined_edges = build_combined_edges(bboxes=bboxes, features=features, classes=labels,use_iou=True, use_category=True, use_feature=True)

    # 获取所有非零元素的索引
    edge_index = torch.nonzero(combined_edges, as_tuple=False).T

    # 获取每条边的权重
    edge_attr = combined_edges[edge_index[0], edge_index[1]]

    graph_data = build_graph_data_dict(node_features, labels, edge_index, edge_attr, combined_edges, bboxes)

    # 保存数据
    save_graph(graph_data, graph_path)

