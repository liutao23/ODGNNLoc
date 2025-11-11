#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2025/1/9 11:21

import matplotlib

matplotlib.use('TkAgg')  # 设置支持图形界面的后端
import numpy as np
from sklearn.decomposition import PCA
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data


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

# 标签字典
label_dict = {
    1: "low_rise residential",
    2: "mid_rise residential",
    3: "high_rise residential",
    4: "saving box",
    5: "baseball field",
    6: "basketball field",
    7: "playground",
    8: "bridge",
    9: "Irregular buildings",
    10: "intersection",
    11: "parking lot",
    12: "chimney",
    13: "tennis court",
    14: "football field",
    15: "rugby field",
    16: "lighthouse",
    17: "Underpass intersections",
    18: "foot bridge",
    19: "port",
    20: "spherical construction",
    21: "park",
    22: "badminton court",
    23: "volleyball court",
    24: "electric tower",
    25: "recreational facilities",
    26: "ship",
    27: "pool"
}

import logging
logging.basicConfig(level=logging.INFO)

def visualize_graph(data, graph_name, label_dict=None, use_node_features=False):
    """
    可视化图形。
    :param data: 包含图数据的字典
    :param graph_name: 图名称，用于图像保存
    :param label_dict: 类别标签映射字典
    :param use_node_features: 是否使用节点特征来调整可视化效果
    """
    # 输入检查
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary.")
    if 'combined_edges' not in data or 'node_features' not in data or 'labels' not in data:
        raise KeyError("Input data must contain 'combined_edges', 'node_features', and 'labels'.")

    # 提取图数据
    combined_edges = data['combined_edges']
    edge_index = (combined_edges > 0).nonzero(as_tuple=False).T
    node_features = data['node_features']
    edge_attr = combined_edges[edge_index[0], edge_index[1]]
    node_labels = data['labels']

    # 检查边索引是否有效
    if edge_index.shape[1] == 0:
        logging.warning("No edges found in combined_edges.")

    # 创建图
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(source, target, weight=edge_attr[i].item() if edge_attr is not None else 1)

    # 获取节点标签
    node_labels_dict = {}
    for i in range(len(node_labels)):
        label_id = node_labels[i].item()
        if i in G.nodes:  # 只包含图 G 中存在的节点
            node_labels_dict[i] = label_dict.get(label_id, f"Unknown({label_id})")

    # 计算节点位置
    pos = nx.spring_layout(G, seed=42)

    # 绘制图
    plt.figure(figsize=(8, 8))
    if use_node_features:
        pca = PCA(n_components=1)
        reduced_features = pca.fit_transform(node_features.detach().cpu().numpy())
        node_size = np.maximum((reduced_features[:, 0] + 1) * 100, 1)
        node_color = reduced_features[:, 0]  # 使用降维后的特征作为颜色
    else:
        node_color = 'skyblue'  # 默认颜色
        node_size = 500

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.viridis, alpha=0.6)
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color='gray')
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, labels=node_labels_dict, font_size=12, font_color='black')

    plt.title(f"{graph_name} Visualization")
    plt.axis('off')
    plt.savefig(f"{graph_name}.png")
    plt.close()

drone_graph = load_graph_features("/home/liutao/视频/FasterRCNN_graph_construct/image-01.pkl")

# 绘制无人机图和卫星图
visualize_graph(drone_graph, 'drone_graph', label_dict=label_dict)

