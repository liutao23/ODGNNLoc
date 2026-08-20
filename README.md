# IRVL328

## Object-aware Graph Matching Network for Cross-domain UAV Localization

<p align="center">
A benchmark dataset and framework for cross-view and cross-modal UAV localization using object detection and graph matching.
</p>

---

## 📌 Overview

This repository accompanies our work on **cross-domain UAV localization**.

Existing image retrieval methods mainly rely on global scene representations, which may suffer from severe appearance variations caused by viewpoint changes, temporal differences, and sensor modality gaps.

To address this challenge, we explore an object-aware graph matching framework:

- Object detection is used to extract meaningful semantic instances.
- Detected objects are represented as graph nodes.
- Spatial and semantic relations are modeled as graph edges.
- Graph neural networks are used to reason about intra-image structures and inter-image correspondences.

This framework provides an interpretable solution for cross-view and cross-modal localization.

---

## 📰 News

- **2026**: The paper has been accepted by *Chinese Journal of Aeronautics*.
- **2026**: IRVL328 dataset is publicly released.

---

## 📂 Dataset

We release **IRVL328**, an infrared-visible remote sensing localization dataset designed for challenging cross-modal UAV localization scenarios.

The dataset organization follows the conventions of:

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

so that existing cross-view retrieval frameworks can be easily adapted.

---

## 📥 Download

### Dataset

**Baidu Netdisk**

```
Link:
https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj

Password:
52bj
```

**Google Drive**

```
https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing
```

---

### Pretrained Weights

**Baidu Netdisk**

```
Link:
https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA

Password:
52sb
```

The pretrained models contain components related to:

- Object detection
- Graph-based matching modules

---

## 📁 Dataset Structure

```
IRVL328/
├── train/
│   ├── drone/
│   │   ├── 0000/
│   │   │   ├── xx.jpg
│   │   │   └── ...
│   │   ├── 0001/
│   │   │   ├── xx.jpg
│   │   │   └── ...
│   │   └── ...
│   │
│   └── satellite/
│       ├── 0000/
│       │   └── xx.png
│       ├── 0001/
│       │   └── xx.png
│       └── ...
│
├── query_drone/
├── query_satellite/
├── gallery_drone/
└── gallery_satellite/
```

### Notes

- UAV images: `.jpg`
- Satellite images: `.png`

---

# 🔧 Code Availability

Currently available:

- Object detection module
- Graph construction module

The complete implementation, including:

- Training pipeline
- Inference pipeline
- Full graph matching framework

will be released after the official publication process.

For academic non-commercial research purposes, additional implementation details may be provided upon request.

---

## 📮 Contact

Please contact:

```
liutao23@njust.edu.cn
```

For code requests, please include:

- Name
- Affiliation
- Research purpose
- Confirmation of non-commercial academic usage

---

# 📖 Citation

If this dataset or code is useful for your research, please cite:

```bibtex
@article{LIU2026104451,
title = {Object-aware graph matching network for cross-domain remote sensing image localization},
journal = {Chinese Journal of Aeronautics},
pages = {104451},
year = {2026},
issn = {1000-9361},
doi = {https://doi.org/10.1016/j.cja.2026.104451},
url = {https://www.sciencedirect.com/science/article/pii/S1000936126003894},
author = {Tao LIU and Kan REN and Qian CHEN},
keywords = {Remote sensing, Image matching, Object detection, Graph neural networks, Image retrieval, Unmanned aerial vehicles}
}
```

---

# 🙏 Acknowledgements

We thank:

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

for their valuable contributions to cross-view localization research.

---

# 中文说明

## IRVL328：面向跨域无人机定位的目标感知图匹配网络

本仓库提供跨视角、跨模态无人机定位研究所使用的数据集和相关代码。

方法首先利用目标检测提取图像中的显著目标，并将目标构建为图结构：

- 节点表示目标实例；
- 边表示空间和语义关系；
- 图神经网络用于学习图像内部关系以及跨图像匹配关系。

该方法主要关注：

- 跨视角定位
- 跨模态匹配
- 结构化语义理解
- 模型泛化能力研究

---

## 数据集

IRVL328 数据集用于研究复杂跨模态条件下的遥感图像定位问题。

目录结构参考：

- University1652-Baseline
- SUES-200-Benchmark

方便已有跨视角检索代码进行适配。

---

## 开源范围

当前公开：

- 目标检测模块
- 图构建模块

完整训练、推理流程将在论文正式发表后公开。

---

## 引用

如果 IRVL328 数据集或相关代码对您的研究有帮助，请引用上述论文。

---

感谢关注 IRVL328，希望该数据集能够帮助研究者进一步探索跨视角定位中的泛化性和鲁棒性。
