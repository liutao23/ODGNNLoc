# IRVL328

## Object-aware Graph Matching Network for Cross-domain UAV Localization

<p align="center">
A benchmark dataset and framework for cross-view and cross-modal UAV localization using object detection and graph matching.
</p>

---

# 📝 A Note from the Author

## English

This work was completed in November 2023. During the following years, the manuscript went through several rounds of submission and review, including submissions to journals such as **TIP**, **TGRS**, **KBS**, and **JSTARS**. After nearly three years, the work was finally accepted.

I have always believed that the core idea of this work is meaningful: leveraging modern object detection techniques to extract structured semantic information and integrating them with graph matching methods provides a simple, interpretable, and potentially generalizable framework for cross-view localization. With the continuous development of object detection algorithms, such a framework may naturally benefit from stronger detectors in the future.

However, when this work was conducted, many components were implemented from scratch. Despite extensive experiments, the performance was not always competitive with some CNN-based or Transformer-based approaches. During the review process, many concerns focused mainly on quantitative improvements rather than the underlying research idea. This experience also made me reconsider whether this research direction should be further pursued.

Fortunately, this work was eventually recognized and accepted. As my first research paper during my Ph.D., it carries special meaning to me. Although the original idea may no longer be considered novel today, I sincerely hope that the released dataset, especially the **IRVL328** dataset, can provide a useful benchmark for studying model generalization, robustness, and cross-view as well as cross-modal localization under challenging conditions.

Recently, I also explored whether modern large language models could assist in improving the model through hyperparameter optimization. However, after further analysis, I realized that the limitations are not simply caused by parameter settings.

Datasets such as **University1652**, **SUES-200**, and **IRVL328** contain relatively limited numbers of objects per scene. In many satellite tiles, the available semantic objects are sparse, which naturally restricts the effectiveness of graph matching methods. Such approaches may demonstrate greater potential on large-scale urban localization datasets with richer object distributions and more complex structural information.

For these reasons, I have decided not to release the complete training pipeline at this stage. The implementation itself is based on relatively standard components, including object detection and graph reasoning modules, and many parts can now be reproduced with the assistance of modern AI development tools. More importantly, I hope researchers can select methodologies according to their actual objectives.

If the primary goal is achieving the highest retrieval accuracy, object detection combined with graph matching may not always be the optimal solution. However, if the focus is on structured semantic understanding, interpretability, and cross-domain generalization, this direction may still provide valuable insights.

I sincerely hope that this dataset can support future research on cross-view localization, robustness evaluation, and generalization analysis.

Best wishes for your research.

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

---

## 中文

本文工作完成于 **2023 年 11 月**。之后几年中，论文经历了多轮投稿与审稿，包括 **TIP、TGRS、KBS、JSTARS** 等期刊，最终历时近三年被正式接收。

我一直认为，这项工作的核心思想具有一定价值：利用现代目标检测技术提取图像中的结构化语义信息，并结合图匹配方法，可以构建一个简单、直观且具有解释性的跨视角定位框架。随着目标检测算法不断发展，该框架理论上也能够从更强大的检测器中进一步受益。

然而，在开展这项工作时，许多模块都是从零实现的。尽管进行了大量实验，模型性能仍然无法始终与部分基于 CNN 或 Transformer 的方法相比。在论文审稿过程中，许多意见主要集中于性能指标提升，而对于方法本身的研究价值关注相对较少。这段经历也让我重新思考是否应该继续推进这一研究方向。

幸运的是，这项工作最终得到了认可并被接收。作为我博士阶段的第一篇研究论文，它对我具有特殊意义。虽然如今这一思路可能已经不再被认为具有高度新颖性，但我仍然希望公开的数据集，尤其是 **IRVL328** 数据集，能够为研究者提供一个具有挑战性的跨视角、跨模态定位测试平台，用于研究模型泛化能力、鲁棒性以及复杂环境下的定位问题。

最近，我也尝试探索是否可以利用现代大语言模型辅助模型优化，例如进行超参数搜索。然而经过进一步分析后，我认为该方法的限制并不仅仅来自参数设置。

例如 **University1652、SUES-200 以及 IRVL328** 等数据集中，单幅场景中的目标数量相对有限，卫星图像瓦片中的有效语义节点通常较为稀疏。因此，在这种场景下，图匹配方法本身可能会受到一定限制。未来，在具有更加丰富目标分布和复杂结构信息的大规模城市级定位数据集中，这类方法或许能够展现更大的潜力。

基于上述原因，目前我决定暂时不公开完整训练流程。一方面，该方法主要由目标检测、图构建以及图推理等相对标准的模块组成，随着现代 AI 辅助开发工具的发展，其中许多部分已经较容易复现；另一方面，我也希望研究者能够根据实际研究目标选择合适的方法。

如果目标是追求最高检索精度，那么目标检测结合图匹配的方法并不一定始终是最优选择。但如果关注结构化语义理解、模型可解释性以及跨域泛化能力，这一方向仍然具有一定研究意义。

衷心希望该数据集能够帮助未来研究者进一步探索跨视角定位中的泛化性、鲁棒性以及跨模态匹配问题。

祝科研顺利。
---

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
