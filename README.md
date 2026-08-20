# IRVL328

## Object-aware Graph Matching Network for Cross-domain Remote Sensing Image Localization

[English](#english) | [中文说明](#中文说明)

[![Paper](https://img.shields.io/badge/Paper-CJA%202026-blue)](https://doi.org/10.1016/j.cja.2026.104451)
[![Dataset](https://img.shields.io/badge/Dataset-IRVL328-orange)](#dataset)
[![Code](https://img.shields.io/badge/Code-Partially%20Released-lightgrey)](#code-availability)

---

## 📝 A Note from the Author

### English

This work was completed in **November 2023**. During the following years, the manuscript went through several rounds of submission and review, including submissions to journals such as **TIP**, **TGRS**, **KBS**, and **JSTARS**. After nearly three years, the work was finally accepted.

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

This repository accompanies our work on **cross-domain remote sensing image localization**.

We first use **object detection** to extract salient instances from UAV / satellite images, and then construct graphs where objects act as nodes and spatial/semantic relations as edges. A **graph neural network** is applied to reason over **intra-image** and **inter-image** relations, enabling robust matching across **viewpoint**, **modality**, and **time** gaps.

## 🎯 Motivation

Cross-domain and cross-modal remote sensing image geo-localization remains challenging due to large appearance discrepancies and unstable semantic correspondence across heterogeneous sensors and platforms. Existing methods mainly rely on scene-level or global representations, which often struggle to achieve reliable alignment in complex environments, especially under severe modality gaps such as infrared-to-visible matching.

To facilitate research in this setting, we introduce **IRVL328**, a new infrared-visible remote sensing localization dataset designed to reflect challenging cross-modal variations.

## 🧠 Framework

1. **Object Detection**  
   Extract salient structural regions from UAV and satellite images.

2. **Graph Construction**  
   Represent detected objects as nodes and their spatial/semantic relations as edges.

3. **Graph Neural Reasoning**  
   Jointly model inter-image correspondences and intra-image relations through dual-graph reasoning.

4. **Cross-domain Matching**  
   Perform robust retrieval and localization across viewpoint, modality, and time differences.

---

## 📂 Dataset

### IRVL328

We release the dataset used in our experiments. The directory organization follows:

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

so that existing codebases can be easily adapted.

### Download

#### Baidu Netdisk (Dataset)

- **Link:** <https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj>
- **Code:** `52bj`

#### Baidu Netdisk (Pretrained Weights)

- **Link:** <https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA>
- **Code:** `52sb`

Pretrained models for object detection / graph matching modules.

#### Google Drive (Dataset)

- **Link:** <https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing>

### Directory Structure

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
    │   │   └── 0200/
    │   └── satellite/
    │       ├── 0000/
    │       │   └── xx.png
    │       ├── 0001/
    │       │   └── xx.png
    │       └── ...
    │       └── 0200/
    ├── query_drone/
    ├── query_satellite/
    ├── gallery_drone/
    └── gallery_satellite/

**Notes**

- Folder style follows University1652-Baseline / SUES-200-Benchmark conventions.
- UAV (drone) images: `.jpg`
- Satellite images: `.png`

---

## 🔧 Code Availability

Currently, we publicly release:

- Code for the **object detection** module
- Code for the **graph construction** module

The **complete training and inference pipeline** will be released after official publication. Limited components may be provided for non-commercial academic research upon request.

### Requesting Code Before Publication

Please email **liutao23@njust.edu.cn** with the following information:

- **Subject:** Code Request for Cross-View UAV Localization
- Your name
- Affiliation
- Usage purpose (research / education)
- A brief statement agreeing to non-commercial use only

After the paper is accepted, the complete codebase will be released in this repository.

---

## 📖 Citation

If this dataset or code is useful in your research, please cite:

    @article{LIU2026104451,
      title = {Object-aware graph matching network for cross-domain remote sensing image localization},
      journal = {Chinese Journal of Aeronautics},
      pages = {104451},
      year = {2026},
      issn = {1000-9361},
      doi = {https://doi.org/10.1016/j.cja.2026.104451},
      url = {https://www.sciencedirect.com/science/article/pii/S1000936126003894},
      author = {Tao LIU and Kan REN and Qian CHEN},
      keywords = {Remote sensing, Image matching, Object detection, Graph neural networks, Image retrieval, Unmanned aerial vehicles},
      abstract = {Cross-domain and cross-modal remote sensing image geo-localization remains challenging due to large appearance discrepancies and unstable semantic correspondence across heterogeneous sensors and platforms. Existing methods mainly rely on scene-level or global representations, which often struggle to achieve reliable alignment in complex environments, especially under severe modality gaps such as infrared-to-visible matching. To facilitate research in this setting, this study introduces IRVL328, a new infrared-visible remote sensing localization dataset designed to reflect challenging cross-modal variations. Meanwhile, this study proposes an object-aware graph matching framework that integrates object detection with dual-graph neural reasoning, where salient structural regions are represented as graph nodes and both inter-image correspondences and intra-image relations are jointly modeled; a training-only node alignment strategy is further introduced to enhance supervision without increasing inference complexity. Experimental results show that the proposed method achieves competitive performance on SUES-200 and strong performance on IRVL328 and DenseUAV, particularly in challenging cross-modal settings.}
    }

---

## 🙏 Acknowledgements

We acknowledge:

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

for their dataset organization conventions and baseline implementations.

---

## 📮 Contact

For questions or collaboration, please contact:

**Tao Liu**  
Email: liutao23@njust.edu.cn

---

# 中文说明

## 作者寄语

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

## 简介

本仓库配套我们关于 **跨域遥感图像定位** 的研究。

我们首先通过 **目标检测** 从无人机 / 卫星图像中提取显著目标实例，然后构建以目标为节点、空间/语义关系为边的图结构，并使用 **图神经网络** 在图层面建模图像内与图像间关系，从而实现跨视角、跨模态、跨时间的鲁棒匹配与定位。

## 当前开源范围

目前已公开：

- **目标检测模块**代码
- **图构建（Graph Construction）模块**代码

完整代码（包括训练、推理及全流程实现）将在论文被正式接收后，于本仓库统一开放。在此之前，如因科研 / 教学需要获取更多实现细节，可邮件联系作者。

## 数据集

我们提供实验所用数据集，其目录结构遵循：

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

以方便社区直接复用现有检索框架和评测脚本。

### 下载地址

#### 百度网盘（数据集）

- 链接：<https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj>
- 提取码：`52bj`

#### 百度网盘（预训练权重）

- 链接：<https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA>
- 提取码：`52sb`

提供目标检测 / 图匹配相关预训练模型。

#### Google Drive（数据集）

- 链接：<https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing>

### 目录结构

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
    │   │   └── 0200/
    │   └── satellite/
    │       ├── 0000/
    │       │   └── xx.png
    │       ├── 0001/
    │       │   └── xx.png
    │       └── ...
    │       └── 0200/
    ├── query_drone/
    ├── query_satellite/
    ├── gallery_drone/
    └── gallery_satellite/

说明：

- 目录组织与 University1652-Baseline / SUES-200-Benchmark 一致，方便对齐社区已有方法。
- 无人机（drone）图像为 `.jpg`，卫星（satellite）图像为 `.png`。

## 代码获取说明

论文被接收后，我们将在本仓库公开完整实现，便于复现和拓展。

## 引用

如本数据集或代码对您的研究有帮助，请引用：

    @article{LIU2026104451,
      title = {Object-aware graph matching network for cross-domain remote sensing image localization},
      journal = {Chinese Journal of Aeronautics},
      pages = {104451},
      year = {2026},
      issn = {1000-9361},
      doi = {https://doi.org/10.1016/j.cja.2026.104451},
      url = {https://www.sciencedirect.com/science/article/pii/S1000936126003894},
      author = {Tao LIU and Kan REN and Qian CHEN},
      keywords = {Remote sensing, Image matching, Object detection, Graph neural networks, Image retrieval, Unmanned aerial vehicles},
      abstract = {Cross-domain and cross-modal remote sensing image geo-localization remains challenging due to large appearance discrepancies and unstable semantic correspondence across heterogeneous sensors and platforms. Existing methods mainly rely on scene-level or global representations, which often struggle to achieve reliable alignment in complex environments, especially under severe modality gaps such as infrared-to-visible matching. To facilitate research in this setting, this study introduces IRVL328, a new infrared-visible remote sensing localization dataset designed to reflect challenging cross-modal variations. Meanwhile, this study proposes an object-aware graph matching framework that integrates object detection with dual-graph neural reasoning, where salient structural regions are represented as graph nodes and both inter-image correspondences and intra-image relations are jointly modeled; a training-only node alignment strategy is further introduced to enhance supervision without increasing inference complexity. Experimental results show that the proposed method achieves competitive performance on SUES-200 and strong performance on IRVL328 and DenseUAV, particularly in challenging cross-modal settings.}
    }

## 致谢

感谢：

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

在数据集设计与目录规范上的重要参考。

## 联系

如有问题或合作意向，请联系：

**刘涛**  
邮箱：liutao23@njust.edu.cn
