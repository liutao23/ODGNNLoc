``` markdown
# IRVL328

> A cross-domain UAV localization framework based on object-aware graph matching and structured semantic reasoning.

## Object-aware Graph Matching Network for Cross-domain UAV Localization

[English]| [中文说明]

---

## English Version

---

## A Note from the Author

This work was completed in November 2023. During the following years, the manuscript was submitted to several journals, including TIP, TGRS, KBS, and JSTARS, before finally being accepted after nearly three years.

I have always believed that the core idea of this work is meaningful: using modern object detection techniques to extract structured semantic information and combining them with existing graph matching methods provides a simple and interpretable framework for cross-view localization. With the continuous advancement of object detection algorithms, this framework may naturally benefit from stronger detectors in the future.

However, at the time this work was conducted, many components were implemented from scratch, and despite extensive experiments, the performance was not as competitive as some CNN-based or Transformer-based approaches. Most reviewer concerns focused on quantitative performance rather than the novelty of the idea. This experience also made me reconsider whether this research direction should be continued.

Fortunately, this work was eventually accepted. As my first research paper during my Ph.D., it carries special meaning to me. Although the idea itself may no longer be considered novel today, I hope that the released dataset, especially the IRVL328 dataset, can provide a useful benchmark for evaluating model generalization ability under challenging cross-view and cross-modal scenarios.

Recently, I also explored whether modern large language models could help optimize the model through hyperparameter tuning. However, after further analysis, I realized that the limitations are not simply caused by parameter settings. The University1652, SUES-200, and IRVL328 datasets contain relatively limited objects per scene, and individual satellite tiles often provide only sparse graph nodes. Therefore, graph matching methods may inherently face difficulties in these scenarios. Such approaches may demonstrate stronger potential on large-scale urban localization datasets with richer object distributions.

For these reasons, I decided not to release the complete training pipeline at this stage. The implementation itself is relatively straightforward, and the involved techniques can be reproduced using existing AI tools(which can be reproduced using existing AI development tools). More importantly, I hope future researchers focus on choosing methods that match their goals: object detection combined with graph matching provides interpretability and structural reasoning ability, but it may not always be the optimal choice when the primary objective is achieving the highest possible retrieval accuracy.

I sincerely hope this dataset can help the community study generalization, robustness, and cross-view localization. Best wishes for your research.

---

### Overview

This repository accompanies our work on **cross-view UAV localization**.

We first use **object detection** to extract salient instances from UAV / satellite images, and then construct graphs where objects act as nodes and spatial/semantic relations as edges. A **graph neural network** is applied to reason over **intra-image** and **inter-image** relations, enabling robust matching across **time**, **viewpoint**, and **modality** gaps.

### Code availability

Currently, we publicly release:

- Code for the **object detection** module
- Code for the **graph construction** module

The **full implementation** of our method (including training, inference, and the complete pipeline) will be **released after the paper is officially accepted**.

---

### Dataset

We release the dataset used in our experiments, organized in the same style as:

- [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

so that existing codebases can be easily adapted.

#### Download links

- **Baidu Netdisk (Dataset)**  
  Link: https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj  
  Code: **52bj**  
  *Shared via Baidu Netdisk Super Member v4.*

- **Baidu Netdisk (Pretrained Weights)**  
  Link: https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA  
  Code: **52sb**  
  *Pretrained models for our object detection / graph modules.*

- **Google Drive (Dataset)**  
  Link: https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing

#### Directory structure

```text
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
```
Notes
Folder style follows
University1652-Baseline
/
SUES-200-Benchmark
conventions.
UAV (drone) images: `.jpg`
Satellite images: `.png`
---
How to get the full code
Before acceptance, we may share additional parts of the implementation
by request for non-commercial research/education use.
Please email:
liutao23@njust.edu.cn
with:
Subject: `Code Request for Cross-View UAV Localization`
Your name
Affiliation
Usage purpose (research / education)
A brief statement agreeing to non-commercial use only
After the paper is accepted, the complete codebase will be released
in this repository.
---
Citation
If this dataset or code is useful in your research, please cite:
``` bibtex
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
```
---
Acknowledgments
We acknowledge:
University1652-Baseline
SUES-200-Benchmark
for their dataset organization conventions and baseline implementations.
---
中文版本
---
作者寄语
本文工作完成于 2023 年 11 月。之后的几年中，论文经历了多次投稿，包括
TIP、TGRS、KBS、JSTARS 等期刊，最终历时近三年被接收。
我一直认为这篇工作的核心思想具有一定价值：利用先进的目标检测技术提取结构化语义信息，并结合现有的图匹配方法，可以构建一个简单、直观且具有解释性的跨视角定位框架。随着目标检测技术不断发展，该框架理论上也能够从更强的检测器中获益。
然而，在论文完成的时期，许多模块均为自主实现。尽管进行了大量实验，但模型性能仍然无法与部分基于
CNN 或 Transformer
的方法相比。在审稿过程中，许多意见主要集中于指标提升，而较少关注方法本身的创新性。这也曾让我一度考虑是否继续推进这项工作。
幸运的是，这篇论文最终得到了认可。作为我博士期间的第一篇论文，它对我具有特殊意义。虽然如今该想法可能已经不再新颖，但我希望公开的数据集，尤其是
IRVL328
数据集，能够为研究者测试模型泛化能力、鲁棒性以及跨视角定位提供一定帮助。
最近一段时间，我也尝试思考是否可以利用 ChatGPT
等工具辅助模型调参。然而经过分析后，我认为问题并不仅仅来自参数设置。University1652、SUES-200
以及 IRVL328
数据集中，单幅图像包含的目标数量有限，卫星瓦片中的有效目标节点较少，因此图匹配方法在这种场景下天然受到限制。未来，在具有更丰富目标分布的大规模城市级地理定位数据集上，这类方法可能会展现更大的潜力。
因此，目前我没有公开完整训练代码。一方面，该方法实现逻辑较为简单，所使用的技术在当前
AI
时代已经较容易复现（即便是豆包也可以）；另一方面，我也希望研究者根据实际目标选择合适的方法。如果目标是追求最高指标，目标检测结合图匹配可能并不是最优路线；但如果关注结构化理解、可解释性以及跨域泛化能力，这种思路仍具有一定研究价值。
感谢所有关注和使用本数据集的研究者。希望 IRVL328
能够帮助大家更好地研究跨视角定位中的泛化性和鲁棒性。
祝科研顺利。
---
简介
本仓库配套我们关于 跨视角无人机定位 的研究。
我们首先通过 目标检测 从无人机 /
卫星图像中提取显著目标实例，然后构建以目标为节点、空间/语义关系为边的图结构，并使用
图神经网络
在图层面建模图像内与图像间关系，从而实现跨时间、跨视角、跨模态的鲁棒匹配与定位。
当前开源范围
目前已公开：
目标检测模块代码
图构建（Graph Construction）模块代码
完整代码（包括训练、推理及全流程实现）将在论文被正式接收后，于本仓库统一开放。
在此之前，如因科研 / 教学需要获取更多实现细节，可邮件联系作者。
---
数据集
我们提供实验所用数据集，其目录结构遵循：
University1652-Baseline
SUES-200-Benchmark
以方便社区直接复用现有检索框架和评测脚本。
下载地址
百度网盘（数据集）
链接：https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj
提取码：52bj
百度网盘（预训练权重）
链接：https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA
提取码：52sb ------ 提供目标检测 / 图匹配相关预训练模型
Google Drive（数据集）
链接：https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing
目录结构
``` text
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
```
说明
目录组织与
University1652-Baseline
/
SUES-200-Benchmark
一致，方便对齐社区已有方法。
无人机（drone）图像为 `.jpg`，卫星（satellite）图像为 `.png`。
---
代码获取说明
论文被接收后，我们将在本仓库公开 完整实现，便于复现和拓展。
---
引用
如本数据集或代码对您的研究有帮助，请引用：
@article{LIU2026104451, title = {Object-aware graph matching network for
cross-domain remote sensing image localization}, journal = {Chinese
Journal of Aeronautics}, pages = {104451}, year = {2026}, issn =
{1000-9361}, doi = {https://doi.org/10.1016/j.cja.2026.104451}, url =
{https://www.sciencedirect.com/science/article/pii/S1000936126003894},
author = {Tao LIU and Kan REN and Qian CHEN}, keywords = {Remote
sensing, Image matching, Object detection, Graph neural networks, Image
retrieval, Unmanned aerial vehicles}, abstract = {Cross-domain and
cross-modal remote sensing image geo-localization remains challenging
due to large appearance discrepancies and unstable semantic
correspondence across heterogeneous sensors and platforms. Existing
methods mainly rely on scene-level or global representations, which
often struggle to achieve reliable alignment in complex environments,
especially under severe modality gaps such as infrared-to-visible
matching. To facilitate research in this setting, this study introduces
IRVL328, a new infrared-visible remote sensing localization dataset
designed to reflect challenging cross-modal variations. Meanwhile, this
study proposes an object-aware graph matching framework that integrates
object detection with dual-graph neural reasoning, where salient
structural regions are represented as graph nodes and both inter-image
correspondences and intra-image relations are jointly modeled; a
training-only node alignment strategy is further introduced to enhance
supervision without increasing inference complexity. Experimental
results show that the proposed method achieves competitive performance
on SUES-200 and strong performance on IRVL328 and DenseUAV, particularly
in challenging cross-modal settings.} }
---
致谢
感谢：
University1652-Baseline
SUES-200-Benchmark
在数据集设计与目录规范上的重要参考。
