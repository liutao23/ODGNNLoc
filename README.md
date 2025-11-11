```markdown
# ODGNNLoc

## Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization

[English]| [中文说明]

---

## English

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
IR-VL328/
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
````

**Notes**

* Folder style follows [University1652-Baseline](https://github.com/layumi/University1652-Baseline) / [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark) conventions.
* UAV (drone) images: `.jpg`
* Satellite images: `.png`

---

### How to get the full code

Before acceptance, we may share additional parts of the implementation **by request for non-commercial research/education use**.

Please email:

**[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)**

with:

* Subject: `Code Request for Cross-View UAV Localization`
* Your name
* Affiliation
* Usage purpose (research / education)
* A brief statement agreeing to **non-commercial use only**

After the paper is accepted, the **complete codebase** will be released in this repository.

---

### Citation

If this dataset or code is useful in your research, please cite:

```bibtex
@misc{liu2025objectdetectionoptionalbasis,
      title={Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization},
      author={Tao Liu and Kan Ren and Qian Chen},
      year={2025},
      eprint={2511.02489},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.02489},
}
```

---

### Acknowledgments

We acknowledge:

* [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
* [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

for their dataset organization conventions and baseline implementations.

---

## 中文说明

### 简介

本仓库配套我们关于 **跨视角无人机定位** 的研究。

我们首先通过 **目标检测** 从无人机 / 卫星图像中提取显著目标实例，然后构建以目标为节点、空间/语义关系为边的图结构，并使用 **图神经网络** 在图层面建模图像内与图像间关系，从而实现跨时间、跨视角、跨模态的鲁棒匹配与定位。

### 当前开源范围

目前已公开：

* **目标检测模块代码**
* **图构建（Graph Construction）模块代码**

**完整代码（包括训练、推理及全流程实现）将在论文被正式接收后，于本仓库统一开放。**

在此之前，如因科研 / 教学需要获取更多实现细节，可邮件联系作者。

---

### 数据集

我们提供实验所用数据集，其目录结构遵循：

* [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
* [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

以方便社区直接复用现有检索框架和评测脚本。

#### 下载地址

* **百度网盘（数据集）**
  链接：[https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj](https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj)
  提取码：**52bj**

* **百度网盘（预训练权重）**
  链接：[https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA](https://pan.baidu.com/s/17QHtGe5YWN-g6inP94h7gA)
  提取码：**52sb**
  —— 提供目标检测 / 图匹配相关预训练模型

* **Google Drive（数据集）**
  链接：[https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing](https://drive.google.com/file/d/1okFeWJIuZ49TnkZkkoOYTOl_b0THwUdc/view?usp=sharing)

#### 目录结构

```text
IR-VL328/
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

**说明**

* 目录组织与 [University1652-Baseline](https://github.com/layumi/University1652-Baseline) / [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark) 一致，方便对齐社区已有方法。
* 无人机（drone）图像为 `.jpg`，卫星（satellite）图像为 `.png`。

---

### 代码获取说明

在论文正式接收前，如需进一步代码（仅限科研 / 教学、禁止商业用途），请发送邮件至：

**[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)**

建议格式：

* 标题：`Code Request for Cross-View UAV Localization`
* 正文：姓名、单位、使用用途（科研 / 教学）、以及仅用于非商业用途的简要承诺。

论文被接收后，我们将在本仓库公开 **完整实现**，便于复现和拓展。

---

### 引用

如本数据集或代码对您的研究有帮助，请引用：

```bibtex
@misc{liu2025objectdetectionoptionalbasis,
      title={Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization},
      author={Tao Liu and Kan Ren and Qian Chen},
      year={2025},
      eprint={2511.02489},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.02489},
}
```

---

### 致谢

感谢：

* [University1652-Baseline](https://github.com/layumi/University1652-Baseline)
* [SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

在数据集设计与目录规范上的重要参考。

