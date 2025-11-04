# ODGNNLoc
---

# Object Detection Can Suffice: A Graph Matching Network for Cross-View UAV Localization

[English](#english) | [中文说明](#中文说明)

---

## English

### Overview

This repository accompanies our work on cross-view UAV localization. We leverage **object detection** to extract salient instances from UAV/satellite images, and apply a **graph neural network** to reason over intra-/inter-image relations, enabling robust matching across time, view, and modality gaps.

> **Code availability:** To obtain the code used in our paper, please email **[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)** with your name, affiliation, and intended use (research only).

---

### Dataset

We release the dataset used in our experiments (aligned with the directory style of **University1652** and **SUES200**).

**Download links**

* **Baidu Netdisk:** 链接: [https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj](https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj)  提取码: **52bj**
  *Shared via Baidu Netdisk Super Member v4*
* **Google Drive:** *(to be added)*

**Directory structure**

```
root/
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

> Notes
>
> * Folders follow **University1652 / SUES200** conventions.
> * UAV (drone) images are `.jpg`; satellite images are `.png`.

---

### How to get the code

Please send an email to **[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)** with:

* Subject: `Code Request for Cross-View UAV Localization`
* Body: your name, affiliation, purpose (research/education), and a brief statement agreeing to non-commercial use.

---

### Citation

If this dataset or code is useful in your research, please cite:

```bibtex
@article{ODGNNLoc,
  title   = {Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization},
  journal = { },
  volume  = {},
  pages   = {},
  year    = {2024},
  month   = {Jan},
  doi     = {10.1109/tim.2023.3342849},
  author  = {Liu, Tao and Ren, Kan and Chen, Qian},
  language= {English}
}
```

---

### Acknowledgments

We acknowledge **University1652** and **SUES200** for their dataset organization conventions.

---

## 中文说明

### 简介

本仓库配套我们关于**跨视角无人机定位**的研究。我们先用**目标检测**从无人机/卫星图像中提取显著目标，再利用**图神经网络**在图层面建模图像内与图像间关系，从而实现跨时间、跨视角、跨模态的稳健匹配。

> **代码获取：** 论文实现代码可邮件索取，请发送至 **[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)**，并附上姓名、单位与用途（仅限科研/教学）。

---

### 数据集

我们提供实验所用数据集（目录结构遵循 **University1652** 与 **SUES200**）。

**下载地址**

* **百度网盘：** 链接: [https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj](https://pan.baidu.com/s/1yarjck2JBaXJ7s3nDZaITw?pwd=52bj)  提取码：**52bj**
  —— 来自百度网盘超级会员v4的分享
* **谷歌硬盘：** *待更新*

**目录结构**

```
root/
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

> 说明
>
> * 目录与 **University1652 / SUES200** 一致。
> * 无人机（drone）图像为 `.jpg`，卫星（satellite）图像为 `.png`。

---

### 代码索取方式

请发送邮件至 **[liutao23@njust.edu.cn](mailto:liutao23@njust.edu.cn)**，标题建议：`Code Request for Cross-View UAV Localization`，正文请包含：

* 姓名、单位；
* 用途（科研/教学）；
* 简述承诺**仅限非商业用途**。

---

### 引用

若本数据/代码对您的研究有帮助，请引用下述文献：

```bibtex
@article{ODGNNLoc,
  title   = {Object Detection as an Optional Basis: A Graph Matching Network for Cross-View UAV Localization},
  journal = { },
  volume  = {},
  pages   = {},
  year    = {2024},
  month   = {Jan},
  doi     = {10.1109/tim.2023.3342849},
  author  = {Liu, Tao and Ren, Kan and Chen, Qian},
  language= {English}
}
```

---

### 致谢

感谢 **University1652** 与 **SUES200** 数据集在目录组织上的启发。

---

