> **Knowledge Distillation with Predicted Depth for Robust and Lightweight Face Presentation Attack Detection**

---

## Overview
This project augments face anti-spoofing (FacePAD) with **synthetic depth cues** predicted from RGB frames via a monocular depth estimator. We provide multiple models to augment these depth-maps along with RGB video frame for classification; in early and late fusion appraoches. For **knowledge distillation (KD)**; the **teacher** model from our paper uses a **dual-branch** network (RGB branch + depth branch) and fuses features before classification; the **student** learns to mimic the teacher via **KD** and runs **without** depth at inference time.

- Depth highlights 3D structure that is difficult to fake with print/video replays.
- KD keeps deployment **fast and light** (mobile-friendly), while retaining most of the teacher’s accuracy.

---

## Note
Please note that we have only provided unified code dataset/model classes and utilized functions for processing/training. We plan to test the update the curernt form of code, and provide remaining methods. For questions, or bugs, please open an **Issue** on this repository.

---

## Key Features
- **Depth-augmented dual branch:** RGB textures + depth structure fused (feature concatenation) before the classifier.
- **KD for real-time use:** Distill the dual-branch teacher into a single-branch MobileNet student (RGB-only inference).
- **Standard FacePAD metrics:** HTER, EER, AUC-ROC; and for OULU-NPU, APCER/BPCER/ACER.
- **Simple codebase:** Single training script and compact model/dataset utilities.

---

## Repository Structure
```
.
├── datasets.py   # dataset indexing, transforms, loaders
├── models.py     # dual-branch teacher and compact KD student
├── train.py      # training & evaluation entry point (teacher / student)
├── utils.py      # metrics, logging, helpers
└── LICENSE       # MIT License
```

---

## Setup

I used python 3.9x based conda env and dependencies.

---

## Datasets
This repo supports four standard FacePAD benchmarks (data: video files):
- **Replay-Attack**
- **Replay-Mobile**
- **ROSE-Youtu**
- **OULU-NPU**

> Frames are typically resized to **224×224** during preprocessing/augmentation.

---

## Preparing Depth Maps
The **teacher** (and other depth-augmented models) consumes RGB frames **and** their **aligned** depth maps. You can generate depth maps offline with monocular depth estimator (we used *Depth-Anything V2, as reported in the paper).

---

## Training

> Run `python train.py -h` to see **all available options** in your environment.  

### Train the Dual-Branch Teacher (RGB+Depth)
- **Backbone:** MobileNetV3-Large for RGB and depth branches.  
- **Fusion:** feature concatenation before the classifier.  
- **Typical hyper-parameters (paper):** Adam, LR `1e-4`, batch size `16`, up to `100` epochs with early stopping on val loss.

### Train the KD Student (RGB-only)
The student learns from:
1) ground-truth labels (cross-entropy), and  
2) the teacher’s **soft logits** (KL with temperature **T**, weighted by **α**).

> **Notes**
> - The student runs **without** depth at inference time.
> - During training, the teacher requires depth to produce soft logits for distillation.
> - Our KD parameters: `kd-alpha ∈ [0.7]`, `kd-T ∈ [3]`, for all datasets utilized.

See the logs directory for logs & training checkpoints.

---

## Evaluation & Metrics
The training script reports standard metrics:
- **HTER**, **EER**, **AUC-ROC**
- **APCER/BPCER/ACER** (for OULU-NPU protocols)

---

## Tips
- **Backbone choice:** We provide MobileNetV3-Large for a strong accuracy/efficiency trade-off; alternative backbones are quite easy to update in model classes.
- **Fusion:** Simple **concatenation** of RGB/depth features generally works best before the classifier in our experiments.
- **Depth runtime cost:** Generate depth **offline** for training the teacher; the KD student **does not** require depth at inference.

---

## Acknowledgements
- The monocular depth estimator used in the paper was **Depth-Anything-V2 (Base)**; thanks to its authors and community.
- Thanks to the providers of the **Replay-Attack**, **Replay-Mobile**, **ROSE-Youtu**, and **OULU-NPU** datasets.

---

## Contact
For questions, or bugs, please open an **Issue** on this repository.

---

## Citation
If you find this repository useful, please cite:

**M. S. Jabbar, T. H. M. Siddique, K. Huang, S. Khan**  
*Knowledge distillation with predicted depth for robust and lightweight face presentation attack detection*.  
**Knowledge-Based Systems**, 329 (2025) 114325.  
https://doi.org/10.1016/j.knosys.2025.114325

```bibtex
@article{JABBAR2025114325,
title = {Knowledge distillation with predicted depth for robust and lightweight face presentation attack detection},
journal = {Knowledge-Based Systems},
volume = {329},
pages = {114325},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114325},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125013656},
author = {Muhammad Shahid Jabbar and Taha Hasan Masood Siddique and Kejie Huang and Shujaat Khan},
}
```

---
