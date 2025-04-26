# Knowledge Distillation for Efficient Deep Learning

This project implements a knowledge distillation framework where a lightweight student model is trained to mimic a larger, more accurate teacher model. The goal is to achieve competitive accuracy while reducing the computational load, making it suitable for deployment on resource-constrained devices such as mobile phones, IoT systems, and embedded AI platforms.

## 📁 Repository Structure

```
.
├── data/                        # Dataset-related info or download instructions
├── models/                      # Pretrained models (.pth files)
├── notebooks/                   # Jupyter notebooks for training and evaluation
├── scripts/                     # Python scripts for evaluation and utilities
├── student_models/              # MobileNetV3 / EfficientNet variants
├── requirements.txt             # List of required Python packages
├── README.md                    # Project overview and usage instructions
├── .gitignore                   # Files/folders to ignore in Git
└── LICENSE                      # License file
```

## 🧠 Project Overview

- **Teacher Model**: ResNetv2-50x1
- **Student Models**: EfficientNet-Lite0, EfficientNet-B0, MobileNetV3-Small
- **Datasets Used**: Fashion-MNIST (grayscale clothing), Food101 (RGB food images)
- **Goal**: Optimize deep learning models for edge deployment without major accuracy loss.

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Recommended environment:
- Python 3.9+
- PyTorch 2.x
- CUDA enabled (if available)

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knowledge-distillation-efficient.git
cd knowledge-distillation-efficient
```

2. Run Jupyter notebooks from the `notebooks/` folder to train and evaluate models.

3. Use `scripts/evaluate_models.py` for quick evaluation across models.

## 🧪 Results Snapshot

| Model            | Dataset       | Accuracy | Params  | Epochs |
|------------------|---------------|----------|---------|--------|
| ResNetv2-50x1    | Fashion-MNIST | ~93%     | 23.5M   | 5      |
| EfficientNet-Lite0 | Fashion-MNIST | ~96.17% | 4.7M    | 5      |
| MobileNetV3-Small | Fashion-MNIST | ~89%    | 2.9M    | 5      |
| ResNetv2-50x1    | Food101       | ~81.72%  | 23.5M   | 5      |
| EfficientNet-B0  | Food101       | ~82.86%  | 5.3M    | 5      |

## 📂 Datasets

- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist
- **Food101**: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

## 📄 License

This project is licensed under the MIT License.

---

Feel free to fork, contribute, or use the models for your research or mobile AI applications!