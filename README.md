
# 🧠 Fast & Efficient Knowledge Distillation with Only 5 Epochs

This project demonstrates a novel approach to training compact deep learning models using **Knowledge Distillation** in just **five epochs**. Our distilled student models **outperform their teacher models** and **match or exceed** state-of-the-art benchmarks on **FashionMNIST** and **Food101** datasets.

## 📖 Overview

Training deep models usually requires extensive time and resources. We address this challenge using knowledge distillation to transfer knowledge from a large pre-trained teacher (BiT-M ResNet-50) to smaller student networks (MobileNetV3, EfficientNet-B0, and B3), all trained in only 5 epochs. Despite the constrained training, the student models achieve outstanding performance.

## 🧠 Key Features

- ✅ 5-Epoch Knowledge Distillation Strategy
- ✅ Student Models Surpass Teachers
- ✅ Cross-dataset Evaluation (Simple + Complex)
- ✅ Baseline vs KD Comparison Included
- ✅ Real-world Deployment Potential

## 📊 Datasets Used

- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

## 🔍 How It Works

- **Teacher Model**: BiT-M ResNet-50 (pre-trained)
- **Student Models**: MobileNetV3, EfficientNet-B0, EfficientNet-B3
- **Distillation Setup**: Temperature scaling + combined KL + CE loss
- **Training**: Both teacher and student models trained for only 5 epochs

## 📁 Project Structure

```
├── pytorch.ipynb          # Teacher training (FashionMNIST)
├── pytorch_f101.ipynb     # Teacher training (Food101)
├── pytorch_stu.ipynb      # Student distillation (FashionMNIST)
├── pytorch_stu2.ipynb     # Student distillation (Food101)
├── results/               # Accuracy logs and model comparisons
├── models/                # Saved models (optional)
└── README.md              # This file
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies (if a requirements.txt is provided)
pip install -r requirements.txt
```

## 📈 Results

| Model             | Dataset       | No-KD Accuracy | KD Accuracy |
|------------------|---------------|----------------|-------------|
| MobileNetV3       | FashionMNIST  | 94.94%         | 96.17%      |
| EfficientNet-B0   | Food101       | 23.07%         | 82.86%      |
| EfficientNet-B3   | Food101       | N/A            | 84.11%      |


## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Hinton et al. for foundational KD work
- Google Research for BiT-M and EfficientNet architectures
- FashionMNIST & Food101 dataset creators