# 🛡️ Trustworthy Federated Learning for Heart Disease Prediction

A comprehensive framework demonstrating privacy-preserving, secure, and fair machine learning for healthcare applications using Federated Learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-✓-green)
![Privacy](https://img.shields.io/badge/Differential%20Privacy-✓-blueviolet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This project implements a **Trustworthy AI** pipeline for heart disease prediction that addresses key challenges in healthcare ML:
- **Privacy Preservation** through Federated Learning and Differential Privacy
- **Security** against adversarial attacks  
- **Fairness** across demographic groups
- **Explainability** with SHAP and LIME
- **Performance** comparable to centralized approaches

**Key Finding**: 🎯 **Federated Learning achieves 86.7% accuracy, outperforming centralized approaches (83.3%) while preserving patient privacy.**

## 🎯 Key Features

| Feature | Implementation | Result |
|---------|----------------|--------|
| **Federated Learning** | Custom FedAvg implementation | **86.7% accuracy** |
| **Differential Privacy** | DP-SGD with privacy accounting | 85.0% accuracy (ε=15.33) |
| **Adversarial Robustness** | FGSM attacks evaluation | **LOW risk** level |
| **Fairness Analysis** | Disparate Impact mitigation | Gender bias detection |
| **Explainability** | SHAP & LIME interpretations | Feature importance analysis |

## 📊 Performance Comparison

| Model | Accuracy | Privacy Level | Security |
|-------|----------|---------------|----------|
| **Federated Learning** | **86.7%** | Medium | Medium |
| Centralized NN | 83.3% | Low | Medium |
| DP-Trained NN | 85.0% | High | Medium |
| Random Forest | 85.0% | Low | Low |

## 🏗️ Architecture

```
📁 trustworthy-fl-heart-disease/
├── 📁 notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_centralized_baseline.ipynb
│   ├── 3_federated_learning_tff.ipynb
│   ├── 4_explainability_shap_lime.ipynb
│   ├── 5_fairness_analysis_mitigation.ipynb
│   ├── 6_adversarial_security.ipynb
│   └── 7_differential_privacy.ipynb
├── 📁 src/
│   ├── data_loader.py
│   ├── models.py
│   ├── federated_tff.py
│   ├── adversarial.py
│   ├── privacy.py
│   └── explainability.py
├── 📁 results/
├── run_all.py
└── requirements.txt
```

## 🖥️ Application Screen Recordings

### 📊 Dashboard
![Dashboard](https://github.com/user-attachments/assets/68584f64-dce3-41fd-91de-84929f303caf)

---

### 📈 Model Comparison
![Model comparison](https://github.com/user-attachments/assets/30fb7d54-2693-4d3f-833d-39e07463967a)


---

### 🔮 Real-Time Prediction
![Realtime prediction](https://github.com/user-attachments/assets/51d2ec1b-e217-43f3-92ff-57d2e577fbbe)


---

### 📉 Training Metrics
![Training metrics](https://github.com/user-attachments/assets/3656a501-3d48-4353-a778-c63f3e60862e)



## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/trustworthy-fl-heart-disease.git
cd trustworthy-fl-heart-disease
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
cd notebooks
python run_all.py
```

### Run Individual Components

```python
from src.data_loader import load_and_preprocess_data
X, y, scaler, features = load_and_preprocess_data()
```

```python
from src.models import create_keras_model
model = create_keras_model(13)
model.fit(X, y, epochs=50)
```

```python
from src.federated_tff import build_federated_trainer
trainer = build_federated_trainer(input_dim=13)
global_model = trainer(client_data, rounds=50)
```

## 📈 Results & Findings

### Federated Learning Performance
- Accuracy: 86.7%
- Privacy: Medium
- Scalability: 5 simulated hospitals

### Privacy-Accuracy Tradeoff

| Privacy Level | Accuracy | ε |
|---------------|----------|---|
| No DP | 83.3% | ∞ |
| Weak DP | 81.7% | 76.66 |
| Medium DP | 85.0% | 15.33 |
| Strong DP | 83.3% | 7.67 |

### Security Analysis
- Clean Accuracy: 83.3%
- Adversarial Accuracy: 70.0%
- Robustness Drop: 13.3%

## 🔬 Research Contributions

1. Federated Learning outperforms centralized approaches  
2. Differential Privacy improves generalization  
3. Privacy-preserving ML maintains accuracy  
4. End-to-end trustworthy ML pipeline for healthcare

## 🛠️ Technical Implementation

### Federated Learning
```python
def federated_train(client_data, rounds=50):
    for round in range(rounds):
        local_models = [train_local(client) for client in client_data]
        global_model = average_weights(local_models)
    return global_model
```

### Differential Privacy
```python
def train_with_dp(model, X_train, y_train, noise_multiplier=1.0):
    for batch in data:
        gradients = compute_gradients(batch)
        gradients = clip_gradients(gradients, clip_norm=1.0)
        noisy_gradients = add_gaussian_noise(gradients, noise_multiplier)
        model.update(noisy_gradients)
```

## 📚 Dataset

- Source: UCI Heart Disease Dataset (Cleveland)
- Samples: 297
- Features: 13
- Target: Binary classification

## 🎮 Usage Examples

```python
from src.federated_tff import build_federated_trainer
trainer = build_federated_trainer(input_dim=13)
client_data = simulate_hospitals(X_train, y_train, n_clients=5)
global_model = trainer(client_data, rounds=50, verbose=1)
```

```python
from src.explainability import explain_with_shap, explain_with_lime
explain_with_shap(model, X_train, X_test, features)
explain_with_lime(model, X_train, features, instance_idx=0)
```

## 📄 Citation

```bibtex
@software{trustworthy_fl_heart_2024,
  title = {Trustworthy Federated Learning for Heart Disease Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/trustworthy-fl-heart-disease}
}
```

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to the branch  
5. Open a Pull Request  

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- UCI Machine Learning Repository  
- TensorFlow Federated Team  
- Privacy and Fairness Research Community  

**⭐ Star the repo if you find it useful!**



