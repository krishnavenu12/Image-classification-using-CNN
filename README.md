# 🧠 Multi-Output Image Classification with CNNs

This project implements a multi-output Convolutional Neural Network (CNN) that simultaneously classifies images from the **CIFAR-10** and **CIFAR-100** datasets. Developed during my internship at **Navodhitha Technologies**, it focuses on multi-task learning in image classification.

---

## 📌 Features

- 🧠 **Dual-output CNN** trained on CIFAR-10 & CIFAR-100
- 🎯 Achieved **>75% validation accuracy** on both outputs
- 🔍 Visualizations: training curves, confusion matrix, metrics
- 💾 Save/load model in HDF5 format (`.h5`)
- ⚙️ Modular and easy to extend for future research or deployment

---

## 🧪 Datasets Used

- **CIFAR-10**: 60,000 color images (32x32) in 10 classes
- **CIFAR-100**: 60,000 color images (32x32) in 100 classes
- Both datasets are automatically loaded via `tensorflow.keras.datasets`.

---

## 📁 Project Structure

```

image-classification-cnn/
├── model.py             # CNN model with shared base and dual output heads
├── train.py             # Handles training, saving, and plotting
├── evaluate.py          # Model evaluation and confusion matrix
├── utils.py             # Helper functions for plotting and preprocessing
├── cifar10\_cifar100.h5  # (Optional) Pretrained model file
├── requirements.txt     # Required packages
└── README.md            # Project documentation

````

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/yourusername/image-classification-cnn.git
cd image-classification-cnn
pip install -r requirements.txt
````

### 🏋️‍♂️ Training

```bash
python train.py
```

### 📈 Evaluation

```bash
python evaluate.py
```

---

## 📊 Results

| Output    | Accuracy (Validation) |
| --------- | --------------------- |
| CIFAR-10  | \~78%                 |
| CIFAR-100 | \~76%                 |

* 📉 Training loss and accuracy curves
* ✅ Confusion matrix for both outputs
* 🧠 Efficient multi-task learning with shared base layers

---

## 💡 Internship at Navodhitha Technologies

During my internship at **Navodhitha Technologies**, I had the opportunity to work on advanced computer vision tasks involving multi-label classification and CNN architectures.

**Key Learnings**:

* Hands-on experience with real-world CNN-based classification
* Developed and tuned multi-output neural networks
* Gained insights into performance analysis, optimization, and dataset preprocessing

---

## 📌 Future Enhancements

* 🔁 Add data augmentation (`ImageDataGenerator`)
* 🏗 Replace base CNN with pretrained models (e.g., ResNet, MobileNet)
* 📲 Deploy using Streamlit or Flask web UI
* 📦 Extend model to work with custom datasets

---

## 👨‍💻 Author

**Krishna Venugopal**
📍 India
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
🐱 [GitHub](https://github.com/yourusername)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

```

---

Let me know if you'd like help with:
- Creating `requirements.txt`
- Uploading pretrained `.h5` model
- Streamlit deployment
- Making the repo public-ready

I can also provide badges, visuals, or GitHub Actions for CI/CD if needed.
```
