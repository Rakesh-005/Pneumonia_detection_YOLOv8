# 🩺 Pneumonia Detection from Chest X-Rays Using YOLOv8

This project aims to detect **pneumonia in chest X-ray images** using the YOLOv8 object detection model. The system was trained and evaluated on both the **Roboflow dataset** and the **Kaggle Chest X-ray Pneumonia dataset**, achieving high accuracy and strong generalization performance. It includes a real-time **web interface** for image-based diagnosis.

---

## 🧠 Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)**
- **OpenCV**, **Albumentations**
- **Flask** (for web interface)
- **Matplotlib, Seaborn, Scikit-learn** (for analysis)
- **Google Colab / Jupyter Notebooks**

---

## 📊 Model Performance

### ✅ Roboflow Dataset (Internal)
- **Accuracy:** 97.14%
- **F1 Score:** 98.50%
- **mAP@50:** 98.67%

### 🔁 Kaggle Dataset (External Testing)
- **Accuracy:** ~90.65%
- **Precision:** 95.57%
- **F1 Score:** 93.4%
- **Recall:** ~91.48%

<details>
<summary>📈 See Evaluation Visuals</summary>

- Confusion Matrix  
  ![Confusion Matrix](confusion%20matrix.png)

- Confusion Matrix (Kaggle)  
  ![Confusion Matrix Kaggle](confusion%20matrix%20kaggle'.png)

- ROC Curve  
  ![ROC](ROC.jpeg)

- Precision-Recall Curve  
  ![PR Curve](PR_curve%20(1).png)

- YOLOv8 Architecture  
  ![Architecture](yolo%20arch.png)

</details>

---

## 📁 Project Structure

```
├── app.py                          # Flask web interface
├── requirements.txt               # Dependencies
├── README.md
├── LICENSE
├── Notebooks/
│   ├── roboflow-2nd-model-training.ipynb
│   └── chest-x-ray-testing.ipynb
├── Result Images/
│   ├── op_norm.png
│   ├── op_pneu.png
│   └── val_batch1_pred.jpg ...
├── Model Performance Visuals/
│   ├── confusion matrix.png
│   ├── ROC.jpeg ...
├── Sample Images/
│   ├── IM-0001-0001.jpeg
│   └── person63_virus_121.jpeg ...
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/pneumonia-yolov8.git
cd pneumonia-yolov8
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web Interface

```bash
python app.py
```

> 📍 Upload a chest X-ray image, and the model will classify it as **Normal** or **Pneumonia** with bounding box visualization.
---
## 🔬 Sample Predictions

| Normal | Pneumonia |
|--------|-----------|
| ![Normal](Result%20Images/op%20norm.png) | ![Pneumonia](Result%20Images/op%20pneu.png) |

---



## 📚 Dataset Used

- **Roboflow Custom Dataset** – 3,000 images (70% train, 20% val, 10% test)
- **Kaggle Chest X-ray Dataset** – External testing  
  📦 [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🧠 How It Works

1. **Preprocessing & Augmentation**: Image resizing, normalization, augmentations via Albumentations
2. **Model Training**: YOLOv8m trained on annotated chest X-rays using Ultralytics
3. **Evaluation**: Confusion matrix, ROC, Precision-Recall, External validation
4. **Deployment**: Flask interface for real-time diagnosis

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Contributing

Pull requests are welcome! If you'd like to improve this project, feel free to fork and submit changes.

---

## 👨‍💻 Author

**Rakesh Sarma Ponukupati**  
📧 rakesh20050618@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rakesh-sarma-ponukupati-6b3512259/) | [GitHub](https://github.com/Rakesh-005)

---
