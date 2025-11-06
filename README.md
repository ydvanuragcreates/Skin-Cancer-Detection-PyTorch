# 🩺 Skin Cancer Detection (ISIC 2020) using PyTorch & EfficientNet

This project is a deep learning solution for classifying skin lesions as benign (non-cancerous) or malignant (cancerous) using the ISIC 2020 dataset. The primary challenge of this dataset is its extreme class imbalance (~98% benign vs. 2% malignant), which this project addresses using a weighted loss strategy.

### Demo: Live Prediction on Validation Data
Here are some examples of the model's predictions on unseen validation images:

![Prediction 1](https://drive.google.com/file/d/12R17kdOvD5RSvZDd0U2VjwI6oIsj_iuO/view?usp=drive_link)


---

## 🎯 The Challenge: Extreme Class Imbalance

The ISIC 2020 dataset contains 33,126 training images, but the classes are not equal:
* **Benign (Normal):** 32,542 images (98.2%)
* **Malignant (Cancer):** 584 images (1.8%)

This means a "dumb" model that always guesses "Benign" would achieve 98.2% accuracy. Therefore, "accuracy" is a useless metric. The true goal is to correctly identify the rare malignant cases (i.e., maximize **Recall**).

---

## 💡 My Solution

To solve this, I implemented a pipeline in PyTorch with a focus on handling the imbalance.

1.  **Model:** **EfficientNetB0** (pre-trained on ImageNet) for Transfer Learning.
2.  **Framework:** PyTorch
3.  **Key Technique (Handling Imbalance):** I used a **Weighted Loss Function** (`BCEWithLogitsLoss`). A `pos_weight` of **55.7** was calculated (`Negatives / Positives`), forcing the model to treat a single malignant error as 55.7 times worse than a benign error.
4.  **Key Metric:** I optimized for **Validation AUC (Area Under the Curve)**, which is the standard metric for imbalanced classification, instead of accuracy.
5.  **Optimization:** Trained using **Automatic Mixed Precision (AMP)** for faster (float16) training on the T4 GPU.

---

## 📈 Final Results

The model was trained for 10 epochs, with the best model (based on `Val AUC`) saved automatically.

* **Best Validation AUC:** **0.8523 (85.2%)**
* **Malignant Class Recall:** **0.91 (91%)**

### ROC Curve
The final AUC of 0.8523 shows a strong ability to distinguish between the two classes (where 0.5 is random guessing).

![ROC Curve](https://github.com/ydvanuragcreates/Skin-Cancer-Detection-PyTorch/blob/main/ROC_Curve.png)

### Confusion Matrix & Classification Report
The results show our strategy was a success. We achieved our goal of **High Recall (91%)**, meaning the model successfully found 91% (107 out of 117) of the *actual* cancer cases in the validation set. This came at the necessary trade-off of low precision (many false alarms), which is the correct and safe strategy for a medical screening tool.

![Classification Report](https://github.com/ydvanuragcreates/Skin-Cancer-Detection-PyTorch/blob/main/classification_report.png)
![Confusion Matrix](https://github.com/ydvanuragcreates/Skin-Cancer-Detection-PyTorch/blob/main/confusion_matrix.png)

---

## 🚀 How to Use

1.  Download the repository.
2.  Download the trained model weights: [`best_pytorch_model.pth`](https://github.com/ydvanuragcreates/Skin-Cancer-Detection-PyTorch/blob/main/best_pytorch_model.pth)
3.  Install the required libraries:
    ```bash
    pip install timm torchmetrics torch torchvision pandas
    ```
4.  Run the prediction script (see the `.ipynb` file for the full prediction code).

---

## 📜 Report

For a detailed analysis, please see the full [Project_Report.pdf](https://github.com/ydvanuragcreates/Skin-Cancer-Detection-PyTorch/blob/main/Deep%20Learning%20for%20Malignant%20Melanoma%20Detection%20on%20the%20ISIC%202020%20Dataset.pdf).
