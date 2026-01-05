
# 🚗 Vehicle Damage Detection & Classification

An end-to-end **deep learning–based vehicle damage classification system** that detects and categorizes car damage from images into **front/rear and damage severity classes**.  
The project uses a **ResNet-50 transfer learning architecture**, deployed via a **Streamlit web interface** for real-time inference.

---

## 🔍 Problem Statement

Manual vehicle damage assessment is:

- Time-consuming  
- Subjective  
- Expensive at scale (insurance, fleet management, resale platforms)

This project automates damage assessment by classifying vehicle images into the following categories:

- **Front Breakage**
- **Front Crushed**
- **Front Normal**
- **Rear Breakage**
- **Rear Crushed**
- **Rear Normal**

---

## 🧠 Model Architecture

- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Fine-Tuning Strategy**
  - Backbone layers frozen
  - Final ResNet block (Layer4) unfrozen
- **Classifier Head**
  - Dropout (0.2)
  - Fully connected layer with 6 output classes
- **Loss Function:** Cross-Entropy Loss
- **Framework:** PyTorch

The model architecture and inference pipeline are implemented in `model_helper.py`.

---

## 📊 Model Performance

### Confusion Matrix

### Confusion Matrix

The confusion matrix shows strong diagonal dominance, indicating **high classification accuracy**.

Key observations:
- Front damage classes are well separated
- Minor confusion between **Rear Crushed** and **Rear Normal**, which is expected due to visual similarity

![Confusion Matrix](Confusion%20Matrix.png) Normal**, which is expected due to visual similarity

---

## 🖼️ Sample Predictions

### Front Damage Example

**Predicted Class:** `Front Crushed`  
The model correctly identifies visible deformation around the headlamp and front fender area.

![Front Crushed](Front%20Crushed.png)

### Rear Damage Example

**Predicted Class:** `Rear Breakage`  
The model detects structural damage near the rear bumper and tail lamp region.

![Rear Breakage](Rear%20Breakage.png)


---

## 🌐 Web Application (Streamlit)

A lightweight Streamlit web application enables real-time predictions.

### Features
- Image upload (JPG / PNG)
- Image preview
- Instant damage classification result

The UI and inference integration are implemented in `app.py`.

---

## 📁 Project Structure

```
├── app.py
├── model_helper.py
├── model/
│   └── saved_model.pth
├── requirements.txt
├── Confusion Matrix.png
├── Front Crushed.png
├── Rear Breakage.png
└── README.md
```

---

## ⚙️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-damage-detection.git
cd vehicle-damage-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🚀 Use Cases

- Insurance claim automation
- Used-car resale platforms
- Fleet damage inspection
- Vehicle rental verification
- AI-assisted vehicle audits

---

## 📜 License

MIT License
