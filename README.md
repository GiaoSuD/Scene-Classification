# 🧠 Scene Classification with ResNet50 + Web UI

A complete **image scene classification** project using a fine-tuned **ResNet50** model on the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). The project includes:

- ✅ A Flask-based API for inference  
- 🌐 A modern frontend for drag-and-drop predictions  
- 📦 Preprocessing, training notebook, and model saving

---

## 📁 Dataset

The model was trained using the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), which contains 25,000 images (size 150x150) across 6 scene classes:

- `building`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

---

## 🚀 Features

- **CNN Architecture**: Based on ResNet50, with modified classification head
- **Preprocessing**: Images resized to 224x224, normalized with ImageNet stats
- **Training**: Transfer learning with Dropout regularization for robustness
- **Deployment**:
  - `api_scene_prd.py`: REST API for prediction using Flask
  - `scene_classifier_frontend.html`: Interactive web UI with image/folder upload
- **Batch Inference** supported via `/predict_batch`

---

## 🧠 Model Pipeline

```text
[Input Image]
      ↓ Resize + Normalize
[ResNet50 Backbone (pretrained)]
      ↓
[Dropout → FC(256) → ReLU → Dropout → FC(6)]
      ↓
[Softmax Probabilities]
```

- **Transformations**:
  ```python
  transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  ```

---

## 💻 How to Run

### 1️⃣ Backend (Flask API)

```bash
# Install dependencies
pip install flask flask-cors torch torchvision pillow

# Run the API server
python api_scene_prd.py
```

- API will be available at: `http://127.0.0.1:5000`
- Main endpoint: `POST /predict` (accepts image file)

### 2️⃣ Frontend (HTML + JS)

Just open the file:

```
scene_classifier_frontend.html
```

- Supports drag & drop or browsing files
- Sends predictions to backend via `/predict`

> **Note**: Ensure the API is running locally before using the UI.

---

## 🧪 Example Request

```bash
curl -X POST -F "image=@example.jpg" http://127.0.0.1:5000/predict
```

Example JSON response:
```json
{
  "prediction": "forest",
  "confidence": 0.982,
  "filename": "example.jpg",
  "success": true
}
```

---

## 📂 Repository Structure

```
├── Scene_classifier_train.ipynb     # Training & saving model (ResNet50)
├── api_scene_prd.py                 # Flask backend API
├── scene_classifier_frontend.html   # Frontend web UI
├── best_model.pth                   # (You must add this separately)
├── README.md                        # You are here
```

---

## 📌 Notes

- This project assumes `best_model.pth` (trained weights) is available.
- You can retrain the model using `Scene_classifier_train.ipynb`.
- I may have changed this url of this project to deploy, pls check it correct or else it won't work
---

## 📄 License

This project is released under the MIT License.
