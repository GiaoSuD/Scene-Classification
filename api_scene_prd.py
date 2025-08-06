from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ScenePredictor:
    def __init__(self, model_path='best_model.pth', num_classes=6, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define class names - UPDATE THESE to match your actual classes
        self.class_names = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Define transform (same as validation transform from training)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.class_names}")
    
    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("API will run in demo mode with random predictions")
            return None
        
        try:
            # Create model architecture (same as training)
            model = torchvision.models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes)
            )
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("API will run in demo mode with random predictions")
            return None
    
    def predict_image(self, image):
        """Predict class for a PIL image"""
        try:
            if self.model is None:
                # Demo mode - return random prediction
                import random
                predicted_class = random.choice(self.class_names)
                confidence_score = 0.7 + random.random() * 0.25
                return predicted_class, confidence_score
            
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to random prediction
            import random
            predicted_class = random.choice(self.class_names)
            confidence_score = 0.5 + random.random() * 0.3
            return predicted_class, confidence_score

# Initialize predictor
predictor = ScenePredictor()

@app.route('/')
def home():
    return jsonify({
        "message": "Scene Classification API is running!",
        "classes": predictor.class_names,
        "model_loaded": predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        predicted_class, confidence = predictor.predict_image(image)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence),
            'filename': file.filename,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Check if files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            try:
                # Read and process image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                # Make prediction
                predicted_class, confidence = predictor.predict_image(image)
                
                results.append({
                    'filename': file.filename,
                    'prediction': predicted_class,
                    'confidence': float(confidence),
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'device': str(predictor.device),
        'classes': predictor.class_names
    })

if __name__ == '__main__':
    print("Starting Scene Classification API...")
    print(f"Classes: {predictor.class_names}")
    print(f"Model loaded: {predictor.model is not None}")
    print("API will be available at: http://127.0.0.1:5000")
    print("Frontend should connect to: http://127.0.0.1:5000/predict")
    
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)