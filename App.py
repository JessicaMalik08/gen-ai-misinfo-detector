from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import base64
import io
import warnings
from PIL import Image
import librosa
import numpy as np
import tensorflow as tf
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import lime
from lime import lime_text
from lime.lime_image import LimeImageExplainer
import google.generativeai as genai

warnings.filterwarnings('ignore')

# ------------------------
# Flask setup
# ------------------------
app = Flask(__name__)
CORS(app)

# Gemini client - FIXED: Use environment variable properly
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    print("[WARNING] GEMINI_API_KEY not found. Explanation features will be limited.")
    gemini_model = None

# ------------------------
# Explanation Engine (Gemini)
# ------------------------
class ExplanationLayerOSS:
    CONFIDENCE_MAP = {
        (0.0, 0.4): "low chance of being fake",
        (0.4, 0.7): "moderate chance of being fake",
        (0.7, 1.0): "high chance of being fake"
    }

    TECHNIQUE_LIBRARY = {
        "audio": {
            "robotic tone": "AI voice synthesis often produces slightly monotone or robotic speech.",
            "frequency artifacts": "Deepfake voices can leave behind unusual frequency spikes.",
            "unnatural pauses": "Synthetic voices sometimes pause in ways that don't match human breathing."
        },
        "text": {
            "emotional language": "Fake news often exaggerates emotions to manipulate the reader.",
            "clickbait": "Overly dramatic headlines are designed to grab attention, not inform.",
            "repetitive phrases": "AI-generated or fake content often repeats words unnaturally."
        },
        "visual": {
            "blurry edges": "AI-generated images often fail to render sharp edges.",
            "lighting mismatch": "Fake composites show inconsistent lighting between objects.",
            "distorted hands/faces": "Deepfakes sometimes struggle with human anatomy."
        }
    }

    def _confidence_label(self, score: float) -> str:
        for (low, high), label in self.CONFIDENCE_MAP.items():
            if low <= score <= high:
                return label
        return "uncertain"

    def _reason_parser(self, model_output):
        modality = model_output.get("modality")
        score = model_output.get("score", 0.0)
        features = model_output.get("features", [])

        feature_explanations = []
        for f in features:
            extra = self.TECHNIQUE_LIBRARY.get(modality, {}).get(f, "")
            feature_explanations.append(f"{f} ({extra})" if extra else f)

        return f"{modality.upper()} model flagged: {', '.join(feature_explanations)}. Confidence={score:.2f} ({self._confidence_label(score)})."

    def generate_explanation(self, model_outputs, mode="general"):
        if isinstance(model_outputs, dict):
            model_outputs = [model_outputs]

        reasons = [self._reason_parser(out) for out in model_outputs]
        combined_reasons = "\n".join(reasons)

        if not gemini_model:
            return {
                "verdict": "suspicious",
                "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
                "technical_reasons": reasons,
                "llm_explanation": "Gemini API not configured. Using fallback explanation."
            }

        try:
            prompt = f"""
            You are an AI misinformation educator.
            Evidence from detection models:
            {combined_reasons}

            TASK:
            - Translate this into a {mode}-level explanation
            - Cover:
              1. Why it may be misleading
              2. What visible signs a normal user could spot
              3. Practical steps to verify
              4. Educational tip to avoid similar tricks in future
            Output as clear structured JSON with keys:
            why_misleading, visible_signs, verification_steps, educational_tip, summary
            """

            response = gemini_model.generate_content(prompt)

            return {
                "verdict": "suspicious",
                "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
                "technical_reasons": reasons,
                "llm_explanation": response.text
            }
        except Exception as e:
            return {
                "verdict": "suspicious",
                "confidence_summary": [self._confidence_label(out.get("score", 0.0)) for out in model_outputs],
                "technical_reasons": reasons,
                "llm_explanation": f"Error generating explanation: {str(e)}"
            }

engine = ExplanationLayerOSS()

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        outputs = data.get("model_outputs", [])
        mode = data.get("mode", "general")
        result = engine.generate_explanation(outputs, mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------
# FIXED MODEL LOADING HELPERS
# ------------------------
def load_pickle_model(path):
    """Load scikit-learn models saved with pickle"""
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load pickle model {path}: {str(e)}")
    return None

def load_torch_model(path, model_class=None):
    """Load PyTorch models saved with torch.save()"""
    if os.path.exists(path):
        try:
            # Try loading as state dict first
            if model_class:
                model = model_class()
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                return model
            else:
                # Try loading as complete model
                return torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"[ERROR] Failed to load PyTorch model {path}: {str(e)}")
    return None

def load_keras_model(path):
    """Load Keras/TensorFlow models"""
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f"[ERROR] Failed to load Keras model {path}: {str(e)}")
            # Try loading as weights only
            try:
                print(f"[INFO] Attempting to load {path} as weights file...")
                # You would need to define your model architecture here
                # model = create_your_model_architecture()
                # model.load_weights(path)
                # return model
                return None
            except Exception as e2:
                print(f"[ERROR] Failed to load as weights: {str(e2)}")
    return None

# Create directories
os.makedirs("modeltext/checkpoints", exist_ok=True)
os.makedirs("visual", exist_ok=True)
os.makedirs("audio/checkpoints", exist_ok=True)

# FIXED: Load models with proper error handling
print("[INFO] Loading models...")

# Text model (likely scikit-learn)
text_model = load_pickle_model("modeltext/checkpoints/model.pkl")
text_vectorizer = load_pickle_model("modeltext/checkpoints/vectorizer.pkl")

# Visual model (problematic .h5 file)
visual_model = load_keras_model("visual/vit_deepfake_visual.h5")

# Audio model (might be PyTorch based on the error)
audio_model = load_pickle_model("audio/checkpoints/best_model.pkl")
if audio_model is None:
    # Try loading as PyTorch model
    audio_model = load_torch_model("audio/checkpoints/best_model.pkl")

# Fallback vectorizer
if text_vectorizer is None:
    print("[INFO] Creating fallback TF-IDF vectorizer")
    text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

print(f"[INFO] Models loaded - Text: {text_model is not None}, Visual: {visual_model is not None}, Audio: {audio_model is not None}")

# ------------------------
# EXPLANATION HELPERS
# ------------------------
def generate_text_explanation(text, model, vectorizer, prediction_prob):
    try:
        if model and vectorizer and hasattr(model, "predict_proba"):
            explainer = lime_text.LimeTextExplainer(class_names=["Real", "Fake"])
            def predict_proba_fn(texts):
                features = vectorizer.transform(texts)
                return model.predict_proba(features)
            exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
            explanation_data = []
            for word, importance in exp.as_list():
                explanation_data.append({
                    "word": word,
                    "importance": float(importance),
                    "contribution": "supports_fake" if importance > 0 else "supports_real"
                })
            return {
                "method": "LIME",
                "top_features": sorted(explanation_data, key=lambda x: abs(x['importance']), reverse=True)[:10]
            }
        else:
            return {"method": "Statistical Analysis", "note": "Model not available - using fallback"}
    except Exception as e:
        return {"error": f"Text explanation error: {str(e)}"}

def generate_image_explanation(image_array, model, prediction_prob):
    try:
        img = image_array[0] if len(image_array.shape) == 4 else image_array
        h, w, c = img.shape
        return {
            "method": "Statistical Analysis",
            "dimensions": f"{w}x{h}x{c}",
            "brightness_mean": float(np.mean(img)),
            "model_available": model is not None
        }
    except Exception as e:
        return {"error": f"Image explanation error: {str(e)}"}

def generate_audio_explanation(audio_features, model, audio_duration, prediction_prob):
    try:
        return {
            "method": "Audio Feature Analysis",
            "duration_seconds": audio_duration,
            "feature_count": len(audio_features[0]) if len(audio_features.shape) > 1 else len(audio_features),
            "model_available": model is not None
        }
    except Exception as e:
        return {"error": f"Audio explanation error: {str(e)}"}

# ------------------------
# ROUTES
# ------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Deepfake Detection API", "version": "1.0"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "text": text_model is not None,
            "text_vectorizer": text_vectorizer is not None,
            "visual": visual_model is not None,
            "audio": audio_model is not None
        },
        "gemini_configured": gemini_model is not None
    })

@app.route("/check/text", methods=["POST"])
def check_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text = data.get("content", "")
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
            
        pred, proba = 0, 0.5
        if text_model and text_vectorizer:
            try:
                features = text_vectorizer.transform([text])
                pred = text_model.predict(features)[0]
                proba = text_model.predict_proba(features)[0].max()
            except Exception as e:
                print(f"[ERROR] Text prediction failed: {str(e)}")
                
        explanation = generate_text_explanation(text, text_model, text_vectorizer, float(proba))
        return jsonify({
            "type": "text", 
            "verdict": "fake" if pred == 1 else "real", 
            "confidence": float(proba), 
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check/image", methods=["POST"])
def check_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        img_b64 = data.get("image", "")
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400
            
        # Handle data URL format
        if img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",")[1]
            
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            img_resized = img.resize((224, 224))
            img_arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
            
        pred_class, confidence = 0, 0.5
        if visual_model:
            try:
                pred = visual_model.predict(img_arr)[0]
                confidence = float(np.max(pred))
                pred_class = int(np.argmax(pred))
            except Exception as e:
                print(f"[ERROR] Image prediction failed: {str(e)}")
                
        explanation = generate_image_explanation(img_arr, visual_model, confidence)
        return jsonify({
            "type": "image", 
            "verdict": "fake" if pred_class == 1 else "real", 
            "confidence": confidence, 
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check/audio", methods=["POST"])
def check_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
            
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
            
        # Secure filename handling
        import uuid
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = f"/tmp/{temp_filename}"
        
        try:
            file.save(temp_path)
            y, sr = librosa.load(temp_path, sr=16000)
            os.remove(temp_path)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            
            pred, proba = 0, 0.5
            if audio_model:
                try:
                    if hasattr(audio_model, 'predict'):
                        pred = audio_model.predict(features)[0]
                        if hasattr(audio_model, 'predict_proba'):
                            proba = audio_model.predict_proba(features)[0].max()
                    else:
                        # Handle PyTorch models
                        with torch.no_grad():
                            features_tensor = torch.FloatTensor(features)
                            output = audio_model(features_tensor)
                            pred = torch.argmax(output, dim=1).item()
                            proba = torch.softmax(output, dim=1).max().item()
                except Exception as e:
                    print(f"[ERROR] Audio prediction failed: {str(e)}")
                    
            explanation = generate_audio_explanation(features, audio_model, float(len(y)/sr), float(proba))
            return jsonify({
                "type": "audio", 
                "verdict": "fake" if pred == 1 else "real", 
                "confidence": float(proba), 
                "duration": float(len(y)/sr), 
                "explanation": explanation
            })
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
    