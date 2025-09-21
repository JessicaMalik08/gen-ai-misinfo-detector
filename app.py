from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import base64
import io
from PIL import Image
import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Explanation layer imports
import lime
from lime import lime_text
from lime.lime_image import LimeImageExplainer

app = Flask(__name__)
CORS(app)  # allow Chrome extension to call

# Add cache control to prevent browser caching issues
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Helper function to load pickle models safely
def load_pickle_model(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"[SUCCESS] Loaded pickle model: {path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load pickle model {path}: {str(e)}")
            return None
    else:
        print(f"[WARNING] Pickle model not found: {path}")
        return None

# Helper function to load Keras model safely
def load_keras_model(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print(f"[SUCCESS] Loaded Keras model: {path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load Keras model {path}: {str(e)}")
            return None
    else:
        print(f"[WARNING] Keras model not found: {path}")
        return None

# Create model directories if they don't exist
os.makedirs("modeltext/checkpoints", exist_ok=True)
os.makedirs("visual", exist_ok=True)
os.makedirs("audio/checkpoints", exist_ok=True)
os.makedirs("explanations", exist_ok=True)

# EXPLANATION LAYER FUNCTIONS

def generate_text_explanation(text, model, vectorizer, prediction_prob):
    """Generate explanation for text predictions using LIME or fallback analysis"""
    try:
        # Only use LIME if we have a proper model with predict_proba
        if model is not None and vectorizer is not None and hasattr(model, 'predict_proba') and hasattr(vectorizer, 'transform'):
            # Create LIME explainer for text
            explainer = lime_text.LimeTextExplainer(class_names=['Real', 'Fake'])
            
            # Define prediction function for LIME
            def predict_proba_fn(texts):
                features = vectorizer.transform(texts)
                return model.predict_proba(features)
            
            # Generate explanation
            exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)
            
            # Extract explanation data
            explanation_data = []
            for word, importance in exp.as_list():
                explanation_data.append({
                    "word": word,
                    "importance": float(importance),
                    "contribution": "supports_fake" if importance > 0 else "supports_real"
                })
            
            # Get top suspicious and authentic words
            sorted_exp = sorted(explanation_data, key=lambda x: abs(x['importance']), reverse=True)
            
            return {
                "explanation_available": True,
                "method": "LIME",
                "top_features": sorted_exp[:10],
                "fake_indicators": [item for item in sorted_exp if item['contribution'] == 'supports_fake'][:5],
                "real_indicators": [item for item in sorted_exp if item['contribution'] == 'supports_real'][:5],
                "explanation_summary": f"LIME analysis of {len(explanation_data)} words shows decision factors"
            }
        else:
            # Fallback analysis when models aren't available
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0
            
            return {
                "explanation_available": True,
                "method": "Statistical Analysis",
                "fallback_analysis": {
                    "text_length": char_count,
                    "word_count": word_count,
                    "average_word_length": round(avg_word_length, 2),
                    "analysis_note": "Model unavailable - providing basic text statistics"
                },
                "explanation_summary": f"Analyzed text with {word_count} words ({char_count} characters)"
            }
        
    except Exception as e:
        print(f"Text explanation error: {str(e)}")
        return {
            "explanation_available": False,
            "error": "Could not generate text explanation",
            "fallback_analysis": {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_note": "Error during explanation generation"
            }
        }

def generate_image_explanation(image_array, model, prediction_prob):
    """Generate explanation for image predictions using LIME or fallback analysis"""
    try:
        # Only use LIME if we have a proper model 
        if model is not None and hasattr(model, 'predict'):
            # For now, provide statistical analysis instead of LIME due to complexity
            # LIME for images requires careful setup and can be unreliable
            pass
        
        # Provide statistical analysis of the image
        if len(image_array.shape) == 4:
            img = image_array[0]  # Remove batch dimension
        else:
            img = image_array
            
        # Basic image statistics
        height, width, channels = img.shape
        mean_brightness = float(np.mean(img))
        std_brightness = float(np.std(img))
        
        # Color distribution analysis
        red_channel_mean = float(np.mean(img[:, :, 0])) if channels >= 3 else 0
        green_channel_mean = float(np.mean(img[:, :, 1])) if channels >= 3 else 0
        blue_channel_mean = float(np.mean(img[:, :, 2])) if channels >= 3 else 0
        
        return {
            "explanation_available": True,
            "method": "Statistical Analysis",
            "image_analysis": {
                "dimensions": f"{width}x{height}x{channels}",
                "brightness_stats": {
                    "mean": round(mean_brightness, 3),
                    "std": round(std_brightness, 3)
                },
                "color_analysis": {
                    "red_mean": round(red_channel_mean, 3),
                    "green_mean": round(green_channel_mean, 3),
                    "blue_mean": round(blue_channel_mean, 3)
                }
            },
            "interpretation": {
                "verdict": "Analysis based on image statistical properties",
                "analysis_summary": f"Processed {width}x{height} image with {channels} channels",
                "confidence_factors": [
                    "Image brightness patterns",
                    "Color distribution analysis",
                    "Pixel intensity variations"
                ]
            },
            "model_status": "available" if model is not None else "unavailable"
        }
        
    except Exception as e:
        print(f"Image explanation error: {str(e)}")
        return {
            "explanation_available": False,
            "error": "Could not generate image explanation",
            "fallback_analysis": {
                "image_dimensions": f"{image_array.shape}",
                "analysis_note": "Error during image explanation generation"
            }
        }

def generate_audio_explanation(audio_features, model, audio_duration, prediction_prob):
    """Generate explanation for audio predictions"""
    try:
        # For audio, we'll analyze feature importance and temporal patterns
        if hasattr(model, 'feature_importances_'):
            # Tree-based model with feature importance
            feature_names = [f"MFCC_{i+1}" for i in range(len(audio_features[0]))]
            importances = model.feature_importances_
            
            # Get top important features
            feature_importance_pairs = list(zip(feature_names, importances))
            sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
            
            return {
                "explanation_available": True,
                "method": "Feature Importance Analysis",
                "audio_analysis": {
                    "duration_seconds": audio_duration,
                    "features_analyzed": len(feature_names),
                    "top_suspicious_features": sorted_features[:5],
                    "feature_distribution": {
                        "mean_importance": float(np.mean(importances)),
                        "max_importance": float(np.max(importances)),
                        "feature_variance": float(np.var(importances))
                    }
                },
                "interpretation": {
                    "verdict": f"Model decision based on {len(feature_names)} audio features",
                    "key_indicators": [f"{name}: {importance:.4f}" for name, importance in sorted_features[:3]],
                    "analysis_summary": f"Top {len(sorted_features[:5])} features contributed most to the decision"
                }
            }
        else:
            # Neural network or other model - provide general analysis
            feature_stats = {
                "mean_values": audio_features[0].tolist()[:10],  # Show first 10 MFCC values
                "feature_range": {
                    "min": float(np.min(audio_features)),
                    "max": float(np.max(audio_features)),
                    "std": float(np.std(audio_features))
                }
            }
            
            return {
                "explanation_available": True,
                "method": "Audio Feature Analysis",
                "audio_analysis": {
                    "duration_seconds": audio_duration,
                    "mfcc_features": len(audio_features[0]),
                    "feature_statistics": feature_stats
                },
                "interpretation": {
                    "verdict": "Analysis based on Mel-frequency cepstral coefficients (MFCC)",
                    "analysis_summary": f"Processed {audio_duration:.2f} seconds of audio with {len(audio_features[0])} features",
                    "confidence_factors": [
                        "Audio frequency patterns",
                        "Spectral characteristics", 
                        "Temporal audio features"
                    ]
                }
            }
            
    except Exception as e:
        print(f"Audio explanation error: {str(e)}")
        return {
            "explanation_available": False,
            "error": "Could not generate audio explanation",
            "fallback_analysis": {
                "duration_seconds": audio_duration,
                "analysis_note": "Unable to generate detailed explanations due to missing models"
            }
        }

# Initialize models and vectorizer
text_model = None
text_vectorizer = None
visual_model = None
audio_model = None

# Load models based on actual project structure
try:
    print("Attempting to load models...")
    
    # Text model from modeltext folder
    text_model = load_pickle_model("modeltext/checkpoints/model.pkl")
    # Load separate vectorizer file
    text_vectorizer = load_pickle_model("modeltext/checkpoints/vectorizer.pkl")
    
    # If no vectorizer is found, create a default one for demonstration
    if text_vectorizer is None:
        print("[INFO] No vectorizer found, creating default TF-IDF vectorizer")
        text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        # Note: This vectorizer would need to be fitted on training data in practice
        # For now, we'll handle this in the prediction function

    # Visual/Image model from visual folder (Keras model)
    visual_model = load_keras_model("visual/vit_deepfake_visual.h5")

    # Audio model (you'll need to add this to your structure)
    audio_model = load_pickle_model("audio/checkpoints/best_model.pkl")

    print("Model loading summary:")
    print(f"Text model loaded: {text_model is not None}")
    print(f"Text vectorizer loaded: {text_vectorizer is not None}")
    print(f"Visual model loaded: {visual_model is not None}")
    print(f"Audio model loaded: {audio_model is not None}")

except Exception as e:
    print(f"Error loading models: {e}")
    text_model = visual_model = audio_model = None
    text_vectorizer = None

# ROUTES

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Deepfake Detection API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/check/text": "POST - Analyze text for deepfake content",
            "/check/image": "POST - Analyze image for deepfake content", 
            "/check/audio": "POST - Analyze audio for deepfake content"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify API status"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "text": text_model is not None,
            "text_vectorizer": text_vectorizer is not None,
            "visual": visual_model is not None,
            "audio": audio_model is not None
        },
        "timestamp": str(np.datetime64('now', 'D'))
    })

@app.route("/check/text", methods=["POST"])
def check_text():
    """Analyze text content for deepfake detection"""

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text = data.get("content", "")
        if not text or not text.strip():
            return jsonify({"error": "No text content provided"}), 400

        # Text preprocessing
        text = str(text).strip()
        
        # Check if model is available for prediction
        if text_model is not None:
            # Try different prediction approaches
            if hasattr(text_model, 'predict_text'):
                result = text_model.predict_text(text)
                pred = result['prediction']
                proba = result['confidence']
            elif hasattr(text_model, 'predict') and hasattr(text_model, 'predict_proba'):
                try:
                    pred = text_model.predict([text])[0]
                    proba = text_model.predict_proba([text])[0].max()
                except Exception as e:
                    # If direct prediction fails, try with vectorizer
                    if text_vectorizer is not None and hasattr(text_vectorizer, 'transform'):
                        features = text_vectorizer.transform([text])
                        pred = text_model.predict(features)[0]
                        proba = text_model.predict_proba(features)[0].max()
                    else:
                        raise e
            elif text_vectorizer is not None and hasattr(text_vectorizer, 'transform'):
                features = text_vectorizer.transform([text])
                pred = text_model.predict(features)[0]
                proba = text_model.predict_proba(features)[0].max()
            else:
                # Fallback when model exists but no proper vectorizer
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                features = vectorizer.fit_transform([text])
                pred = text_model.predict(features)[0]
                proba = text_model.predict_proba(features)[0].max()
            
            model_available = True
        else:
            # Model not available - provide fallback analysis
            pred = 0  # Default to "real" 
            proba = 0.5  # Neutral confidence
            model_available = False

        # Generate explanation (works with or without model)
        explanation = generate_text_explanation(text, text_model, text_vectorizer, float(proba))
        
        return jsonify({
            "type": "text",
            "verdict": "fake" if pred == 1 else "real",
            "confidence": float(proba),
            "content_length": len(text),
            "model_available": model_available,
            "processed_successfully": True,
            "explanation": explanation
        })
        
    except Exception as e:
        print(f"Text prediction error: {str(e)}")
        return jsonify({"error": f"Text prediction error: {str(e)}"}), 500

@app.route("/check/image", methods=["POST"])
def check_image():
    """Analyze image content for deepfake detection"""

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        img_b64 = data.get("image", "")
        if not img_b64:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if img_b64.startswith('data:image'):
                img_b64 = img_b64.split(',')[1]
            
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Resize image for ViT model (assuming 224x224 input size)
        img_resized = img.resize((224, 224))
        img_arr = np.array(img_resized).astype(np.float32) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

        # Check if model is available for prediction
        if visual_model is not None:
            # Make prediction
            pred = visual_model.predict(img_arr)[0]
            confidence = float(np.max(pred))
            predicted_class = int(np.argmax(pred))
            model_available = True
        else:
            # Model not available - provide fallback analysis
            predicted_class = 0  # Default to "real"
            confidence = 0.5  # Neutral confidence
            model_available = False

        # Generate explanation (works with or without model)
        explanation = generate_image_explanation(img_arr, visual_model, confidence)
        
        return jsonify({
            "type": "image",
            "verdict": "fake" if predicted_class == 1 else "real",
            "confidence": confidence,
            "image_size": f"{img.size[0]}x{img.size[1]}",
            "model_available": model_available,
            "processed_successfully": True,
            "explanation": explanation
        })

    except Exception as e:
        print(f"Image prediction error: {str(e)}")
        return jsonify({"error": f"Image prediction error: {str(e)}"}), 500

@app.route("/check/audio", methods=["POST"])
def check_audio():
    """Analyze audio content for deepfake detection"""

    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        file = request.files["file"]
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Load audio file with librosa
        try:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            # Load audio with librosa
            y, sr = librosa.load(temp_path, sr=16000)
            
            # Clean up temporary file
            os.remove(temp_path)
            
        except Exception as e:
            return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 400

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Check if model is available for prediction
        if audio_model is not None:
            # Make prediction
            pred = audio_model.predict(features)[0]
            proba = audio_model.predict_proba(features)[0].max()
            model_available = True
        else:
            # Model not available - provide fallback analysis
            pred = 0  # Default to "real"
            proba = 0.5  # Neutral confidence
            model_available = False

        # Generate explanation (works with or without model)
        explanation = generate_audio_explanation(features, audio_model, float(len(y) / sr), float(proba))
        
        return jsonify({
            "type": "audio",
            "verdict": "fake" if pred == 1 else "real", 
            "confidence": float(proba),
            "audio_duration": float(len(y) / sr),
            "sample_rate": int(sr),
            "model_available": model_available,
            "processed_successfully": True,
            "explanation": explanation
        })

    except Exception as e:
        print(f"Audio prediction error: {str(e)}")
        return jsonify({"error": f"Audio prediction error: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

# RUN APP
if __name__ == "__main__":
    print("Starting Deepfake Detection API...")
    print("Available endpoints:")
    print("- GET  /        : API information")
    print("- GET  /health  : Health check")
    print("- POST /check/text  : Text analysis")
    print("- POST /check/image : Image analysis")
    print("- POST /check/audio : Audio analysis")
    
    # Cloud Run uses PORT environment variable, default to 5000 for Replit, 8080 for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


    

