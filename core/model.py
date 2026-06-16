import joblib
import os
import numpy as np

# Path to the saved model relative to the app root
MODEL_PATH = "tumor_classifier.pkl"

def load_model():
    """
    Loads the trained model.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

def predict_tumor(features_dict, model):
    """
    Takes the extracted features dict and the loaded model,
    returns the prediction and confidence score.
    """
    if model is None:
        return "Model not found", 0.0
        
    # Convert dict to expected array format
    # Order must match the training script exactly
    feature_vector = [
        features_dict['area'],
        features_dict['mean_intensity'],
        features_dict['std_intensity'],
        features_dict.get('contrast', 0),
        features_dict.get('dissimilarity', 0),
        features_dict.get('homogeneity', 0),
        features_dict.get('energy', 0),
        features_dict.get('correlation', 0)
    ]
    
    # Reshape for single sample prediction
    X = np.array(feature_vector).reshape(1, -1)
    
    # Predict
    prediction_num = model.predict(X)[0]
    
    # Get probability/confidence
    try:
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities[prediction_num] * 100
    except:
        # If model doesn't support predict_proba
        confidence = 100.0
        
    result_text = "Abnormal (Tumor Detected)" if prediction_num == 1 else "Normal (No Tumor Detected)"
    
    return result_text, round(confidence, 2)
