import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import from our core package
from core.image_processing import load_and_preprocess, remove_noise, sharpen_image, segment_image, extract_features

def load_dataset(dataset_path):
    """
    Loads images from 'yes' and 'no' folders, extracts features, and creates dataset.
    """
    features_list = []
    labels_list = []
    
    classes = {'no': 0, 'yes': 1}
    
    print("Extracting features... this may take a moment.")
    
    for class_name, label in classes.items():
        class_dir = os.path.join(dataset_path, class_name)
        # Handle different image extensions
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
            image_paths.extend(glob.glob(os.path.join(class_dir, ext)))
            
        for img_path in image_paths:
            try:
                # 1. Preprocess
                img = load_and_preprocess(img_path)
                
                # 2. Process to get mask
                noise_rm_img = remove_noise(img)
                sharp_img = sharpen_image(noise_rm_img)
                mask, _ = segment_image(img) # Use original un-enhanced image
                
                # 3. Extract Features
                features_dict = extract_features(img, mask)
                
                # Convert dict to array of features
                # Order: area, mean_intensity, std_intensity, contrast, dissimilarity, homogeneity, energy, correlation
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
                
                features_list.append(feature_vector)
                labels_list.append(label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(features_list), np.array(labels_list)

def train_and_save_model(dataset_path, model_save_path):
    X, y = load_dataset(dataset_path)
    
    if len(X) == 0:
        print("Error: No valid data found. Please check the dataset path.")
        return
        
    print(f"Successfully loaded {len(X)} samples.")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train a more powerful model (Random Forest)
    print("Training Random Forest Model...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Using a Pipeline ensures that scaling is applied consistently during prediction
    svm_classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    svm_classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))
    
    # Generate and save Model Performance Graphs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
    
    os.makedirs('static/assets', exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Tumor'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    plt.title('Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('static/assets/confusion_matrix.png')
    plt.close(fig)
    
    # 2. Metrics Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'])
    plt.ylim(0, 1.1)
    plt.title('Overall Model Performance Metrics')
    plt.ylabel('Score (0.0 to 1.0)')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('static/assets/metrics_bar.png')
    plt.close('all')
    
    # Save the trained model
    joblib.dump(svm_classifier, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    # The path verified with the user
    DATASET_DIR = "archive (3)/brain_tumor_dataset/"
    MODEL_PATH = "tumor_classifier.pkl"
    
    if not os.path.exists(DATASET_DIR):
        print(f"Could not find dataset directory at {DATASET_DIR}")
        print("Please ensure the dataset is unzipped correctly.")
    else:
        train_and_save_model(DATASET_DIR, MODEL_PATH)
