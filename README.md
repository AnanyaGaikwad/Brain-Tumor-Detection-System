# NeuroScan AI – Brain Tumor Detection System

NeuroScan AI is an AI-powered medical image analysis system designed to detect brain tumors from MRI scans. The project combines advanced image processing techniques with machine learning to enhance MRI images, segment tumor regions, extract meaningful features, and classify scans as Tumor or Normal.

The system performs preprocessing, contrast enhancement, noise removal, image sharpening, tumor segmentation, and feature extraction using statistical and GLCM-based texture features. A Random Forest classifier is used for prediction, and the entire workflow is deployed through a Flask-based web application that provides real-time visualization and diagnosis support.

## Key Features
- MRI image enhancement and preprocessing
- Noise reduction using Median and Gaussian filtering
- Tumor segmentation and ROI detection
- Statistical and GLCM texture feature extraction
- Random Forest-based tumor classification
- Interactive Flask web dashboard
- Real-time prediction and visualization

## Tech Stack
- Python
- OpenCV
- NumPy
- Scikit-learn
- Flask
- Matplotlib
- HTML/CSS/JavaScript

## Results
- Accuracy: 94%
- Precision: 94%
- Recall: 94%
- F1-Score: 94%

This project aims to assist radiologists and healthcare professionals by providing faster, explainable, and automated brain tumor detection from MRI images.
