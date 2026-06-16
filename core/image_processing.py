import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops

def load_and_preprocess(image_path, size=(256, 256)):
    """
    Loads an image, converts to grayscale, and resizes it.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Resize image
    img_resized = cv2.resize(img, size)
    
    # Normalization is often useful, but for CV2 functions 0-255 uint8 is standard.
    # We will keep it as uint8 for standard image processing.
    return img_resized

def enhance_image(img):
    """
    Applies Histogram Equalization and Contrast Stretching.
    Returns the enhanced image.
    """
    # 1. Histogram Equalization
    eq_img = cv2.equalizeHist(img)
    
    # 2. Contrast Stretching (Min-Max normalization)
    min_val, max_val, _, _ = cv2.minMaxLoc(eq_img)
    if max_val > min_val:
        stretched_img = np.uint8(255 * ((eq_img - min_val) / (max_val - min_val)))
    else:
        stretched_img = eq_img
        
    return stretched_img

def remove_noise(img):
    """
    Applies Median Filter and Gaussian Blur for noise removal.
    """
    # 1. Median Filter (good for salt and pepper noise)
    median_filtered = cv2.medianBlur(img, 3)
    
    # 2. Gaussian Blur (smooths overall)
    blurred = cv2.GaussianBlur(median_filtered, (5, 5), 0)
    
    return blurred

def sharpen_image(img):
    """
    Applies sharpening using Laplacian and High-pass filters.
    """
    # High-pass filter approach using unsharp masking
    # Blurred image is already smooth. 
    # Unsharp mask: Sharpened = Original + (Original - Blurred) * amount
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    unsharp_image = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
    
    return unsharp_image

def segment_image(img):
    """
    Applies Thresholding and Morphological Operations to highlight suspicious regions.
    Returns the binary mask and the edges.
    """
    # 1. High Thresholding (Tumors are typically hyper-intense/bright)
    # Using a fixed high threshold is often better than Otsu for finding anomalies
    _, binary_mask = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological opening to remove small noise (like pieces of skull)
    kernel = np.ones((5,5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Edge Detection (Canny)
    edges = cv2.Canny(img, 100, 200)
    
    # Highlight region (Overlay edges on original or return mask)
    # For visualization, we might want to return the mask to find contours later
    return binary_mask, edges

def extract_features(img, mask):
    """
    Extracts statistical and texture features from the segmented region.
    """
    features = {}
    
    # 1. Area of segmented region (number of white pixels in mask)
    area = np.sum(mask == 255)
    features['area'] = area
    
    # Apply mask to image to get pixel values of the tumor region
    tumor_pixels = img[mask == 255]
    
    # 2. Mean intensity
    mean_intensity = np.mean(tumor_pixels) if len(tumor_pixels) > 0 else 0
    features['mean_intensity'] = mean_intensity
    
    # 3. Standard deviation
    std_intensity = np.std(tumor_pixels) if len(tumor_pixels) > 0 else 0
    features['std_intensity'] = std_intensity
    
    # 4. Texture features (GLCM) - optional but good for ML
    # Using a reduced bin size (e.g., 256 -> 32) speeds up GLCM
    img_binned = (img / 8).astype(np.uint8) 
    glcm = graycomatrix(img_binned, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    
    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
    features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    features['energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    
    return features

def generate_histograms(original_img, enhanced_img, save_path_prefix):
    """
    Generates and saves histograms for the original and enhanced images.
    Returns the file paths to the saved histograms.
    """
    orig_hist_path = f"{save_path_prefix}_hist_orig.png"
    enh_hist_path = f"{save_path_prefix}_hist_enh.png"
    
    # Original Histogram
    plt.figure(figsize=(5, 4))
    plt.hist(original_img.ravel(), 256, [0, 256], color='blue')
    plt.title('Original Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(orig_hist_path)
    plt.close()
    
    # Enhanced Histogram
    plt.figure(figsize=(5, 4))
    plt.hist(enhanced_img.ravel(), 256, [0, 256], color='green')
    plt.title('Enhanced Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(enh_hist_path)
    plt.close()
    
    return orig_hist_path, enh_hist_path

def process_pipeline(image_path, output_dir, filename_prefix):
    """
    Runs the complete image processing pipeline and saves intermediate results.
    Returns paths to saved images and extracted features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Preprocess
    orig_img = load_and_preprocess(image_path)
    orig_path = os.path.join(output_dir, f"{filename_prefix}_1_orig.jpg")
    cv2.imwrite(orig_path, orig_img)
    
    # 2. Enhance
    enh_img = enhance_image(orig_img)
    enh_path = os.path.join(output_dir, f"{filename_prefix}_2_enh.jpg")
    cv2.imwrite(enh_path, enh_img)
    
    # 3. Noise Removal
    noise_rm_img = remove_noise(enh_img)
    noise_path = os.path.join(output_dir, f"{filename_prefix}_3_noise_rm.jpg")
    cv2.imwrite(noise_path, noise_rm_img)
    
    # 4. Sharpen
    sharp_img = sharpen_image(noise_rm_img)
    sharp_path = os.path.join(output_dir, f"{filename_prefix}_4_sharp.jpg")
    cv2.imwrite(sharp_path, sharp_img)
    
    # 5. Segment (Use orig_img to avoid histogram equalization artifacts)
    mask, edges = segment_image(orig_img)
    seg_path = os.path.join(output_dir, f"{filename_prefix}_5_mask.jpg")
    cv2.imwrite(seg_path, mask)
    
    # Create highlighted original (draw bounding box around tumor)
    highlighted_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour which is usually the tumor
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100: # filter small noise
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(highlighted_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
    highlight_path = os.path.join(output_dir, f"{filename_prefix}_6_highlight.jpg")
    cv2.imwrite(highlight_path, highlighted_img)
    
    # 6. Extract Features (from orig_img)
    features = extract_features(orig_img, mask)
    
    # 7. Generate Histograms
    hist_prefix = os.path.join(output_dir, filename_prefix)
    orig_hist_path, enh_hist_path = generate_histograms(orig_img, enh_img, hist_prefix)
    
    return {
        'original': orig_path,
        'enhanced': enh_path,
        'noise_removed': noise_path,
        'sharpened': sharp_path,
        'segmented': seg_path,
        'highlighted': highlight_path,
        'original_hist': orig_hist_path,
        'enhanced_hist': enh_hist_path,
        'features': features
    }
