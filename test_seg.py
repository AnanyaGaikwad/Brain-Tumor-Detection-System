import cv2
import numpy as np
from core.image_processing import segment_image, load_and_preprocess
import glob
import os

yes_images = glob.glob("archive (3)/brain_tumor_dataset/yes/*.jpg")[:10]
no_images = glob.glob("archive (3)/brain_tumor_dataset/no/*.jpg")[:10]

print("Testing YES images:")
for path in yes_images:
    img = load_and_preprocess(path)
    mask, _ = segment_image(img)
    area = np.sum(mask == 255)
    print(f"{os.path.basename(path)} Area: {area}")

print("\nTesting NO images:")
for path in no_images:
    img = load_and_preprocess(path)
    mask, _ = segment_image(img)
    area = np.sum(mask == 255)
    print(f"{os.path.basename(path)} Area: {area}")
