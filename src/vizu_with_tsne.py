# Code developped by Clara and Ketsia
# It performed TSNE on image to perform some vizualisation
# TSNE is good to reduce dimensionnality and when the data have no linear relationship
# We can see the dispostion of our datas in a 2 dimensions plots

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

def load_images_from_folder(folder, max_images=500, image_size=(64, 64)):
    """Function to load images and preprocess them into feature vectors"""
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images:
            break
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).resize(image_size).convert('RGB')
            img_array = np.array(img).flatten()
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images)

def combine_datasets(train1, train2):
    """Combine datasets and create labels"""
    data = np.vstack((train1, train2))
    labels = np.array([0] * len(train1) + [1] * len(train2))
    return data, labels

def perform_tsne():
    return 0

def imshow_2d():
    """Plot the results of the TSNE in 2D"""
    return 0

def imshow_3d():
    """Plot the results of the TSNE in 3D"""
    return 0

def main():
    parser = argparse.ArgumentParser("Perfom visualization of the images in a 2D or 3D axis")
    parser.add_argument("--path_image1", type = str, default = "trainA")
    parser.add_argument("--path_image2", type = str, default = "trainB")
    parser.add_argument("--max_images", type = int, default = 500 )
    args = parser.parse_args()
    train1 = load_images_from_folder(folder = args.path_image1, max_images = args.max_images)
    train2 = load_images_from_folder(folder = args.path_image2, max_images = args.max_images)

if __name__ == "__main__()":
    main()

