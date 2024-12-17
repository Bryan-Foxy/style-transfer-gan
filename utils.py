import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, img_size=(256, 256)):
    img = Image.open(image_path).convert("RGB").resize(img_size)
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    return img

def denormalize(img):
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)