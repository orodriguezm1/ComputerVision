
"""Prepare an image tensor for CNNs in TensorFlow/Keras and PyTorch style."""
import cv2
import numpy as np
from common import list_people, load_bgr

name = list_people()[0]
img_bgr = load_bgr(name)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
img_float = img_resized.astype(np.float32) / 255.0

print('Original RGB shape:', img_rgb.shape)
print('Resized shape:', img_resized.shape)
print('Tensor float range:', float(img_float.min()), float(img_float.max()))

# TensorFlow / Keras convention: (N, H, W, C)
keras_input = np.expand_dims(img_float, axis=0)
print('Keras input shape:', keras_input.shape)

# PyTorch convention: (N, C, H, W)
pytorch_input = np.transpose(img_float, (2, 0, 1))
pytorch_input = np.expand_dims(pytorch_input, axis=0)
print('PyTorch-like input shape:', pytorch_input.shape)
