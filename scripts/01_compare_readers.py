
"""Compare OpenCV, Pillow, scikit-image and imageio using the same image."""
import time
import numpy as np
import cv2
from PIL import Image
from skimage import io
import imageio.v3 as iio
from common import RAW_DIR, list_people

name = list_people()[0]
path = RAW_DIR / f"{name}.png"

print(f"Using image: {path}")

img_cv = cv2.imread(str(path))
with Image.open(path) as im:
    img_pil = im.copy()
img_sk = io.imread(str(path))
img_iio = iio.imread(path)

print("\n--- Representation ---")
print("OpenCV:", type(img_cv), img_cv.shape, img_cv.dtype, img_cv.min(), img_cv.max())
print("Pillow:", type(img_pil), img_pil.size, img_pil.mode)
print("skimage:", type(img_sk), img_sk.shape, img_sk.dtype, img_sk.min(), img_sk.max())
print("imageio:", type(img_iio), img_iio.shape, img_iio.dtype, img_iio.min(), img_iio.max())
print("OpenCV uses BGR by default; Pillow/skimage/imageio generally use RGB.")

# Fair-ish timing benchmark
N = 25

def bench_cv():
    for _ in range(N):
        _ = cv2.imread(str(path))

def bench_pil():
    for _ in range(N):
        with Image.open(path) as im:
            _ = np.array(im)   # force decoding

def bench_sk():
    for _ in range(N):
        _ = io.imread(str(path))

def bench_iio():
    for _ in range(N):
        _ = iio.imread(path)

for label, fn in [("OpenCV", bench_cv), ("Pillow", bench_pil), ("skimage", bench_sk), ("imageio", bench_iio)]:
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"{label:8s}: {dt:.4f} s for {N} reads")
