import cv2
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "images" / "raw"

img = cv2.imread(str(RAW_DIR / "person_01_man_young.png"))
print("shape:", img.shape)
print("dtype:", img.dtype)
print("min/max:", img.min(), img.max())
print("memory bytes:", img.nbytes)

# Example conversion BGR -> RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Example grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Example rotation
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), 20, 1.0)
rot = cv2.warpAffine(img, M, (w, h))

# Example reflection
flip = cv2.flip(img, 1)

print("Example files are stored under:")
for p in sorted((BASE / "images").rglob("*.png")):
    print(p.relative_to(BASE))
