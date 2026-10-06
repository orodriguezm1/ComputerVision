
"""Inspect shape, dtype, range and memory usage of all synthetic images."""
from common import list_people, load_bgr

for name in list_people():
    img = load_bgr(name)
    print("="*70)
    print("Image:", name)
    print("shape:", img.shape)
    print("dtype:", img.dtype)
    print("min/max:", int(img.min()), int(img.max()))
    print("memory bytes:", img.nbytes)
    print("channels:", img.shape[2] if img.ndim == 3 else 1)
