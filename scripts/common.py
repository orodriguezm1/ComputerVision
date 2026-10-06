
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "images" / "raw"
META_DIR = BASE_DIR / "metadata"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def list_people():
    return sorted([p.stem for p in RAW_DIR.glob("*.png")])


def load_bgr(name: str):
    path = RAW_DIR / f"{name}.png"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_rgb(name: str):
    return cv2.cvtColor(load_bgr(name), cv2.COLOR_BGR2RGB)


def load_metadata(name: str):
    path = META_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(*parts):
    path = OUTPUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_rgb(path, img_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def save_bgr(path, img_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)


def draw_title(img_bgr, title: str):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (25, 25, 25), -1)
    cv2.putText(out, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_grid(images_bgr, labels, cols=3, gap=12, target_size=None):
    if target_size is not None:
        images_bgr = [cv2.resize(im, target_size) for im in images_bgr]
    h, w = images_bgr[0].shape[:2]
    rows = (len(images_bgr) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * gap, cols * w + (cols + 1) * gap, 3), 245, np.uint8)
    for i, (im, label) in enumerate(zip(images_bgr, labels)):
        r, c = divmod(i, cols)
        y = gap + r * (h + gap)
        x = gap + c * (w + gap)
        canvas[y:y+h, x:x+w] = im
        cv2.rectangle(canvas, (x, y+h-34), (x+w, y+h), (15,15,15), -1)
        cv2.putText(canvas, label, (x+8, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (185,185,185), 1)
    return canvas


def show_matplotlib(img_rgb, title=None, cmap=None):
    plt.figure(figsize=(6, 6))
    if cmap is None:
        plt.imshow(img_rgb)
    else:
        plt.imshow(img_rgb, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
