
"""Reflect an image with respect to an arbitrary line defined by two points.
Uses translate -> rotate -> reflect -> unrotate -> untranslate.
"""
import cv2
import numpy as np
from common import load_bgr, ensure_dir, save_bgr, draw_title, make_grid

name = 'person_01_man_young'
out_dir = ensure_dir('05_reflection_line')
img = load_bgr(name)
h, w = img.shape[:2]

# Two points defining a symmetry line (choose a vertical-ish line through the face)
P1 = np.array([w * 0.47, h * 0.20], dtype=np.float32)
P2 = np.array([w * 0.52, h * 0.83], dtype=np.float32)


def to_homogeneous(M23):
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M23
    return H


def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)


def rotation(theta_rad):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def reflect_x_axis():
    # Reflect across the x-axis in a standard Cartesian system.
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

# In image coordinates y goes downward, but the same composition still produces the desired image-space effect.
v = P2 - P1
theta = np.arctan2(v[1], v[0])
H = translation(P1[0], P1[1]) @ rotation(theta) @ reflect_x_axis() @ rotation(-theta) @ translation(-P1[0], -P1[1])
M = H[:2, :]
reflected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

overlay = img.copy()
cv2.line(overlay, tuple(P1.astype(int)), tuple(P2.astype(int)), (0, 255, 255), 3)
cv2.circle(overlay, tuple(P1.astype(int)), 6, (0, 0, 255), -1)
cv2.circle(overlay, tuple(P2.astype(int)), 6, (255, 0, 0), -1)
cv2.putText(overlay, 'P1', tuple((P1 + [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(overlay, 'P2', tuple((P2 + [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

images = [draw_title(overlay, 'Original + line through P1,P2'), draw_title(reflected, 'Reflection w.r.t. arbitrary line')]
labels = ['original', 'reflected_line']
grid = make_grid(images, labels, cols=2, target_size=(360, 450))
save_bgr(out_dir / f'{name}_reflection_line.png', grid)
print('Saved:', out_dir / f'{name}_reflection_line.png')
print('Homogeneous matrix H =\n', H)
