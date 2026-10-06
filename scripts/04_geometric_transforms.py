
"""Resize, crop, rotate and reflect the images."""
import cv2
from common import list_people, load_bgr, ensure_dir, save_bgr, make_grid, draw_title

out_dir = ensure_dir('04_geometric_transforms')

for name in list_people():
    img = load_bgr(name)
    h, w = img.shape[:2]

    resized = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    resized_vis = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)

    x1, y1, x2, y2 = int(0.2*w), int(0.18*h), int(0.8*w), int(0.82*h)
    crop = img[y1:y2, x1:x2]
    crop_vis = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    M = cv2.getRotationMatrix2D((w/2, h/2), 25, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    reflected = cv2.flip(img, 1)

    images = [
        draw_title(img, 'Original'),
        draw_title(resized_vis, 'Resize 50% (area)'),
        draw_title(crop_vis, 'ROI crop (resized for view)'),
        draw_title(rotated, 'Rotation +25°'),
        draw_title(reflected, 'Horizontal reflection'),
    ]
    labels = ['orig', 'resize', 'crop', 'rotate', 'reflect']
    grid = make_grid(images, labels, cols=3, target_size=(300, 375))
    save_bgr(out_dir / f'{name}_grid.png', grid)
    print('Saved:', out_dir / f'{name}_grid.png')
