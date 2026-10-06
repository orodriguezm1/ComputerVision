
"""Compute Histogram of Oriented Gradients (HOG) and visualize it."""
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from common import list_people, load_bgr, ensure_dir, save_bgr, make_grid, draw_title

out_dir = ensure_dir('07_hog')

for name in list_people():
    img = load_bgr(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    hog_vis = exposure.rescale_intensity(hog_image, in_range='image', out_range=(0, 255)).astype(np.uint8)
    hog_vis_bgr = cv2.cvtColor(hog_vis, cv2.COLOR_GRAY2BGR)

    # Visual explanation: show gradient/cell grid overlay
    overlay = img.copy()
    h, w = gray.shape
    for x in range(0, w, 16):
        cv2.line(overlay, (x, 0), (x, h), (50, 200, 255), 1)
    for y in range(0, h, 16):
        cv2.line(overlay, (0, y), (w, y), (50, 200, 255), 1)
    cv2.putText(overlay, f'HOG feature length: {len(features)}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    images = [
        draw_title(img, 'Original'),
        draw_title(overlay, 'Cells used for HOG'),
        draw_title(hog_vis_bgr, 'HOG visualization'),
    ]
    labels = ['orig', 'cells', 'hog']
    grid = make_grid(images, labels, cols=3, target_size=(320, 400))
    save_bgr(out_dir / f'{name}_hog.png', grid)
    print('Saved:', out_dir / f'{name}_hog.png')
    print(name, 'HOG feature length =', len(features))
