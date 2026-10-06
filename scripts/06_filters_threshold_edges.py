
"""Apply smoothing, thresholding, Sobel and Canny."""
import cv2
import numpy as np
from common import list_people, load_bgr, ensure_dir, save_bgr, make_grid, draw_title

out_dir = ensure_dir('06_filters_edges')

for name in list_people():
    img = load_bgr(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
    median = cv2.medianBlur(gray, 7)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx.astype(np.float32), sobely.astype(np.float32))
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-8))
    canny = cv2.Canny(gray, 80, 160)

    def gray_to_bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    images = [
        draw_title(img, 'Original'),
        draw_title(gray_to_bgr(blur), 'Gaussian blur'),
        draw_title(gray_to_bgr(median), 'Median blur'),
        draw_title(gray_to_bgr(thresh), 'Threshold'),
        draw_title(gray_to_bgr(otsu), 'Otsu threshold'),
        draw_title(gray_to_bgr(magnitude), 'Sobel magnitude'),
        draw_title(gray_to_bgr(canny), 'Canny edges'),
    ]
    labels = ['orig', 'gauss', 'median', 'thresh', 'otsu', 'sobel', 'canny']
    grid = make_grid(images, labels, cols=3, target_size=(300, 375))
    save_bgr(out_dir / f'{name}_grid.png', grid)
    print('Saved:', out_dir / f'{name}_grid.png')
