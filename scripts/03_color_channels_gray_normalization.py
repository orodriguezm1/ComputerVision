
"""Demonstrate BGR->RGB, channels, grayscale, magenta and normalization."""
import cv2
import numpy as np
from common import list_people, load_bgr, ensure_dir, save_bgr, make_grid, draw_title

out_dir = ensure_dir('03_color_channels')

for name in list_people():
    img_bgr = load_bgr(name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(img_bgr)

    red_only = img_rgb.copy(); red_only[:,:,1] = 0; red_only[:,:,2] = 0
    green_only = img_rgb.copy(); green_only[:,:,0] = 0; green_only[:,:,2] = 0
    blue_only = img_rgb.copy(); blue_only[:,:,0] = 0; blue_only[:,:,1] = 0
    magenta = img_rgb.copy(); magenta[:,:,1] = 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    normalized = (img_rgb.astype(np.float32) / 255.0 * 255).astype(np.uint8)

    images = [
        draw_title(img_bgr, 'Original BGR (displayed as image file)'),
        draw_title(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 'Converted to RGB'),
        draw_title(cv2.cvtColor(red_only, cv2.COLOR_RGB2BGR), 'Red channel'),
        draw_title(cv2.cvtColor(green_only, cv2.COLOR_RGB2BGR), 'Green channel'),
        draw_title(cv2.cvtColor(blue_only, cv2.COLOR_RGB2BGR), 'Blue channel'),
        draw_title(cv2.cvtColor(magenta, cv2.COLOR_RGB2BGR), 'Magenta (R+B)'),
        draw_title(gray_bgr, 'Grayscale'),
        draw_title(cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR), 'Normalized [0,1] then visualized'),
    ]
    labels = ['orig', 'rgb', 'R', 'G', 'B', 'magenta', 'gray', 'norm']
    grid = make_grid(images, labels, cols=3, target_size=(300, 375))
    save_bgr(out_dir / f'{name}_grid.png', grid)
    print('Saved:', out_dir / f'{name}_grid.png')
