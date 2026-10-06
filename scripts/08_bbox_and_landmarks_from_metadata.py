
"""Draw the approximate face bounding box and landmarks stored in metadata."""
import cv2
from common import list_people, load_bgr, load_metadata, ensure_dir, save_bgr, make_grid, draw_title

out_dir = ensure_dir('08_bbox_landmarks')

for name in list_people():
    img = load_bgr(name)
    meta = load_metadata(name)
    bbox = meta['bbox_xyxy']
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

    bbox_img = img.copy()
    cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(bbox_img, 'Face bbox', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2, cv2.LINE_AA)

    lm_img = img.copy()
    cv2.rectangle(lm_img, (x1, y1), (x2, y2), (0, 170, 255), 2)
    for label, pt in meta['approx_landmarks'].items():
        x, y = int(pt['x']), int(pt['y'])
        cv2.circle(lm_img, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(lm_img, 'Approx. landmarks', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    grid = make_grid([draw_title(bbox_img, 'Bounding box'), draw_title(lm_img, 'Landmarks')], ['bbox', 'landmarks'], cols=2, target_size=(360, 450))
    save_bgr(out_dir / f'{name}_bbox_landmarks.png', grid)
    print('Saved:', out_dir / f'{name}_bbox_landmarks.png')
