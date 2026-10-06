
"""Draw illustrative pitch, yaw and roll axes from metadata."""
import cv2
import numpy as np
import math
from common import list_people, load_bgr, load_metadata, ensure_dir, save_bgr, draw_title

out_dir = ensure_dir('09_pose')

for name in list_people():
    img = load_bgr(name)
    meta = load_metadata(name)
    bbox = meta['bbox_xyxy']
    pts = meta['approx_landmarks']
    nose = np.array([pts['nose_tip']['x'], pts['nose_tip']['y']], dtype=np.float32)
    pyr = meta['pitch_yaw_roll_degrees']

    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    scale = (x2 - x1) * 0.22
    pitch = math.radians(pyr['pitch'])
    yaw = math.radians(pyr['yaw'])
    roll = math.radians(pyr['roll'])

    out = img.copy()
    origin = tuple(nose.astype(int))
    x_axis = nose + np.array([math.cos(roll) * scale, math.sin(roll) * scale], dtype=np.float32)
    y_axis = nose + np.array([-math.sin(roll) * scale, -math.cos(roll) * scale], dtype=np.float32)
    z_axis = nose + np.array([yaw * 2.5 * scale, -pitch * 2.8 * scale], dtype=np.float32)

    cv2.arrowedLine(out, origin, tuple(x_axis.astype(int)), (0, 0, 255), 4, tipLength=0.15)
    cv2.arrowedLine(out, origin, tuple(y_axis.astype(int)), (0, 255, 0), 4, tipLength=0.15)
    cv2.arrowedLine(out, origin, tuple(z_axis.astype(int)), (255, 0, 0), 4, tipLength=0.15)
    cv2.putText(out, 'X', tuple((x_axis + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(out, 'Y', tuple((y_axis + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(out, 'Z', tuple((z_axis + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2, cv2.LINE_AA)
    cv2.rectangle(out, (20, 15), (290, 95), (25,25,25), -1)
    cv2.putText(out, f"Pitch: {pyr['pitch']:+.1f} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Yaw:   {pyr['yaw']:+.1f} deg", (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Roll:  {pyr['roll']:+.1f} deg", (30, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    out = draw_title(out, 'Illustrative pitch / yaw / roll')

    save_bgr(out_dir / f'{name}_pose.png', out)
    print('Saved:', out_dir / f'{name}_pose.png')
