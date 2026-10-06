
"""Optional: try MediaPipe Face Mesh on the synthetic images.
If it works in your environment, it will draw many face landmarks.
"""
import cv2
from common import list_people, load_bgr, ensure_dir, save_bgr

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(f"MediaPipe is not available: {e}")

out_dir = ensure_dir('11_mediapipe_face_mesh')
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    for name in list_people():
        img_bgr = load_bgr(name)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img_rgb)
        out = img_bgr.copy()
        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=out,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
            print(name, ': face detected by MediaPipe')
        else:
            print(name, ': no face detected by MediaPipe')
        save_bgr(out_dir / f'{name}_mediapipe.png', out)
