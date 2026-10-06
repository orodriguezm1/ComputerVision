from pathlib import Path
import subprocess

import cv2
import mediapipe as mp
import dlib


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

VIDEO_PATH = (
    Path.home()
    / "Downloads"
    / "Woman_messaging_VOXU_about_lighting_20260927205905.mp4"
)

FRAMES_DIR = (
    Path.home()
    / "Downloads"
    / "frames_voxu"
)

FPS = 10

MAX_FACES = 5

WINDOW_NAME = "MediaPipe + dlib + Landmarks"


# ============================================================
# 2. VERIFICAR VIDEO
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el video:\n{VIDEO_PATH}"
    )


# ============================================================
# 3. CREAR CARPETA PARA FRAMES
# ============================================================

FRAMES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. EXTRAER FRAMES CON FFMPEG
# ============================================================

print("Extrayendo frames con FFmpeg...")

output_pattern = (
    FRAMES_DIR
    / "frame_%05d.jpg"
)

command = [
    "ffmpeg",
    "-y",
    "-i",
    str(VIDEO_PATH),
    "-vf",
    f"fps={FPS}",
    "-q:v",
    "2",
    str(output_pattern)
]

subprocess.run(
    command,
    check=True
)

print("Frames extraídos correctamente.")


# ============================================================
# 5. LISTA DE FRAMES
# ============================================================

frames = sorted(
    FRAMES_DIR.glob("frame_*.jpg")
)

if len(frames) == 0:
    raise RuntimeError(
        "No se encontraron frames."
    )

print(
    f"Frames encontrados: {len(frames)}"
)

delay_ms = int(
    1000 / FPS
)


# ============================================================
# 6. CONFIGURAR MEDIAPIPE
# ============================================================

mp_face_detection = (
    mp.solutions.face_detection
)

mp_face_mesh = (
    mp.solutions.face_mesh
)

mp_drawing = (
    mp.solutions.drawing_utils
)

mp_drawing_styles = (
    mp.solutions.drawing_styles
)


# ============================================================
# 7. MEDIAPIPE FACE DETECTION
# ============================================================

face_detection = (
    mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
)


# ============================================================
# 8. MEDIAPIPE FACE MESH
# ============================================================

face_mesh = (
    mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
)


# ============================================================
# 9. DLIB FACE DETECTOR
# ============================================================

dlib_detector = (
    dlib.get_frontal_face_detector()
)


# ============================================================
# 10. CREAR VENTANA
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ============================================================
# 11. PROCESAR FRAME POR FRAME
# ============================================================

last_frame = None

for frame_number, frame_path in enumerate(
    frames,
    start=1
):

    # --------------------------------------------------------
    # Leer frame
    # --------------------------------------------------------

    frame = cv2.imread(
        str(frame_path)
    )

    if frame is None:
        continue


    # --------------------------------------------------------
    # Dimensiones
    # --------------------------------------------------------

    height, width, _ = (
        frame.shape
    )


    # ========================================================
    # 12. CONVERSIÓN PARA MEDIAPIPE
    # ========================================================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # 13. CONVERSIÓN PARA DLIB
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # 14. MEDIAPIPE FACE DETECTION
    # ========================================================

    detection_results = (
        face_detection.process(
            frame_rgb
        )
    )


    # ========================================================
    # 15. DIBUJAR BBOX MEDIAPIPE
    # ========================================================

    if detection_results.detections:

        for i, detection in enumerate(
            detection_results.detections,
            start=1
        ):

            bbox = (
                detection
                .location_data
                .relative_bounding_box
            )

            # -----------------------------------------------
            # Coordenadas normalizadas -> píxeles
            # -----------------------------------------------

            x1 = int(
                bbox.xmin * width
            )

            y1 = int(
                bbox.ymin * height
            )

            box_width = int(
                bbox.width * width
            )

            box_height = int(
                bbox.height * height
            )

            x2 = (
                x1 + box_width
            )

            y2 = (
                y1 + box_height
            )

            # Evitar coordenadas fuera de la imagen

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width - 1,
                x2
            )

            y2 = min(
                height - 1,
                y2
            )


            # -----------------------------------------------
            # Confianza
            # -----------------------------------------------

            confidence = (
                detection.score[0]
            )


            # -----------------------------------------------
            # BBOX MEDIAPIPE - VERDE
            # -----------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # -----------------------------------------------
            # Etiqueta
            # -----------------------------------------------

            cv2.putText(
                frame,
                f"MediaPipe {confidence:.2f}",
                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # 16. DLIB FACE DETECTION
    # ========================================================

    dlib_faces = (
        dlib_detector(
            gray,
            1
        )
    )


    # ========================================================
    # 17. DIBUJAR BBOX DLIB
    # ========================================================

    for i, face in enumerate(
        dlib_faces,
        start=1
    ):

        x1 = (
            face.left()
        )

        y1 = (
            face.top()
        )

        x2 = (
            face.right()
        )

        y2 = (
            face.bottom()
        )


        # -----------------------------------------------
        # Evitar coordenadas fuera de la imagen
        # -----------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )


        # -----------------------------------------------
        # BBOX DLIB - AZUL
        # -----------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )


        # -----------------------------------------------
        # Etiqueta
        # -----------------------------------------------

        cv2.putText(
            frame,
            "dlib",
            (
                x1,
                min(
                    y2 + 25,
                    height - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # 18. MEDIAPIPE FACE MESH
    # ========================================================

    mesh_results = (
        face_mesh.process(
            frame_rgb
        )
    )


    # ========================================================
    # 19. DIBUJAR LANDMARKS
    # ========================================================

    if (
        mesh_results
        .multi_face_landmarks
    ):

        for face_landmarks in (
            mesh_results
            .multi_face_landmarks
        ):

            # -----------------------------------------------
            # Malla facial
            # -----------------------------------------------

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=(
                    mp_face_mesh
                    .FACEMESH_TESSELATION
                ),
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
            )


            # -----------------------------------------------
            # Contornos
            # -----------------------------------------------

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=(
                    mp_face_mesh
                    .FACEMESH_CONTOURS
                ),
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
            )


            # -----------------------------------------------
            # Iris
            # -----------------------------------------------

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=(
                    mp_face_mesh
                    .FACEMESH_IRISES
                ),
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )
            )


    # ========================================================
    # 20. INFORMACIÓN EN PANTALLA
    # ========================================================

    cv2.putText(
        frame,
        f"Frame {frame_number}/{len(frames)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"FPS: {FPS}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 21. LEYENDA
    # ========================================================

    cv2.putText(
        frame,
        "MediaPipe bbox",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "dlib bbox",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 22. MOSTRAR FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )

    last_frame = (
        frame.copy()
    )


    # ========================================================
    # 23. CONTROL DE REPRODUCCIÓN
    # ========================================================

    key = (
        cv2.waitKey(
            delay_ms
        )
        & 0xFF
    )


    # q -> salir
    if key == ord("q"):
        break


    # espacio -> pausa
    if key == 32:

        print(
            f"Pausa en frame "
            f"{frame_number}"
        )

        print(
            "Presiona una tecla "
            "para continuar."
        )

        key_pause = (
            cv2.waitKey(0)
            & 0xFF
        )

        if key_pause == ord("q"):
            break


# ============================================================
# 24. DEJAR ÚLTIMO FRAME VISIBLE
# ============================================================

if last_frame is not None:

    cv2.imshow(
        WINDOW_NAME,
        last_frame
    )


# ============================================================
# 25. ESPERAR ANTES DE CERRAR
# ============================================================

print()
print("Procesamiento terminado.")
print(
    "Presiona cualquier tecla "
    "sobre la ventana para cerrar."
)

cv2.waitKey(0)


# ============================================================
# 26. LIBERAR RECURSOS
# ============================================================

face_detection.close()

face_mesh.close()

cv2.destroyAllWindows()