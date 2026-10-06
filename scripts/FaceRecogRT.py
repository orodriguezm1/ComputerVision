# ============================================================
# RECONOCIMIENTO FACIAL EN TIEMPO REAL
# ============================================================
#
# Pipeline:
#
# HILO 1
# Cámara -> captura de frames a FPS especificados
#
#                  ↓
#
# HILO 2
# SCRFD -> detección facial
# ArcFace -> embedding de 512 dimensiones
# Base CSV -> similitud coseno
# Tracker -> ID y color estable por persona
#
#                  ↓
#
# HILO PRINCIPAL
# Visualización con OpenCV
#
# ============================================================

import sys
import csv
import json
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

MODEL_NAME = "buffalo_l"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


# ------------------------------------------------------------
# MISMA CARPETA UTILIZADA EN EL PROGRAMA ANTERIOR
# ------------------------------------------------------------

MODEL_ROOT = BASE_DIR / "modelos_insightface"

MODEL_DIR = (
    MODEL_ROOT
    / "models"
    / MODEL_NAME
)

CSV_PATH = BASE_DIR / "base_rostros.csv"


# ------------------------------------------------------------
# UMBRAL DE RECONOCIMIENTO
#
# Si:
#
# similitud >= 0.65
#
# se considera coincidencia.
#
# Este valor debe calibrarse posteriormente con tus datos.
# ------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.65


# ------------------------------------------------------------
# Tamaño utilizado por SCRFD
# ------------------------------------------------------------

DET_SIZE = (640, 640)


# ------------------------------------------------------------
# Cámara
# ------------------------------------------------------------

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# ------------------------------------------------------------
# Tracking
# ------------------------------------------------------------

TRACK_IOU_THRESHOLD = 0.25

# Cantidad de frames procesados durante los que conservamos
# un track aunque temporalmente desaparezca.
MAX_TRACK_MISSES = 10


# ============================================================
# 2. COLA: CONSERVAR SIEMPRE EL ELEMENTO MÁS RECIENTE
# ============================================================

def put_latest(q, item):

    """
    Introduce un elemento en una Queue(maxsize=1).

    Si la cola está llena, elimina el elemento viejo.

    Esto evita que el vídeo acumule retraso.
    """

    try:

        q.put_nowait(item)

    except queue.Full:

        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            pass


# ============================================================
# 3. VERIFICAR BUFFALO_L
# ============================================================

def buffalo_instalado():

    recognition_model = (
        MODEL_DIR / "w600k_r50.onnx"
    )

    detection_model = (
        MODEL_DIR / "det_10g.onnx"
    )

    return (
        recognition_model.exists()
        and detection_model.exists()
    )


# ============================================================
# 4. CARGAR BUFFALO_L
# ============================================================

def cargar_modelo():

    print("\n" + "=" * 65)
    print("INSIGHTFACE / BUFFALO_L")
    print("=" * 65)

    if buffalo_instalado():

        print("\n✓ buffalo_l ya está instalado.")
        print("✓ No se descargará nuevamente.")
        print(f"\n{MODEL_DIR}")

    else:

        print("\n⚠ buffalo_l no se encontró.")
        print("InsightFace lo descargará automáticamente.")
        print("Esto solamente debería ocurrir la primera vez.")

    # --------------------------------------------------------
    # Solamente necesitamos:
    #
    # detection    -> SCRFD
    # recognition  -> ArcFace
    #
    # No necesitamos cargar edad/género ni otros modelos.
    # --------------------------------------------------------

    app = FaceAnalysis(
        name=MODEL_NAME,
        root=str(MODEL_ROOT),
        allowed_modules=[
            "detection",
            "recognition"
        ],
        providers=[
            "CPUExecutionProvider"
        ]
    )

    app.prepare(
        ctx_id=-1,
        det_size=DET_SIZE
    )

    print("\n✓ InsightFace cargado.")

    return app


# ============================================================
# 5. NORMALIZACIÓN L2
# ============================================================

def normalizar_embedding(embedding):

    """
    Normalización:

                 x
        x' = -----------
              || x ||_2

    Después:

        cosine_similarity(a,b)

    puede calcularse simplemente con:

        a @ b
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norma = np.linalg.norm(embedding)

    if norma == 0:

        return None

    return embedding / norma


# ============================================================
# 6. CARGAR BASE DE DATOS
# ============================================================

def cargar_base_datos():

    print("\n" + "=" * 65)
    print("CARGANDO BASE DE DATOS FACIAL")
    print("=" * 65)

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"\nNo existe:\n{CSV_PATH}\n\n"
            "Ejecuta primero el programa de registro."
        )


    nombres = []
    embeddings = []


    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            nombre = row["Nombre"].strip()

            embedding = np.array(
                json.loads(
                    row["Embedding"]
                ),
                dtype=np.float32
            )


            # Comprobación esperada para Buffalo_L

            if embedding.shape[0] != 512:

                print(
                    f"⚠ Se ignora {nombre}: "
                    f"embedding de tamaño {embedding.shape[0]}"
                )

                continue


            embedding = normalizar_embedding(
                embedding
            )


            if embedding is None:
                continue


            nombres.append(nombre)

            embeddings.append(embedding)


    if len(embeddings) == 0:

        raise RuntimeError(
            "La base de datos no contiene embeddings válidos."
        )


    # --------------------------------------------------------
    # Matriz:
    #
    #        512
    #       ─────
    #
    # P1   [.....]
    # P2   [.....]
    # P3   [.....]
    #
    # dimensión:
    #
    # número_personas × 512
    #
    # --------------------------------------------------------

    embeddings_matrix = np.vstack(
        embeddings
    ).astype(np.float32)


    print(
        f"\n✓ Embeddings cargados: "
        f"{len(nombres)}"
    )


    personas_unicas = sorted(set(nombres))

    print(
        f"✓ Personas diferentes: "
        f"{len(personas_unicas)}"
    )


    print("\nPersonas registradas:")

    for persona in personas_unicas:

        print(f"  - {persona}")


    return nombres, embeddings_matrix


# ============================================================
# 7. COMPARAR EMBEDDING CONTRA TODA LA BASE
# ============================================================

def reconocer_persona(
    embedding,
    nombres,
    embeddings_db
):

    embedding = normalizar_embedding(
        embedding
    )


    if embedding is None:

        return "No coincidente", 0.0


    # --------------------------------------------------------
    # Como TODOS los embeddings están normalizados:
    #
    # embeddings_db @ embedding
    #
    # calcula simultáneamente:
    #
    # cos(e1, embedding)
    # cos(e2, embedding)
    # cos(e3, embedding)
    # ...
    #
    # Sin bucles Python.
    # --------------------------------------------------------

    similarities = (
        embeddings_db @ embedding
    )


    best_index = int(
        np.argmax(similarities)
    )


    best_similarity = float(
        similarities[best_index]
    )


    if (
        best_similarity
        >= SIMILARITY_THRESHOLD
    ):

        return (
            nombres[best_index],
            best_similarity
        )


    return (
        "No coincidente",
        best_similarity
    )


# ============================================================
# 8. INTERSECTION OVER UNION
# ============================================================

def calcular_iou(box_a, box_b):

    """
    IoU entre dos bounding boxes:

    [x1, y1, x2, y2]
    """

    x1 = max(
        box_a[0],
        box_b[0]
    )

    y1 = max(
        box_a[1],
        box_b[1]
    )

    x2 = min(
        box_a[2],
        box_b[2]
    )

    y2 = min(
        box_a[3],
        box_b[3]
    )


    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )


    intersection = (
        intersection_width
        * intersection_height
    )


    area_a = max(
        0,
        box_a[2] - box_a[0]
    ) * max(
        0,
        box_a[3] - box_a[1]
    )


    area_b = max(
        0,
        box_b[2] - box_b[0]
    ) * max(
        0,
        box_b[3] - box_b[1]
    )


    union = (
        area_a
        + area_b
        - intersection
    )


    if union <= 0:

        return 0.0


    return intersection / union


# ============================================================
# 9. TRACKER SENCILLO
# ============================================================

class SimpleTracker:

    """
    Tracker muy ligero basado en IoU.

    Su objetivo aquí NO es sustituir ByteTrack,
    DeepSORT, etc.

    Solamente queremos:

    - mantener un ID temporal por persona
    - mantener su color entre frames
    """

    def __init__(
        self,
        iou_threshold=0.25,
        max_misses=10
    ):

        self.iou_threshold = iou_threshold

        self.max_misses = max_misses

        self.next_id = 1

        self.tracks = {}


    def update(self, detections):

        """
        detections:

        [
            bbox1,
            bbox2,
            ...
        ]

        Devuelve:

        [
            track_id1,
            track_id2,
            ...
        ]
        """

        if len(detections) == 0:

            # Incrementar ausencia
            to_delete = []

            for track_id in self.tracks:

                self.tracks[
                    track_id
                ]["misses"] += 1

                if (
                    self.tracks[
                        track_id
                    ]["misses"]
                    > self.max_misses
                ):

                    to_delete.append(
                        track_id
                    )


            for track_id in to_delete:

                del self.tracks[
                    track_id
                ]


            return []


        # ----------------------------------------------------
        # Marcar detecciones que todavía no han sido asignadas
        # ----------------------------------------------------

        available_detections = set(
            range(len(detections))
        )


        assignments = {}


        # ----------------------------------------------------
        # Intentar asociar tracks anteriores
        # ----------------------------------------------------

        for track_id in list(
            self.tracks.keys()
        ):

            old_bbox = self.tracks[
                track_id
            ]["bbox"]


            best_iou = 0.0

            best_detection = None


            for detection_index in (
                available_detections
            ):

                iou = calcular_iou(
                    old_bbox,
                    detections[
                        detection_index
                    ]
                )


                if iou > best_iou:

                    best_iou = iou

                    best_detection = (
                        detection_index
                    )


            # ------------------------------------------------
            # Coincidencia encontrada
            # ------------------------------------------------

            if (
                best_detection is not None
                and best_iou
                >= self.iou_threshold
            ):

                assignments[
                    best_detection
                ] = track_id


                self.tracks[
                    track_id
                ]["bbox"] = detections[
                    best_detection
                ]


                self.tracks[
                    track_id
                ]["misses"] = 0


                available_detections.remove(
                    best_detection
                )

            else:

                self.tracks[
                    track_id
                ]["misses"] += 1


        # ----------------------------------------------------
        # Eliminar tracks antiguos
        # ----------------------------------------------------

        tracks_to_delete = [

            track_id

            for track_id, track in (
                self.tracks.items()
            )

            if track["misses"]
            > self.max_misses
        ]


        for track_id in tracks_to_delete:

            del self.tracks[
                track_id
            ]


        # ----------------------------------------------------
        # Crear track para detecciones nuevas
        # ----------------------------------------------------

        for detection_index in (
            available_detections
        ):

            track_id = self.next_id

            self.next_id += 1


            self.tracks[
                track_id
            ] = {

                "bbox":
                    detections[
                        detection_index
                    ],

                "misses":
                    0
            }


            assignments[
                detection_index
            ] = track_id


        # ----------------------------------------------------
        # Devolver IDs en el mismo orden que detections
        # ----------------------------------------------------

        return [

            assignments[i]

            for i in range(
                len(detections)
            )
        ]


# ============================================================
# 10. COLOR DIFERENTE PARA CADA TRACK
# ============================================================

def color_track(track_id):

    """
    Genera colores claramente diferentes a partir del ID.

    Utilizamos HSV y posteriormente convertimos a BGR
    porque OpenCV trabaja en BGR.
    """

    hue = (
        track_id * 47
    ) % 180


    hsv = np.uint8(
        [[[
            hue,
            220,
            255
        ]]]
    )


    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )[0][0]


    return (
        int(bgr[0]),
        int(bgr[1]),
        int(bgr[2])
    )


# ============================================================
# 11. HILO DE CAPTURA
# ============================================================

class CaptureThread(
    threading.Thread
):

    def __init__(
        self,
        frame_queue,
        stop_event,
        target_fps
    ):

        super().__init__(
            daemon=True
        )

        self.frame_queue = (
            frame_queue
        )

        self.stop_event = (
            stop_event
        )

        self.target_fps = (
            target_fps
        )


    def abrir_camara(self):

        # ----------------------------------------------------
        # macOS
        # ----------------------------------------------------

        if sys.platform == "darwin":

            cap = cv2.VideoCapture(
                CAMERA_INDEX,
                cv2.CAP_AVFOUNDATION
            )


            if not cap.isOpened():

                cap.release()

                cap = cv2.VideoCapture(
                    CAMERA_INDEX
                )


        # ----------------------------------------------------
        # Windows / Linux
        # ----------------------------------------------------

        else:

            cap = cv2.VideoCapture(
                CAMERA_INDEX
            )


        if not cap.isOpened():

            raise RuntimeError(
                "No se pudo abrir la cámara."
            )


        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )


        # Puede que el driver ignore esta propiedad.
        # Por eso además controlamos el muestreo
        # mediante temporización.

        cap.set(
            cv2.CAP_PROP_FPS,
            self.target_fps
        )


        # Algunos backends aceptan esto.
        # Reduce buffering de cámara.

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        return cap


    def run(self):

        cap = None

        try:

            cap = self.abrir_camara()


            period = (
                1.0
                / self.target_fps
            )


            next_capture = (
                time.perf_counter()
            )


            while not (
                self.stop_event.is_set()
            ):

                ret, frame = cap.read()


                if not ret:

                    print(
                        "\nError leyendo cámara."
                    )

                    self.stop_event.set()

                    break


                now = (
                    time.perf_counter()
                )


                # --------------------------------------------
                # Solamente enviar frame cuando corresponde
                # según los FPS seleccionados.
                # --------------------------------------------

                if now >= next_capture:

                    put_latest(
                        self.frame_queue,
                        (
                            now,
                            frame
                        )
                    )


                    next_capture += period


                    # Si hemos quedado muy retrasados,
                    # resincronizamos.

                    if (
                        now
                        - next_capture
                        > period
                    ):

                        next_capture = (
                            now
                            + period
                        )


        except Exception as e:

            print(
                f"\nError en captura: {e}"
            )

            self.stop_event.set()


        finally:

            if cap is not None:

                cap.release()


# ============================================================
# 12. HILO DE INTELIGENCIA ARTIFICIAL
# ============================================================

class RecognitionThread(
    threading.Thread
):

    def __init__(
        self,
        frame_queue,
        result_queue,
        stop_event,
        app,
        nombres,
        embeddings_db,
        target_fps
    ):

        super().__init__(
            daemon=True
        )


        self.frame_queue = (
            frame_queue
        )

        self.result_queue = (
            result_queue
        )

        self.stop_event = (
            stop_event
        )

        self.app = app

        self.nombres = nombres

        self.embeddings_db = (
            embeddings_db
        )

        self.target_fps = (
            target_fps
        )


        self.tracker = SimpleTracker(
            iou_threshold=
                TRACK_IOU_THRESHOLD,

            max_misses=
                MAX_TRACK_MISSES
        )


        self.processing_fps = 0.0


    def run(self):

        while not (
            self.stop_event.is_set()
        ):

            try:

                capture_time, frame = (
                    self.frame_queue.get(
                        timeout=0.1
                    )
                )


            except queue.Empty:

                continue


            start = (
                time.perf_counter()
            )


            # =================================================
            # INSIGHTFACE
            #
            # SCRFD:
            #   bbox + landmarks
            #
            # ArcFace:
            #   embedding facial
            # =================================================

            try:

                faces = self.app.get(
                    frame
                )

            except Exception as e:

                print(
                    f"\nError en InsightFace: {e}"
                )

                continue


            # -------------------------------------------------
            # BBOX
            # -------------------------------------------------

            bboxes = [

                face.bbox.astype(int)

                for face in faces
            ]


            # -------------------------------------------------
            # TRACK IDs
            # -------------------------------------------------

            track_ids = (
                self.tracker.update(
                    bboxes
                )
            )


            display = frame.copy()


            height, width = (
                display.shape[:2]
            )


            # =================================================
            # PROCESAR CADA ROSTRO
            # =================================================

            for (
                face,
                bbox,
                track_id
            ) in zip(
                faces,
                bboxes,
                track_ids
            ):

                x1, y1, x2, y2 = bbox


                # Limitar coordenadas al frame

                x1 = max(
                    0,
                    min(
                        int(x1),
                        width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        int(y1),
                        height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        int(x2),
                        width - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        int(y2),
                        height - 1
                    )
                )


                # ---------------------------------------------
                # COLOR ÚNICO POR PERSONA/TRACK
                # ---------------------------------------------

                color = color_track(
                    track_id
                )


                # ---------------------------------------------
                # EMBEDDING FACIAL
                # ---------------------------------------------

                embedding = getattr(
                    face,
                    "embedding",
                    None
                )


                if embedding is None:

                    nombre = (
                        "No coincidente"
                    )

                    similitud = 0.0

                else:

                    (
                        nombre,
                        similitud

                    ) = reconocer_persona(

                        embedding,

                        self.nombres,

                        self.embeddings_db
                    )


                # =============================================
                # BOUNDING BOX
                # =============================================

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )


                # =============================================
                # NOMBRE DEBAJO DEL BBOX
                # =============================================

                texto = nombre


                font = (
                    cv2.FONT_HERSHEY_SIMPLEX
                )

                font_scale = 0.70

                thickness = 2


                (
                    text_width,
                    text_height
                ), baseline = cv2.getTextSize(

                    texto,
                    font,
                    font_scale,
                    thickness
                )


                # ---------------------------------------------
                # Preferimos debajo del bounding box
                # ---------------------------------------------

                text_x = x1

                text_y = (
                    y2
                    + text_height
                    + 12
                )


                # Si no cabe debajo, colocarlo arriba

                if (
                    text_y
                    + baseline
                    >= height
                ):

                    text_y = max(
                        text_height
                        + 5,
                        y1 - 10
                    )


                # ---------------------------------------------
                # Fondo del texto
                # ---------------------------------------------

                rect_y1 = max(
                    0,
                    text_y
                    - text_height
                    - 6
                )


                rect_y2 = min(
                    height - 1,
                    text_y
                    + baseline
                    + 4
                )


                rect_x2 = min(
                    width - 1,
                    text_x
                    + text_width
                    + 10
                )


                cv2.rectangle(
                    display,
                    (
                        text_x,
                        rect_y1
                    ),
                    (
                        rect_x2,
                        rect_y2
                    ),
                    color,
                    -1
                )


                cv2.putText(
                    display,
                    texto,
                    (
                        text_x + 5,
                        text_y
                    ),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA
                )


                # ---------------------------------------------
                # Opcional:
                # mostrar similitud pequeña encima
                # ---------------------------------------------

                score_text = (
                    f"sim: {similitud:.3f}"
                )


                cv2.putText(
                    display,
                    score_text,
                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA
                )


            # =================================================
            # MEDIR FPS REAL DE PROCESAMIENTO
            # =================================================

            end = (
                time.perf_counter()
            )


            processing_time = (
                end - start
            )


            instant_fps = (

                1.0
                / processing_time

                if processing_time > 0

                else 0.0
            )


            # Suavizado exponencial

            if (
                self.processing_fps
                == 0
            ):

                self.processing_fps = (
                    instant_fps
                )

            else:

                self.processing_fps = (

                    0.9
                    * self.processing_fps

                    +

                    0.1
                    * instant_fps
                )


            latency_ms = (
                end
                - capture_time
            ) * 1000


            # =================================================
            # INFORMACIÓN GENERAL
            # =================================================

            cv2.putText(
                display,
                (
                    f"Captura objetivo: "
                    f"{self.target_fps:.1f} FPS"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                display,
                (
                    f"Procesamiento: "
                    f"{self.processing_fps:.1f} FPS"
                ),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                display,
                (
                    f"Latencia: "
                    f"{latency_ms:.0f} ms"
                ),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                display,
                (
                    f"Rostros: "
                    f"{len(faces)}"
                ),
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # -------------------------------------------------
            # Enviar solamente el frame procesado más nuevo
            # -------------------------------------------------

            put_latest(
                self.result_queue,
                display
            )


# ============================================================
# 13. PEDIR FPS
# ============================================================

def pedir_fps():

    print("\n" + "=" * 65)
    print("CONFIGURACIÓN DE CAPTURA")
    print("=" * 65)


    while True:

        valor = input(
            "\nFPS de captura [10]: "
        ).strip()


        if valor == "":

            return 10.0


        try:

            fps = float(valor)

        except ValueError:

            print(
                "Introduce un número válido."
            )

            continue


        if fps <= 0:

            print(
                "Los FPS deben ser mayores que cero."
            )

            continue


        if fps > 60:

            print(
                "Utiliza un valor entre 1 y 60 FPS."
            )

            continue


        return fps


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("InsightFace + Buffalo_L + ArcFace")
    print("=" * 65)


    # --------------------------------------------------------
    # FPS elegidos ANTES de activar la cámara
    # --------------------------------------------------------

    target_fps = pedir_fps()


    print(
        f"\nFPS seleccionados: "
        f"{target_fps}"
    )


    # --------------------------------------------------------
    # Cargar base de datos
    # --------------------------------------------------------

    nombres, embeddings_db = (
        cargar_base_datos()
    )


    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------

    app = cargar_modelo()


    # --------------------------------------------------------
    # Colas
    #
    # maxsize = 1:
    #
    # siempre trabajamos con el frame más reciente.
    # --------------------------------------------------------

    frame_queue = queue.Queue(
        maxsize=1
    )

    result_queue = queue.Queue(
        maxsize=1
    )


    # --------------------------------------------------------
    # Señal compartida para detener todos los hilos
    # --------------------------------------------------------

    stop_event = threading.Event()


    # ========================================================
    # HILO 1
    # CAPTURA
    # ========================================================

    capture_thread = CaptureThread(

        frame_queue=
            frame_queue,

        stop_event=
            stop_event,

        target_fps=
            target_fps
    )


    # ========================================================
    # HILO 2
    # DETECCIÓN + EMBEDDINGS + RECONOCIMIENTO
    # ========================================================

    recognition_thread = (
        RecognitionThread(

            frame_queue=
                frame_queue,

            result_queue=
                result_queue,

            stop_event=
                stop_event,

            app=
                app,

            nombres=
                nombres,

            embeddings_db=
                embeddings_db,

            target_fps=
                target_fps
        )
    )


    # --------------------------------------------------------
    # Iniciar hilos
    # --------------------------------------------------------

    capture_thread.start()

    recognition_thread.start()


    print("\n" + "=" * 65)
    print("SISTEMA ACTIVO")
    print("=" * 65)

    print(
        "\nQ o ESC -> cerrar\n"
    )


    # ========================================================
    # HILO PRINCIPAL
    #
    # Dejamos cv2.imshow aquí.
    #
    # Esto es particularmente importante en macOS.
    # ========================================================

    try:

        while not (
            stop_event.is_set()
        ):

            try:

                frame = (
                    result_queue.get(
                        timeout=0.05
                    )
                )


                cv2.imshow(
                    (
                        "Reconocimiento facial "
                        "- Buffalo_L"
                    ),
                    frame
                )


            except queue.Empty:

                pass


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if (
                key == ord("q")
                or key == 27
            ):

                stop_event.set()

                break


    except KeyboardInterrupt:

        stop_event.set()


    finally:

        print(
            "\nCerrando sistema..."
        )

        stop_event.set()


        capture_thread.join(
            timeout=2
        )

        recognition_thread.join(
            timeout=2
        )


        cv2.destroyAllWindows()


        print(
            "✓ Sistema cerrado."
        )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()