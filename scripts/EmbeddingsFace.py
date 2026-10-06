# ============================================================
# REGISTRO DE PERSONAS MEDIANTE EMBEDDINGS FACIALES
# ============================================================
#
# Flujo:
#
# 1. Verifica si buffalo_l está descargado.
# 2. Si NO está descargado:
#       -> lo descarga automáticamente.
#
# 3. Solicita el nombre de la persona.
#
# 4. Abre la cámara.
#
# 5. Detecta el rostro.
#
# 6. Al presionar ESPACIO:
#       -> obtiene el embedding de 512 dimensiones
#       -> lo normaliza
#       -> lo guarda en un CSV
#
# El CSV tendrá:
#
# Nombre | Embedding
#
# ============================================================


import os
import sys
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

MODEL_NAME = "buffalo_l"

# Carpeta donde está este programa
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


# ------------------------------------------------------------
# Los modelos se guardarán LOCALMENTE dentro del proyecto:
#
# proyecto/
# ├── registrar_rostro.py
# ├── base_rostros.csv
# └── modelos_insightface/
#     └── models/
#         └── buffalo_l/
# ------------------------------------------------------------

MODEL_ROOT = BASE_DIR / "modelos_insightface"

MODEL_DIR = (
    MODEL_ROOT
    / "models"
    / MODEL_NAME
)


# Base de datos de embeddings
CSV_PATH = BASE_DIR / "base_rostros.csv"


# Archivos fundamentales que esperamos encontrar
# dentro de buffalo_l.
#
# w600k_r50.onnx -> reconocimiento / embeddings
# det_10g.onnx    -> detección facial

REQUIRED_MODELS = [
    "w600k_r50.onnx",
    "det_10g.onnx"
]


# ============================================================
# 2. VERIFICAR SI BUFFALO_L ESTÁ INSTALADO
# ============================================================

def buffalo_instalado():

    """
    Comprueba si buffalo_l ya está descargado localmente.
    """

    if not MODEL_DIR.exists():
        return False

    for model_file in REQUIRED_MODELS:

        path = MODEL_DIR / model_file

        if not path.exists():
            return False

    return True


# ============================================================
# 3. CARGAR / DESCARGAR BUFFALO_L
# ============================================================

def cargar_modelo():

    print("\n" + "=" * 60)
    print("INICIALIZACIÓN DE INSIGHTFACE")
    print("=" * 60)

    if buffalo_instalado():

        print("\n✓ buffalo_l ya está instalado.")
        print("✓ No es necesario volver a descargarlo.")
        print(f"\nUbicación:\n{MODEL_DIR}")

    else:

        print("\n⚠ buffalo_l no está instalado.")
        print("Descargando modelo...")
        print("Esto solamente ocurrirá la primera vez.\n")

        # Si existe una descarga incompleta,
        # eliminamos solamente nuestra carpeta local
        # para permitir una descarga limpia.

        if MODEL_DIR.exists():

            print("Se encontró una instalación incompleta.")
            print("Eliminando archivos incompletos...")

            shutil.rmtree(MODEL_DIR)


    # ---------------------------------------------------------
    # FaceAnalysis busca el modelo en:
    #
    # MODEL_ROOT/models/buffalo_l/
    #
    # Si no existe, buffalo_l se descarga automáticamente.
    # ---------------------------------------------------------

    app = FaceAnalysis(
        name=MODEL_NAME,
        root=str(MODEL_ROOT),
        providers=["CPUExecutionProvider"]
    )


    # ctx_id = -1
    #
    # significa CPU.
    #
    # det_size:
    # resolución empleada para la detección facial.

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )


    if not buffalo_instalado():

        print(
            "\nADVERTENCIA: El modelo se cargó, "
            "pero no se localizaron todos los archivos esperados."
        )

    else:

        print("\n✓ Modelo cargado correctamente.")

    return app


# ============================================================
# 4. ABRIR LA CÁMARA
# ============================================================

def abrir_camara():

    """
    Intenta abrir la cámara principal.

    En macOS primero intenta usar AVFoundation.
    Si falla, utiliza el método estándar de OpenCV.
    """

    print("\nAbriendo cámara...")

    # ---------------------------------------------------------
    # macOS
    # ---------------------------------------------------------

    if sys.platform == "darwin":

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_AVFOUNDATION
        )

        if not cap.isOpened():

            cap.release()

            cap = cv2.VideoCapture(0)

    # ---------------------------------------------------------
    # Windows / Linux
    # ---------------------------------------------------------

    else:

        cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        raise RuntimeError(
            "\nNo se pudo acceder a la cámara.\n"
            "Comprueba los permisos de cámara del sistema operativo."
        )


    # Resolución de captura
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    return cap


# ============================================================
# 5. GUARDAR EMBEDDING
# ============================================================

def guardar_embedding(nombre, embedding):

    """
    Guarda:

        Nombre
        Embedding

    en un CSV.

    El embedding se almacena como una lista JSON.
    """


    # ---------------------------------------------------------
    # Convertir numpy array:
    #
    # array([0.1, 0.2, ...])
    #
    # a:
    #
    # "[0.1,0.2,...]"
    # ---------------------------------------------------------

    embedding_json = json.dumps(
        embedding.tolist(),
        separators=(",", ":")
    )


    # ¿Existe ya el CSV?
    archivo_existe = CSV_PATH.exists()


    # ---------------------------------------------------------
    # Abrimos en modo append:
    #
    # "a"
    #
    # significa agregar una nueva persona sin borrar
    # las anteriores.
    # ---------------------------------------------------------

    with open(
        CSV_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)


        # Crear encabezado solamente la primera vez

        if not archivo_existe:

            writer.writerow([
                "Nombre",
                "Embedding"
            ])


        # Agregar persona

        writer.writerow([
            nombre,
            embedding_json
        ])


    print("\n" + "=" * 60)
    print("REGISTRO GUARDADO")
    print("=" * 60)

    print(f"\nNombre: {nombre}")

    print(
        f"Dimensión del embedding: "
        f"{embedding.shape[0]}"
    )

    print(
        f"\nBase de datos:\n"
        f"{CSV_PATH}"
    )


# ============================================================
# 6. REGISTRAR UNA PERSONA
# ============================================================

def registrar_persona(app, nombre):

    cap = abrir_camara()


    print("\n" + "=" * 60)
    print(f"REGISTRANDO A: {nombre}")
    print("=" * 60)

    print(
        "\nMire directamente a la cámara."
    )

    print(
        "Debe aparecer solamente UNA persona."
    )

    print(
        "\nPresione ESPACIO para guardar el rostro."
    )

    print(
        "Presione Q o ESC para cancelar.\n"
    )


    embedding_guardado = False


    while True:

        ret, frame = cap.read()


        if not ret:

            print(
                "No se pudo leer un frame de la cámara."
            )

            break


        # -----------------------------------------------------
        # Detectar rostros
        # -----------------------------------------------------

        faces = app.get(frame)


        # Copia para dibujar
        display = frame.copy()


        # -----------------------------------------------------
        # Dibujar cada rostro
        # -----------------------------------------------------

        for face in faces:

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox


            # Verde si solamente hay una persona
            # Rojo si hay varias

            if len(faces) == 1:

                color = (0, 255, 0)

            else:

                color = (0, 0, 255)


            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


        # -----------------------------------------------------
        # Mensajes en pantalla
        # -----------------------------------------------------

        if len(faces) == 0:

            mensaje = "No se detecta rostro"

            color_texto = (0, 0, 255)


        elif len(faces) == 1:

            mensaje = "Rostro detectado - ESPACIO para registrar"

            color_texto = (0, 255, 0)


        else:

            mensaje = "Debe haber solamente una persona"

            color_texto = (0, 0, 255)


        cv2.putText(
            display,
            mensaje,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_texto,
            2
        )


        cv2.putText(
            display,
            f"Nombre: {nombre}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Registro facial - Buffalo_L / ArcFace",
            display
        )


        # -----------------------------------------------------
        # Leer teclado
        # -----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF


        # =====================================================
        # ESPACIO = 32
        # =====================================================

        if key == 32:

            # Solamente registrar si hay exactamente un rostro

            if len(faces) != 1:

                print(
                    "\nDebe existir exactamente "
                    "un rostro visible."
                )

                continue


            face = faces[0]


            # -------------------------------------------------
            # Obtener embedding ArcFace
            # -------------------------------------------------

            embedding = face.embedding.astype(
                np.float32
            )


            # -------------------------------------------------
            # Normalización L2
            #
            #       x
            # ----------------
            #      ||x||
            #
            # Esto hará posteriormente muy sencilla
            # la comparación mediante cosine similarity.
            # -------------------------------------------------

            norma = np.linalg.norm(embedding)


            if norma == 0:

                print(
                    "Error: embedding con norma cero."
                )

                continue


            embedding_normalizado = (
                embedding / norma
            )


            # -------------------------------------------------
            # Guardar
            # -------------------------------------------------

            guardar_embedding(
                nombre,
                embedding_normalizado
            )


            embedding_guardado = True

            break


        # =====================================================
        # Q o ESC = cancelar
        # =====================================================

        elif key == ord("q") or key == 27:

            print("\nRegistro cancelado.")

            break


    # ---------------------------------------------------------
    # Liberar cámara
    # ---------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()


    return embedding_guardado


# ============================================================
# 7. PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SISTEMA DE REGISTRO FACIAL")
    print("InsightFace - Buffalo_L - ArcFace")
    print("=" * 60)


    # ---------------------------------------------------------
    # PRIMERO:
    # cargar / descargar modelo
    # ---------------------------------------------------------

    app = cargar_modelo()


    # ---------------------------------------------------------
    # SEGUNDO:
    # pedir nombre
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    nombre = input(
        "\nNombre de la persona: "
    ).strip()


    if nombre == "":

        print(
            "\nEl nombre no puede estar vacío."
        )

        return


    # ---------------------------------------------------------
    # TERCERO:
    # inmediatamente abrir cámara
    # ---------------------------------------------------------

    registrado = registrar_persona(
        app,
        nombre
    )


    if registrado:

        print(
            "\n✓ Persona registrada correctamente."
        )

    else:

        print(
            "\nNo se realizó ningún registro."
        )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()