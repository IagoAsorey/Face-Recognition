"""
Módulo para capturar imágenes de rostros usando cámara web.
"""
import cv2
import os
import numpy as np
import imutils

from config import DATA_PATH, IMAGE_SIZE, START_COUNT, MAX_IMAGES, FRAME_WIDTH
from utils import load_dnn_model, init_camera

def faces():
    """
    Obtiene la lista de usuarios (nombres de carpetas en DATA_PATH).
    
    Returns:
        list: Lista de nombres de personas registradas.
    """
    if not os.path.exists(DATA_PATH):
        return []
    return [name for name in os.listdir(DATA_PATH) 
            if os.path.isdir(os.path.join(DATA_PATH, name))]


def capture_faces(face_name):
    """
    Captura imágenes de rostros desde la cámara y las guarda.
    
    Args:
        face_name (str): Nombre de la persona para crear/actualizar su carpeta.
    """
    face_path = os.path.join(DATA_PATH, face_name)
    os.makedirs(face_path, exist_ok=True)

    face_classifier = load_dnn_model()
    cap = init_camera()
    count = START_COUNT

    try:
        while count < MAX_IMAGES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            h, w = frame.shape[:2]

            # Mostrar contador en pantalla
            cv2.putText(frame, f'Capturing face: {count}/{MAX_IMAGES}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Preprocesar el frame para detección de rostros
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_classifier.setInput(blob)
            detections = face_classifier.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confianza mínima
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x1, y1 = box.astype("int")

                    # Dibujar el rectángulo y guardar la imagen
                    cv2.rectangle(frame, (x - 5, y - 5), (x1 + 5, y1 + 5), (0, 255, 0), 2)
                    face = cv2.resize(frame[y:y1, x:x1], IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(face_path, f'rostro_{count}.jpg'), face)
                    count += 1

                    if count >= MAX_IMAGES:
                        break

            cv2.imshow('Face Capture', frame)
            if cv2.waitKey(1) == 27:  # Salir con la tecla ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()