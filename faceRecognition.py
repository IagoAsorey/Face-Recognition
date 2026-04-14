import cv2
import os
import numpy as np

from config import DATA_PATH, MODEL_PATH, THRESHOLD, RECOGNITION_INTERVAL
from utils import load_dnn_model, init_camera, create_face_recognizer


# Procesar cada frame
def process_frame(frame, face_recognizer, face_classif, image_paths, last_recognition_time):
    """
    Procesa un frame detectando y reconociendo rostros.
    
    Args:
        frame (np.ndarray): Frame de video.
        face_recognizer: Modelo de reconocimiento facial.
        face_classif: Modelo DNN para detección.
        image_paths (list): Nombres de personas registradas.
        last_recognition_time (float): Timestamp del último reconocimiento.
        
    Returns:
        tuple: (frame procesado, tiempo de última actualización).
    """
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Detección de rostros
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_classif.setInput(blob)
    detections = face_classif.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confianza mínima
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x, y, x1, y1 = box.astype("int")

            # Procesar rostro detectado
            face_img = cv2.resize(frame[y:y1, x:x1], (150, 150), interpolation=cv2.INTER_CUBIC)
            gray_face = cv2.equalizeHist(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))

            # Inicializar valores predeterminados
            name, color = "Desconocido", (50, 50, 255)

            # Reconocimiento facial con control de intervalo
            if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
                last_recognition_time = current_time
                label, conf = face_recognizer.predict(gray_face)

                # Validar el índice del label
                if 0 <= label < len(image_paths) and conf < THRESHOLD:
                    name, color = image_paths[label], (125, 255, 0)

            # Dibujar resultados (siempre se dibuja el recuadro)
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.rectangle(frame, (x, y1), (x1, y1 + 30), color, -1)
            cv2.putText(frame, f'{name}', (x + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Confianza: {conf:.2f}', (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame, last_recognition_time


def main():
    """
    Función principal para iniciar reconocimiento facial en tiempo real.
    
    Yields:
        np.ndarray: Frames procesados con detección y reconocimiento.
        
    Raises:
        FileNotFoundError: Si el modelo no existe.
    """
    image_paths = os.listdir(DATA_PATH)
    face_recognizer = create_face_recognizer()

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f'Modelo no encontrado: {MODEL_PATH}. '
            f'Entrena el modelo ejecutando training.train_recognizer() primero.'
        )

    face_recognizer.read(MODEL_PATH)
    face_classif = load_dnn_model()
    cap = init_camera()

    last_recognition_time = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield process_frame(frame, face_recognizer, face_classif, image_paths, last_recognition_time)[0]
    finally:
        cap.release()