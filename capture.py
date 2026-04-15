"""
Captura de imágenes de rostros desde cámara web.
"""
import cv2
import os
import face_recognition

from config import DATA_PATH, IMAGE_SIZE, MAX_IMAGES, FRAME_WIDTH, FACE_PADDING

def get_people_list():
    """Devuelve la lista de personas registradas en DATA_PATH."""
    if not os.path.exists(DATA_PATH):
        return []
    return [name for name in os.listdir(DATA_PATH) 
            if os.path.isdir(os.path.join(DATA_PATH, name))]

def _pad_bbox(top, right, bottom, left, frame_h, frame_w, padding=FACE_PADDING):
    """
    Amplía el bounding box con un margen relativo.
    """
    h = bottom - top
    w = right - left
    ph = int(h * padding)
    pw = int(w * padding)
    return (
        max(0, top - ph),
        min(frame_w, right + pw),
        min(frame_h, bottom + ph),
        max(0, left - pw),
    )

def capture_faces(face_name):
    """Captura imágenes de un rostro desde la cámara."""
    face_path = os.path.join(DATA_PATH, face_name)
    os.makedirs(face_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Evita frames acumulados en el buffer
    
    count = 0
    print(f"Capturando para: {face_name}")
    print(f"Presiona ESC para terminar")

    try:
        while count < MAX_IMAGES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            scale = FRAME_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * scale)))
            h, w = frame.shape[:2]

            cv2.putText(frame, f'Capturando: {count}/{MAX_IMAGES}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')

            for (top, right, bottom, left) in face_locations:
                top, right, bottom, left = _pad_bbox(top, right, bottom, left, h, w)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(face_path, f'rostro_{count:03d}.jpg'), face_img)
                    count += 1

                if count >= MAX_IMAGES:
                    break

            cv2.imshow('Captura de Rostro', frame)
            if cv2.waitKey(1) == 27:
                break

        print(f"{count} imágenes capturadas para {face_name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
