"""
Captura de imágenes de rostros para entrenamiento.
"""
import cv2
import os
import face_recognition

from config import DATA_PATH, IMAGE_SIZE, MAX_IMAGES, FRAME_WIDTH, FACE_PADDING


def get_people_list():
    """Lista todas las personas registradas en el directorio de datos."""
    if not os.path.exists(DATA_PATH):
        return []
    
    return [
        name for name in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, name))
    ]


def _expand_bounding_box(top, right, bottom, left, frame_height, frame_width, padding=FACE_PADDING):
    """Amplía bounding box con margen relativo alrededor del rostro."""
    bbox_height = bottom - top
    bbox_width = right - left
    padding_h = int(bbox_height * padding)
    padding_w = int(bbox_width * padding)
    
    return (
        max(0, top - padding_h),
        min(frame_width, right + padding_w),
        min(frame_height, bottom + padding_h),
        max(0, left - padding_w),
    )


def capture_faces(person_name):
    """Captura imágenes para entrenar el modelo con un nuevo rostro."""
    person_path = os.path.join(DATA_PATH, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Abrir cámara
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    
    # Minimizar buffer para obtener frames frescos
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    capture_count = 0
    print(f"Capturando para: {person_name}")
    print(f"Imágenes a capturar: {MAX_IMAGES}")
    print(f"Presiona ESC para terminar")

    try:
        while capture_count < MAX_IMAGES:
            success, frame = video_capture.read()
            if not success:
                break

            # Preparar frame para visualización
            frame = cv2.flip(frame, 1)
            aspect_ratio = frame.shape[0] / frame.shape[1]
            frame = cv2.resize(frame, (FRAME_WIDTH, int(FRAME_WIDTH * aspect_ratio)))
            frame_height, frame_width = frame.shape[:2]

            # Mostrar progreso
            cv2.putText(
                frame, f'Capturando: {capture_count}/{MAX_IMAGES}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2
            )

            # Detectar rostros en el frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')

            # Procesar cada rostro detectado
            for (top, right, bottom, left) in face_locations:
                top, right, bottom, left = _expand_bounding_box(
                    top, right, bottom, left,
                    frame_height, frame_width
                )
                
                # Dibujar rectángulo
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Extraer y guardar rostro
                face_region = frame[top:bottom, left:right]
                if face_region.size > 0:
                    face_region_resized = cv2.resize(face_region, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
                    filename = os.path.join(person_path, f'rostro_{capture_count:03d}.jpg')
                    cv2.imwrite(filename, face_region_resized)
                    capture_count += 1

                if capture_count >= MAX_IMAGES:
                    break

            # Mostrar frame live
            cv2.imshow('Captura de Rostro', frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

        print(f"OK: {capture_count} imágenes capturadas para {person_name}")

    finally:
        video_capture.release()
        cv2.destroyAllWindows()
