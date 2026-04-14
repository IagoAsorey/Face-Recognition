"""Reconocimiento facial en tiempo real."""
import cv2
import os
import pickle
import face_recognition

from config import EMBEDDINGS_PATH, DISTANCE_THRESHOLD, RECOGNITION_INTERVAL


def recognize():
    """Generador para reconocimiento facial en tiempo real."""
    if not os.path.isfile(EMBEDDINGS_PATH):
        raise FileNotFoundError(f'Modelo no encontrado: {EMBEDDINGS_PATH}')
    
    print('🔄 Cargando modelo...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    
    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)
        knn = data['knn']
        normalizer = data['normalizer']
        people_list = data['people']
    
    print(f'✅ Personas: {people_list}')
    print('🎥 Iniciando reconocimiento...')
    
    last_recognition_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if len(face_locations) == 0:
                yield frame
                continue
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                top = max(0, top)
                left = max(0, left)
                bottom = min(frame.shape[0], bottom)
                right = min(frame.shape[1], right)
                
                encoding_normalized = normalizer.transform([encoding])[0]
                
                name, color = "Desconocido", (50, 50, 255)
                distance = 1.0
                
                if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
                    last_recognition_time = current_time
                    
                    distances, indices = knn.kneighbors([encoding_normalized], n_neighbors=1)
                    distance = distances[0][0]
                    label = indices[0][0]
                    
                    if 0 <= label < len(people_list) and distance < DISTANCE_THRESHOLD:
                        name = people_list[label]
                        color = (125, 255, 0)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, -1)
                cv2.putText(frame, f'{name}', (left + 5, bottom + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'{distance:.2f}', (left, top - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            yield frame
    
    finally:
        cap.release()
