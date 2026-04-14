"""Entrenamiento del modelo con embeddings y KNN."""
import os
import pickle
import face_recognition
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

from config import DATA_PATH, EMBEDDINGS_PATH
import capture


def train_recognizer():
    """Entrena el modelo KNN con embeddings de rostros."""
    people_list = capture.get_people_list()
    
    embeddings = []
    labels = []
    
    for label, person_name in enumerate(people_list):
        person_path = os.path.join(DATA_PATH, person_name)
        for file_name in os.listdir(person_path):
            img_path = os.path.join(person_path, file_name)
            try:
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    embeddings.append(face_encodings[0])
                    labels.append(label)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
    
    if not embeddings:
        raise RuntimeError('Sin imágenes para entrenar. Captura rostros primero.')
    
    print(f'Embeddings: {len(embeddings)}, Personas: {len(people_list)}')
    
    normalizer = Normalizer(norm='l2')
    embeddings_normalized = normalizer.fit_transform(embeddings)
    
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(embeddings_normalized, labels)
    
    model_data = {'knn': knn, 'normalizer': normalizer, 'people': people_list}
    
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'✅ Modelo guardado: {EMBEDDINGS_PATH}')
