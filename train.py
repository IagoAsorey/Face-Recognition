"""
Entrenamiento del modelo con embeddings y KNN.
"""
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
 
    if not people_list:
        raise RuntimeError('No hay personas registradas en Data/. Captura rostros primero.')
    
    embeddings = []
    labels = []
    
    for label, person_name in enumerate(people_list):
        person_path = os.path.join(DATA_PATH, person_name)
        image_files = [f for f in os.listdir(person_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
 
        for file_name in image_files:
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
    
    n_neighbors = min(3, len(embeddings))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(embeddings_normalized, labels)
    
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    model_data = {'knn': knn, 'normalizer': normalizer, 'people': people_list}
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'Modelo guardado: {EMBEDDINGS_PATH}')
