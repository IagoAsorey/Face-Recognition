"""
Entrenamiento del modelo KNN con embeddings faciales.
"""
import os
import pickle
import face_recognition
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

from config import DATA_PATH, EMBEDDINGS_PATH
import capture


def train_recognizer():
    """Entrena el modelo KNN con los rostros capturados."""
    # Obtener lista de personas
    people_list = capture.get_people_list()
    if not people_list:
        raise RuntimeError('No hay personas registradas en Data/. Captura rostros primero.')
    
    embeddings = []
    labels = []
    
    # Procesar cada persona
    for label_id, person_name in enumerate(people_list):
        person_path = os.path.join(DATA_PATH, person_name)
        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        print(f"Procesando: {person_name} ({len(image_files)} imágenes)")
        
        # Procesar cada imagen
        for filename in image_files:
            img_path = os.path.join(person_path, filename)
            try:
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Tomar el primer rostro detectado
                    embeddings.append(face_encodings[0])
                    labels.append(label_id)
            except Exception as e:
                print(f"Warning - Error: {filename} - {e}")
    
    # Validar que hay datos para entrenar
    if not embeddings:
        raise RuntimeError('Sin imágenes para entrenar. Captura rostros primero.')
    
    print(f"\nEstadísticas:")
    print(f"  - Embeddings: {len(embeddings)}")
    print(f"  - Personas: {len(people_list)}")
    
    # Normalizar embeddings (escala L2)
    normalizer = Normalizer(norm='l2')
    embeddings_normalized = normalizer.fit_transform(embeddings)
    
    # Entrenar KNN classifier
    n_neighbors = min(3, len(embeddings))
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric='euclidean',
        weights='distance'
    )
    classifier.fit(embeddings_normalized, labels)
    
    # Guardar modelo
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    model_data = {
        'knn': classifier,
        'normalizer': normalizer,
        'people': people_list
    }
    
    with open(EMBEDDINGS_PATH, 'wb') as model_file:
        pickle.dump(model_data, model_file)
    
    print(f"\nModelo guardado: {EMBEDDINGS_PATH}")
