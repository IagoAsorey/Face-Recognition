"""
Módulo para entrenar el modelo de reconocimiento facial.
"""
import cv2
import os
import numpy as np

from config import DATA_PATH, MODEL_PATH
import faceCapture
from utils import create_face_recognizer

# Carga las imágenes y etiquetas desde las carpetas de personas
def load_images(data_path, people_list):
    """
    Carga imágenes de rostros y sus etiquetas desde carpetas.
    
    Args:
        data_path (str): Ruta a la carpeta Data.
        people_list (list): Lista de nombres de personas.
        
    Returns:
        tuple: (labels, faces_data) - Etiquetas e imágenes en escala de grises.
    """
    labels, faces_data = [], []
    for label, person_name in enumerate(people_list):
        person_path = os.path.join(data_path, person_name)
        for file_name in os.listdir(person_path):
            img_path = os.path.join(person_path, file_name)
            image = cv2.imread(img_path, 0)  # Escala de grises
            if image is not None:
                labels.append(label)
                faces_data.append(image)
            else:
                print(f'⚠️ Error cargando imagen: {img_path}')
    return labels, faces_data


def train_recognizer():
    """
    Entrena el modelo de reconocimiento facial y lo guarda.
    
    Raises:
        RuntimeError: Si no hay imágenes para entrenar en la carpeta Data/.
    """
    people_list = faceCapture.faces()
    labels, faces_data = load_images(DATA_PATH, people_list)
    
    if not faces_data:
        raise RuntimeError(
            'No se encontraron imágenes para entrenar. '
            'Captura rostros primero usando el módulo faceCapture.'
        )

    face_recognizer = create_face_recognizer()
    face_recognizer.train(faces_data, np.array(labels))
    face_recognizer.write(MODEL_PATH)
    print(f'✓ Modelo entrenado y guardado en: {MODEL_PATH}')