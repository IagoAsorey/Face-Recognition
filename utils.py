"""
Funciones utilitarias compartidas entre módulos.
"""
import cv2
import os
from config import DNN_PROTO_PATH, DNN_MODEL_PATH


def load_dnn_model():
    """
    Carga el modelo DNN pre-entrenado para detección de rostros.
    
    Returns:
        cv2.dnn.Net: Modelo de red neuronal DNN.
        
    Raises:
        FileNotFoundError: Si los archivos del modelo no existe.
    """
    if not os.path.exists(DNN_PROTO_PATH) or not os.path.exists(DNN_MODEL_PATH):
        raise FileNotFoundError(
            f"Archivos del modelo DNN no encontrados. "
            f"Verifica que existan:\n"
            f"- {DNN_PROTO_PATH}\n"
            f"- {DNN_MODEL_PATH}"
        )
    return cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)


def init_camera(camera_id=0):
    """
    Inicializa la cámara web.
    
    Args:
        camera_id (int): ID de la cámara (default: 0 para cámara principal).
        
    Returns:
        cv2.VideoCapture: Objeto de captura de video.
        
    Raises:
        RuntimeError: Si no se puede abrir la cámara.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Error: No se puede abrir la cámara.")
    return cap


def create_face_recognizer():
    """
    Crea un reconocedor de rostros usando OpenCV.
    Intenta usar FisherFaceRecognizer si está disponible,
    sino usa LBPHFaceRecognizer.
    
    Returns:
        cv2.face.FaceRecognizer: Objeto reconocedor de rostros.
        
    Raises:
        RuntimeError: Si no se tiene cv2.face disponible.
    """
    if hasattr(cv2, 'face'):
        if hasattr(cv2.face, 'FisherFaceRecognizer_create'):
            return cv2.face.FisherFaceRecognizer_create()
        if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            print('⚠️ FisherFaceRecognizer no disponible; usando LBPHFaceRecognizer.')
            return cv2.face.LBPHFaceRecognizer_create()

    raise RuntimeError(
        'cv2.face no disponible. Instala opencv-contrib-python:\n'
        'pip install opencv-contrib-python'
    )
