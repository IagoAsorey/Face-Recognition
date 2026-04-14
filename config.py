"""
Configuración centralizada del proyecto de reconocimiento facial.
"""

# ===== RUTAS DE ARCHIVOS =====
DATA_PATH = "Data"                          # Carpeta con datos de entrenamiento
EMBEDDINGS_PATH = "Data/embeddings.pkl"    # Modelo entrenado (embeddings + KNN)

# ===== CONFIGURACIÓN DE CAPTURA (faceCapture.py) =====
IMAGE_SIZE = (224, 224)         # Tamaño requerido por face_recognition
FRAME_WIDTH = 640               # Ancho del frame capturado
MAX_IMAGES = 100                # Mínimo suficiente para face-recognition
START_COUNT = 0                 # Contador inicial de imágenes

# ===== CONFIGURACIÓN DE RECONOCIMIENTO (faceRecognition.py) =====
DISTANCE_THRESHOLD = 0.6        # Umbral de similitud: valores menores = más restrictivo
RECOGNITION_INTERVAL = 0.5      # Intervalo mínimo entre reconocimientos (segundos)

# ===== CONFIGURACIÓN DE GUI (main.py) =====
IMAGE_FRAME_SIZE = (640, 480)   # Tamaño del frame mostrado en la interfaz
