"""
Configuración centralizada del proyecto de reconocimiento facial.
"""

# ===== RUTAS DE ARCHIVOS =====
DATA_PATH = "Data"                                              # Carpeta con datos de entrenamiento
MODEL_PATH = "modeloFisherFace.xml"                            # Modelo entrenado de reconocimiento
DNN_PROTO_PATH = "DNN/deploy.prototxt"                         # Arquitectura de red DNN
DNN_MODEL_PATH = "DNN/res10_300x300_ssd_iter_140000.caffemodel"  # Pesos pre-entrenados DNN

# ===== CONFIGURACIÓN DE CAPTURA (faceCapture.py) =====
IMAGE_SIZE = (150, 150)         # Tamaño estándar de imágenes de entrenamiento
FRAME_WIDTH = 640               # Ancho del frame capturado (redimensionado)
MAX_IMAGES = 500                # Máximo de imágenes por persona
START_COUNT = 0                 # Contador inicial de imágenes

# ===== CONFIGURACIÓN DE RECONOCIMIENTO (faceRecognition.py) =====
THRESHOLD = 50                  # Umbral de confianza: valores menores = más confianza
RECOGNITION_INTERVAL = 0.5      # Intervalo mínimo entre reconocimientos (segundos)

# ===== CONFIGURACIÓN DE GUI (main.py) =====
NEW_FACE_NAME = None            # Nombre de la nueva cara a capturar
IMAGE_FRAME_SIZE = (640, 480)   # Tamaño del frame mostrado en la interfaz
