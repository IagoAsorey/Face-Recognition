"""
Configuración centralizada del proyecto de reconocimiento facial.
"""

# ===== RUTAS =====
DATA_PATH = "Data"                          # Carpeta con datos de entrenamiento
EMBEDDINGS_PATH = "Data/embeddings.pkl"     # Modelo entrenado (embeddings + KNN)

# ===== CAPTURA =====
IMAGE_SIZE = (224, 224)     # Tamaño requerido por face_recognition
FRAME_WIDTH = 640           # Ancho del frame capturado
MAX_IMAGES = 100            # Mínimo suficiente para face-recognition
FACE_PADDING = 0.2          # Padding relativo alrededor del crop → mejora encodings

# ===== RECONOCIMIENTO =====
DISTANCE_THRESHOLD = 0.6    # Umbral de similitud: valores menores = más restrictivo

# ===== RENDIMIENTO (ajusta según tu CPU) =====
DETECTION_SCALE = 0.5   # Escala el frame antes de detectar: 0.5 = 4× menos píxeles
FRAME_SKIP = 3          # Lanza detección 1 de cada N frames (el resto usa caché)

# ===== GUI =====
IMAGE_FRAME_SIZE = (640, 480)   # Tamaño del frame mostrado en la interfaz
