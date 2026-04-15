"""
Configuración centralizada del sistema de reconocimiento facial.
"""

# ============================================================================
# RUTAS DE DATOS Y MODELOS
# ============================================================================

DATA_PATH = "Data"      # Carpeta principal de entrenamiento. Contiene subcarpetas con nombre de persona
EMBEDDINGS_PATH = "Data/embeddings.pkl"     # Ruta del modelo entrenado (KNN + normalizer + lista de personas)


# ==============================================================================
# PARÁMETROS DE CAPTURA
# ==============================================================================

IMAGE_SIZE = (224, 224)     # Tamaño objetivo para rostros capturados. Requerido por face_recognition.
FRAME_WIDTH = 640           # Ancho del frame capturado en vivo desde la cámara.
MAX_IMAGES = 100            # Número de imágenes a capturar por persona. 100+ recomendado para precisión.
FACE_PADDING = 0.2          # Margen relativo alrededor del bbox detectado (20% de altura/ancho) 


# ==============================================================================
# PARÁMETROS DE RECONOCIMIENTO
# ==============================================================================

DISTANCE_THRESHOLD = 0.8    # Umbral de similitud euclidiana para clasificación.Valores menores = más restrictivo (0.6-0.9)


# ==============================================================================
# OPTIMIZACIONES DE RENDIMIENTO
# ==============================================================================

DETECTION_SCALE = 0.25      # Factor de reducción para detección HOG (0.25-0.5).
FRAME_SKIP = 5              # Procesar detector cada N frames.


# ==============================================================================
# INTERFAZ GRÁFICA
# ==============================================================================

IMAGE_FRAME_SIZE = (640, 480)   # Dimensiones del frame de video mostrado en la ventana tkinter.
