# Sistema de Reconocimiento Facial en Tiempo Real

Sistema de reconocimiento facial en **tiempo real** con arquitectura multi-thread, detección HOG, extracción de características preentrenadas e interfaz gráfica en Tkinter.

## Requisitos

- **Python:** 3.8+
- **Webcam:** Funcional
- **Dependencias:** Ver `requirements.txt`

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

### Flujo básico

1. **Capturar rostro:** Click en "Agregar Rostro" -> "Nuevo Rostro" -> Ingresa nombre -> "Capturar y Entrenar"
   - Se capturan ~100 imágenes automáticamente (ESC para cancelar)
   - El modelo se entrena automáticamente

2. **Reconocimiento:** Click en "Iniciar Reconocimiento"
   - Cuadros verdes = rostros reconocidos
   - Cuadros rojos = rostros desconocidos

3. **Agregar más personas:** Repetir paso 1

## Configuración

Editar `config.py` para ajustar:

```python
MAX_IMAGES = 100              # Imágenes por persona
DISTANCE_THRESHOLD = 0.8      # Umbral de similitud
DETECTION_SCALE = 0.5         # Escala de detección (reducir para más velocidad)
FRAME_SKIP = 5                # Procesa 1 de cada N frames
```

## Estructura del Proyecto

```
├── main.py           # GUI (Tkinter)
├── config.py         # Configuración
├── recognition.py    # Motor de reconocimiento + threads
├── capture.py        # Captura de imágenes
├── train.py          # Entrenamiento del modelo
├── requirements.txt  # Dependencias
├── .gitignore        # Archivos a ignorar
└── Data/             # Datos y modelo entrenado (generado)
```

## Solución de Problemas

**Cámara no abre:** `cap = cv2.VideoCapture(1)` en main.py (cambiar ID)

**Reconocimiento impreciso:** Aumenta `MAX_IMAGES` en config.py (200-300)

**Video lento:** Aumenta `DETECTION_SCALE` a 0.75 o `FRAME_SKIP` a 10

**Falsos positivos:** Reduce `DISTANCE_THRESHOLD` a 0.6

## Tecnologías

- OpenCV: Procesamiento de video
- face_recognition: Detección HOG y embeddings
- scikit-learn: Clasificación KNN
- Tkinter: Interfaz gráfica

## Referencias

- [face_recognition](https://github.com/ageitgey/face_recognition)
- [OpenCV](https://docs.opencv.org/)
- [scikit-learn](https://scikit-learn.org/)


