# 🎯 Face Recognition System

Sistema completo de reconocimiento facial en tiempo real usando OpenCV y redes neuronales convolucionales (DNN). Captura, entrena y reconoce rostros mediante una interfaz gráfica intuitiva.

## ✨ Características

- ✅ **Captura de rostros** en tiempo real desde webcam
- ✅ **Detección de rostros** usando modelo DNN pre-entrenado (SSD)
- ✅ **Entrenamiento automático** del modelo FisherFace
- ✅ **Reconocimiento en tiempo real** con confianza mostrada
- ✅ **Interfaz gráfica** fácil de usar (Tkinter)
- ✅ **Modular y optimizado** - código limpio y reutilizable

## 🛠️ Requisitos

- Python 3.8+
- Webcam/Cámara web
- 500 MB de espacio libre

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/Face-Recognition.git
cd Face-Recognition
```

### 2. Crear entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
```

**Activar en Windows:**
```bash
venv\Scripts\activate
```

**Activar en Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Iniciar la aplicación
```bash
python main.py
```

### Workflow típico

#### 1. **Agregar un nuevo rostro**
- Click en **"Agregar Rostro"**
- Selecciona **"Nuevo Rostro"** o **"Seleccionar Existente"**
- Ingresa el nombre o selecciona de la lista
- Click en **"Capturar y Entrenar"**
- La cámara capturará automáticamente 500 imágenes
- El modelo se entrenará automáticamente (presiona ESC para cancelar captura)

#### 2. **Iniciar reconocimiento**
- Click en **"Iniciar Reconocimiento"**
- La cámara mostrará detección y reconocimiento en tiempo real
- Los rostros conocidos se mostrarán en **verde** con su nombre
- Los rostros desconocidos se mostrarán en **rojo**

## 📁 Estructura del Proyecto

```
Face-Recognition/
├── main.py                          # Interfaz gráfica (GUI)
├── config.py                        # Configuración centralizada
├── utils.py                         # Funciones utilitarias comunes
├── faceCapture.py                   # Captura de imágenes de rostros
├── training.py                      # Entrenamiento del modelo
├── faceRecognition.py               # Reconocimiento en tiempo real
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore                       # Archivos ignorados por git
├── DNN/                             # Modelos pre-entrenados
│   ├── deploy.prototxt              # Arquitectura de la red neuronal
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pesos pre-entrenados
└── Data/                            # Datos de entrenamiento (generado)
    └── NombrePersona/               # Una carpeta por persona
        └── rostro_0.jpg, rostro_1.jpg, ...
```

## ⚙️ Configuración

Edita `config.py` para personalizar:

```python
IMAGE_SIZE = (150, 150)         # Tamaño de imágenes de entrenamiento
MAX_IMAGES = 500                # Imágenes a capturar por persona
THRESHOLD = 50                  # Umbral de confianza (menor = más strict)
RECOGNITION_INTERVAL = 0.5      # Segundos entre reconocimientos
FRAME_WIDTH = 640               # Ancho de frame capturado
```

## 🧠 Modelos Utilizados

### Detección: SSD (Single Shot MultiBox Detector)
- **Modelo:** ResNet-10 (300x300)
- **Ventajas:** Rápido y preciso, entrenado en WIDER FACE
- **Ubicación:** `DNN/`
- **Regeneración:** Descargable desde repositorios oficiales

### Reconocimiento: FisherFace
- **Algoritmo:** PCA + LDA (Linear Discriminant Analysis)
- **Ventajas:** Robusto, rápido, poco requerimiento computacional
- **Modelo:** Se regenera automáticamente con `training.train_recognizer()`
- **Fallback:** Si no está disponible FisherFace, usa LBPHFaceRecognizer

## 🔧 Módulos

### `config.py`
Configuración centralizada del proyecto. Todas las rutas y parámetros en un solo lugar.

### `utils.py` ⭐
Funciones compartidas para evitar duplicación:
- `load_dnn_model()` - Carga el modelo de detección
- `init_camera()` - Inicializa la webcam
- `create_face_recognizer()` - Crea el reconocedor (con fallback)

### `faceCapture.py`
Captura imágenes de rostros:
- `faces()` - Lista personas registradas
- `capture_faces(name)` - Captura 500 imágenes por persona

### `training.py`
Entrena el modelo:
- `load_images()` - Carga imágenes de Data/
- `train_recognizer()` - Entrena y guarda modelo

### `faceRecognition.py`
Reconocimiento en vivo:
- `process_frame()` - Detecta y reconoce rostros en un frame
- `main()` - Generador de frames procesados

### `main.py`
Interfaz gráfica con clase `FaceRecognitionApp` que gestiona todo.

## 📊 Archivos Generados

| Archivo | Descripción | ¿Se regenera? |
|---------|-------------|--------------|
| `modeloFisherFace.xml` | Modelo entrenado | ✅ Sí (al entrenar) |
| `Data/` | Imágenes de entrenamiento | ✅ Sí (al capturar) |
| `__pycache__/` | Caché de Python | ✅ Sí (automático) |

## 🐛 Solución de Problemas

### **Error: "cv2.face no disponible"**
```bash
pip install opencv-contrib-python
```

### **Error: "Cámara no se abre"**
- Verifica que la cámara no esté en uso por otra aplicación
- Intenta cambiar el ID de cámara en `utils.py`: `init_camera(camera_id=1)`

### **Reconocimiento impreciso**
- Captura más imágenes (aumenta `MAX_IMAGES` en config.py)
- Mejora la iluminación
- Reduce `THRESHOLD` para ser más estricto

### **El programa se congela**
- Presiona ESC durante la captura para cancelar
- Las operaciones de entrenamiento pueden tomar tiempo con muchas imágenes

## 🚀 Mejoras Futuras

- [ ] Soporte para múltiples modelos (VGGFace, FaceNet)
- [ ] Base de datos para almacenar resultados
- [ ] WebAPI REST para integración remota
- [ ] Soporte GPU CUDA
- [ ] Enmascaramiento de privacidad post-reconocimiento
- [ ] Estadísticas y reportes

## 📝 Licencia

Este proyecto es de código abierto. Úsalo libremente en tus proyectos.

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:
1. Fork el repositorio
2. Crea una rama con tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en GitHub.

---

**Hecho con ❤️ | Desarrollado con OpenCV y Python**
