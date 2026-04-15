"""
Reconocimiento facial en tiempo real con threading y detección paralela.
"""
import cv2
import os
import pickle
import threading
import queue
import face_recognition

from config import EMBEDDINGS_PATH, DISTANCE_THRESHOLD, DETECTION_SCALE, FRAME_SKIP


class CameraThread(threading.Thread):
    """Hilo dedicado para capturar frames de la cámara."""

    def __init__(self, video_capture: cv2.VideoCapture):
        super().__init__(daemon=True)
        self._video_capture = video_capture
        self._current_frame = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        """Bucle de captura: lee frames continuamente del dispositivo."""
        while not self._stop_event.is_set():
            success, frame = self._video_capture.read()
            if not success:
                break
            
            # Guardar frame más reciente (espejo horizontal para selfie)
            with self._frame_lock:
                self._current_frame = cv2.flip(frame, 1)

    def get_frame(self):
        """Devuelve copia del frame más reciente, o None si no hay ninguno."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def stop(self):
        """Detiene el thread de forma segura."""
        self._stop_event.set()



class DetectorThread(threading.Thread):
    """Hilo dedicado para detectar y clasificar rostros."""

    def __init__(self):
        super().__init__(daemon=True)
        # Colas no-bloqueantes para comunicación
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        
        # Modelo KNN entrenado
        self.classifier = None
        self.normalizer = None
        self.people_names = []
        
        self._stop_event = threading.Event()
        self._load_model()

    def _load_model(self):
        """Carga el modelo entrenado desde disco."""
        if not os.path.exists(EMBEDDINGS_PATH):
            return
        
        try:
            with open(EMBEDDINGS_PATH, 'rb') as model_file:
                model_data = pickle.load(model_file)
                self.classifier = model_data.get('knn')
                self.normalizer = model_data.get('normalizer')
                self.people_names = model_data.get('people', [])
        except Exception as e:
            print(f"Warning - Error cargando modelo: {e}")

    def run(self):
        """Bucle principal: procesa frames de la cola de entrada."""
        while not self._stop_event.is_set():
            try:
                # Esperar frame con timeout para permitir shutdown limpio
                frame = self.input_queue.get(timeout=0.1)
                
                # Señal de parada
                if frame is None:
                    break
                
                # Detectar y procesar
                detections = self._detect_faces(frame) if self.classifier else []
                self._send_results(detections)
                
            except queue.Empty:
                pass

    def _detect_faces(self, frame):
        """Detecta rostros en el frame y retorna lista de detecciones."""
        # Reducir tamaño para acelerar detección
        scaled_frame = cv2.resize(
            frame, (0, 0), 
            fx=DETECTION_SCALE, 
            fy=DETECTION_SCALE
        )
        rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros (HOG)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        if not face_locations:
            return []
        
        # Extraer características y clasificar
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_encodings_normalized = self.normalizer.transform(face_encodings)
        
        # Predicción
        distances = self.classifier.kneighbors(face_encodings_normalized, n_neighbors=1)[0]
        class_indices = self.classifier.predict(face_encodings_normalized)
        
        # Construir resultados
        detections = []
        for class_idx, face_loc, min_distance in zip(class_indices, face_locations, distances):
            top, right, bottom, left = face_loc
            
            # Restaurar a escala original
            top = int(top / DETECTION_SCALE)
            right = int(right / DETECTION_SCALE)
            bottom = int(bottom / DETECTION_SCALE)
            left = int(left / DETECTION_SCALE)
            
            # Clasificar como conocido o desconocido
            is_match = min_distance[0] <= DISTANCE_THRESHOLD
            person_name = self.people_names[class_idx] if is_match else "Desconocido"
            color = (0, 255, 0) if is_match else (0, 0, 255)  # Verde vs Rojo
            
            detections.append(((top, right, bottom, left), person_name, color, min_distance[0]))
        
        return detections

    def _send_results(self, detections):
        """Envía resultados descartando anteriores si la cola está llena."""
        try:
            self.output_queue.put_nowait(detections)
        except queue.Full:
            # Descartar resultado antiguo y enviar nuevo
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(detections)
            except queue.Full:
                pass

    def submit(self, frame):
        """Submite un frame para procesamiento."""
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            # Frame anterior descartado, enviar nuevo
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(frame)
            except queue.Full:
                pass

    def get_results(self):
        """Obtiene resultados sin bloquear. Retorna None si no hay nuevos."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Detiene el thread de forma segura."""
        try:
            self.input_queue.put(None)  # Señal de parada
        except queue.Full:
            pass
        self._stop_event.set()



def recognize(video_capture=None):
    """Generador de reconocimiento facial en tiempo real."""
    # Inicializar captura si no se proporciona
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
    
    # Iniciar threads
    camera_thread = CameraThread(video_capture)
    detector_thread = DetectorThread()
    camera_thread.start()
    detector_thread.start()
    
    # Estado
    frame_count = 0
    latest_detections = []
    
    try:
        while True:
            frame = camera_thread.get_frame()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Procesar cada N frames
            if frame_count % FRAME_SKIP == 0:
                detector_thread.submit(frame)
            
            # Obtener resultados si hay nuevos
            new_detections = detector_thread.get_results()
            if new_detections is not None:
                latest_detections = new_detections
            
            # Dibujar detecciones en frame
            _draw_detections(frame, latest_detections)
            
            yield frame
            
    finally:
        camera_thread.stop()
        detector_thread.stop()
        video_capture.release()


def _draw_detections(frame, detections):
    """Dibuja rostros detectados en el frame (modifica in-place)."""
    height, width = frame.shape[:2]
    
    for (top, right, bottom, left), person_name, color, distance in detections:
        # Limitar al tamaño del frame
        top = max(0, min(top, height))
        bottom = max(0, min(bottom, height))
        left = max(0, min(left, width))
        right = max(0, min(right, width))
        
        # Dibujar rectángulo de rostro
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Dibujar etiqueta con nombre
        cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
        cv2.putText(
            frame, person_name,
            (left + 5, bottom + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2
        )