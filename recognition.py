"""
Reconocimiento facial en tiempo real.
"""
import cv2
import os
import pickle
import threading
import face_recognition

from config import EMBEDDINGS_PATH, DISTANCE_THRESHOLD, DETECTION_SCALE, FRAME_SKIP

class FaceDetectionThread(threading.Thread):
    """
    Hilo daemon que recibe frames, detecta caras, calcula encodings y clasifica con KNN.
    """

    def __init__(self, knn, normalizer, people_list):
        super().__init__(daemon=True)
        self._knn = knn
        self._normalizer = normalizer
        self._people = people_list

        self._input_frame = None
        self._results: list = []          # [(loc, name, color, dist), ...]
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()

    # ------------------------------------------------------------------ API

    def submit(self, frame):
        """Envía un frame para procesar (no bloqueante)."""
        with self._lock:
            self._input_frame = frame.copy()
        self._new_frame.set()

    def get_results(self):
        """Devuelve los últimos resultados calculados (no bloqueante)."""
        with self._lock:
            return list(self._results)

    def stop(self):
        """Señaliza al hilo para que termine."""
        self._stop.set()
        self._new_frame.set()   # Desbloquea la espera si está dormido

    # ------------------------------------------------------------------ Loop

    def run(self):
        while not self._stop.is_set():
            self._new_frame.wait()
            self._new_frame.clear()
            if self._stop.is_set():
                break

            with self._lock:
                frame = self._input_frame
            if frame is None:
                continue

            # 1. Escalar frame → detección rápida
            small = cv2.resize(frame, (0, 0),
                               fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations_small = face_recognition.face_locations(rgb_small, model='hog')

            if not locations_small:
                with self._lock:
                    self._results = []
                continue

            # 2. Escalar coordenadas al tamaño original
            inv = 1.0 / DETECTION_SCALE
            locations_full = [
                (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
                for (t, r, b, l) in locations_small
            ]

            # 3. Encodings sobre el frame a resolución completa (más preciso)
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_full, locations_full)

            # 4. Clasificación KNN (coste despreciable)
            results = []
            for enc, loc in zip(encodings, locations_full):
                enc_norm = self._normalizer.transform([enc])[0]
                distances, indices = self._knn.kneighbors([enc_norm], n_neighbors=1)
                dist = float(distances[0][0])
                label = int(indices[0][0])

                if 0 <= label < len(self._people) and dist < DISTANCE_THRESHOLD:
                    name = self._people[label]
                    color = (125, 255, 0)   # Verde → conocido
                else:
                    name = "Desconocido"
                    color = (50, 50, 255)   # Rojo → desconocido

                results.append((loc, name, color, dist))

            with self._lock:
                self._results = results


# --------------------------------------------------------------------------- #

def recognize():
    """
    Generador que produce frames BGR con las anotaciones pintadas.
    """
    if not os.path.isfile(EMBEDDINGS_PATH):
        raise FileNotFoundError(f'Modelo no encontrado: {EMBEDDINGS_PATH}')

    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # Siempre el frame más reciente

    detector = FaceDetectionThread(data['knn'], data['normalizer'], data['people'])
    detector.start()
    print(f"Personas cargadas: {data['people']}")
    print("Iniciando reconocimiento...")

    frame_count = 0
    cached_results: list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Enviar al hilo de detección cada FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0:
                detector.submit(frame)

            # Actualizar caché solo si el hilo tiene nuevos resultados
            new = detector.get_results()
            if new is not None:
                cached_results = new

            # Dibujar resultados cacheados sobre el frame actual
            h, w = frame.shape[:2]
            for (top, right, bottom, left), name, color, dist in cached_results:
                # Clamp por si el frame cambió de tamaño
                top    = max(0, min(top, h))
                bottom = max(0, min(bottom, h))
                left   = max(0, min(left, w))
                right  = max(0, min(right, w))

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 5, bottom + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'{dist:.2f}', (left, top - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            yield frame

    finally:
        detector.stop()
        cap.release()