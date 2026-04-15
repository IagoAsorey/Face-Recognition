"""
Reconocimiento facial en tiempo real.
"""
import cv2
import os
import pickle
import threading
import face_recognition

from config import EMBEDDINGS_PATH, DISTANCE_THRESHOLD, DETECTION_SCALE, FRAME_SKIP


# ─────────────────────────────────────────────────────────────────────────────
# Hilo de cámara
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Lee frames de la cámara en bucle y guarda siempre el más reciente."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def get_frame(self):
        """Devuelve una copia del frame más reciente, o None si aún no hay ninguno."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Hilo de detección
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetectionThread(threading.Thread):
    """
    Recibe frames, detecta caras (downscaled), calcula encodings (full-res)
    y clasifica con KNN.

    get_results() → list | None
      - list  : hay un resultado NUEVO disponible (puede ser [] si no se detectó cara)
      - None  : no hay resultado nuevo desde la última consulta (usa el caché)
    """

    def __init__(self, knn, normalizer, people_list):
        super().__init__(daemon=True)
        self._knn = knn
        self._normalizer = normalizer
        self._people = people_list

        self._input_frame = None
        self._results: list = []
        self._new_result = False      # ← flag: hay resultado sin consumir
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()

    # ── API pública ────────────────────────────────────────────────────────

    def submit(self, frame):
        """Envía un frame para procesar (no bloqueante, descarta el anterior)."""
        with self._lock:
            self._input_frame = frame.copy()
        self._new_frame.set()

    def get_results(self):
        """
        Devuelve los resultados solo cuando el hilo acaba de producir uno nuevo.
        Entre detecciones devuelve None para que el caller mantenga su caché.
        """
        with self._lock:
            if self._new_result:
                self._new_result = False
                return list(self._results)
        return None

    def stop(self):
        self._stop.set()
        self._new_frame.set()

    # ── Bucle interno ──────────────────────────────────────────────────────

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

            # 1. Detección en frame reducido (rápido)
            small = cv2.resize(frame, (0, 0),
                               fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations_small = face_recognition.face_locations(rgb_small, model='hog')

            if not locations_small:
                with self._lock:
                    self._results = []
                    self._new_result = True
                continue

            # 2. Escalar coordenadas al tamaño original
            inv = 1.0 / DETECTION_SCALE
            locations_full = [
                (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
                for (t, r, b, l) in locations_small
            ]

            # 3. Encodings sobre frame full-res (más precisos que en el reducido)
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
                self._new_result = True


# ─────────────────────────────────────────────────────────────────────────────
# Generador principal
# ─────────────────────────────────────────────────────────────────────────────

def recognize():
    """
    Generador que produce frames BGR anotados.
    Uso:  gen = recognize(); frame = next(gen)
    Al llamar gen.close() o salir del bucle libera cámara e hilos.
    """
    if not os.path.isfile(EMBEDDINGS_PATH):
        raise FileNotFoundError(f'Modelo no encontrado: {EMBEDDINGS_PATH}')

    with open(EMBEDDINGS_PATH, 'rb') as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede abrir la cámara")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cam_thread = CameraThread(cap)
    detector = FaceDetectionThread(data['knn'], data['normalizer'], data['people'])
    cam_thread.start()
    detector.start()

    print(f"Personas cargadas: {data['people']}")
    print("Iniciando reconocimiento...")

    frame_count = 0
    cached_results: list = []

    try:
        while True:
            # Frame más reciente sin bloquear el hilo principal
            frame = cam_thread.get_frame()
            if frame is None:
                continue

            frame_count += 1

            # Enviar al detector solo cada FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0:
                detector.submit(frame)

            # Solo actualizar caché cuando el detector produce resultado NUEVO
            # → entre medias las cajas permanecen (sin parpadeo)
            new = detector.get_results()
            if new is not None:
                cached_results = new

            # Dibujar caché sobre el frame actual
            h, w = frame.shape[:2]
            for (top, right, bottom, left), name, color, dist in cached_results:
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
        cam_thread.stop()
        cap.release()