"""
Reconocimiento facial en tiempo real.
"""
import cv2
import os
import pickle
import threading
import queue
import face_recognition
from config import EMBEDDINGS_PATH, DISTANCE_THRESHOLD, DETECTION_SCALE, FRAME_SKIP

# ─────────────────────────────────────────────────────────────────────────────
# Hilo de cámara
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Lee frames de la cámara en bucle y guarda siempre el más reciente."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self._cap, self._frame, self._lock, self._stop = cap, None, threading.Lock(), threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                break
            with self._lock:
                self._frame = cv2.flip(frame, 1)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop.set()

# ─────────────────────────────────────────────────────────────────────────────
# Hilo de detección de rostros
# ─────────────────────────────────────────────────────────────────────────────

class DetectorThread(threading.Thread):
    """Procesa frames en un hilo separado para detectar y clasificar rostros."""

    def __init__(self):
        super().__init__(daemon=True)
        self.input_q, self.output_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.knn_clf, self.normalizer, self.people, self._stop = None, None, [], threading.Event()
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'rb') as f:
                model_data = pickle.load(f)
                self.knn_clf = model_data.get('knn')
                self.normalizer = model_data.get('normalizer')
                self.people = model_data.get('people', [])

    def run(self):
        """Bucle principal que procesa frames de la cola."""
        while not self._stop.is_set():
            try:
                frame = self.input_q.get(timeout=0.1)
                if frame is None:
                    break
                self._put_result(self._detect(frame) if self.knn_clf else [])
            except queue.Empty:
                pass

    def _detect(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small, model='hog')
        if not face_locs:
            return []
        
        face_enc = face_recognition.face_encodings(rgb_small, face_locs)
        face_enc_norm = self.normalizer.transform(face_enc)
        dists = self.knn_clf.kneighbors(face_enc_norm, n_neighbors=1)[0]
        indices = self.knn_clf.predict(face_enc_norm)
        results = []
        for idx, (t, r, b, l), dist in zip(indices, face_locs, dists):
            t, r, b, l = int(t/DETECTION_SCALE), int(r/DETECTION_SCALE), int(b/DETECTION_SCALE), int(l/DETECTION_SCALE)
            name = self.people[idx] if dist[0] <= DISTANCE_THRESHOLD else "Desconocido"
            color = (0, 255, 0) if dist[0] <= DISTANCE_THRESHOLD else (0, 0, 255)
            results.append(((t, r, b, l), name, color, dist[0]))
        return results

    def _put_result(self, results):
        self.output_q.put_nowait(results) if not self.output_q.full() else \
            (self.output_q.get_nowait(), self.output_q.put_nowait(results))

    def submit(self, frame):
        self.input_q.put_nowait(frame) if not self.input_q.full() else \
            (self.input_q.get_nowait(), self.input_q.put_nowait(frame))

    def get_results(self):
        """Devuelve los resultados más recientes sin bloquear."""
        try:
            return self.output_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Detiene el hilo de forma segura."""
        try:
            self.input_q.put(None)
        except queue.Full:
            pass
        self._stop.set()

# ─────────────────────────────────────────────────────────────────────────────
# Función Generadora / Lógica Principal
# ─────────────────────────────────────────────────────────────────────────────

def recognize(cap=None):
    """Generador que captura, detecta rostros y devuelve frames dibujados."""
    cap = cap or cv2.VideoCapture(0)
    cam, det, frame_count, cached = CameraThread(cap), DetectorThread(), 0, []
    cam.start()
    det.start()
    
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                det.submit(frame)
            if (new := det.get_results()) is not None:
                cached = new
            
            h, w = frame.shape[:2]
            for (t, r, b, l), name, color, _ in cached:
                t, b, l, r = max(0, min(t, h)), max(0, min(b, h)), max(0, min(l, w)), max(0, min(r, w))
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.rectangle(frame, (l, b), (r, b + 30), color, cv2.FILLED)
                cv2.putText(frame, name, (l + 5, b + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            yield frame
    finally:
        cam.stop()
        det.stop()
        cap.release()