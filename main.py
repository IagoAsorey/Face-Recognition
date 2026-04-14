"""
Interfaz gráfica para el sistema de reconocimiento facial.
Permite capturar nuevos rostros, entrenar el modelo y realizar reconocimiento en tiempo real.
"""
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

from config import IMAGE_FRAME_SIZE
import capture
import train
import recognition

class FaceRecognitionApp:
    """Aplicación GUI para reconocimiento facial."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected_name: str | None = None
        self.is_recognizing = False
        self.gen = None

        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Área de video ──────────────────────────────────────────────
        self.image_frame = tk.Frame(
            main_frame,
            width=IMAGE_FRAME_SIZE[0],
            height=IMAGE_FRAME_SIZE[1],
            bg="gray"
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self._show_placeholder()

        # ── Botones ────────────────────────────────────────────────────
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)

        self.btn_start = ttk.Button(
            btn_frame, text="Iniciar Reconocimiento",
            command=self.start_recognition
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(
            btn_frame, text="Detener",
            command=self.stop_recognition,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame, text="Agregar Rostro",
            command=self._add_face_window
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame, text="Salir",
            command=self._on_close
        ).pack(side=tk.LEFT, padx=5)

    def _show_placeholder(self):
        for w in self.image_frame.winfo_children():
            w.destroy()
        ttk.Label(
            self.image_frame,
            text="El video aparecerá aquí",
            foreground="white",
            background="gray"
        ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ------------------------------------------------------------------ Reconocimiento

    def start_recognition(self):
        if self.is_recognizing:
            return
        try:
            self.gen = recognition.recognize()
            self.is_recognizing = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self._update_frame()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {e}")

    def stop_recognition(self):
        self.is_recognizing = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.gen:
            self.gen.close()    # Dispara finally en recognize() → libera cámara y hilo
            self.gen = None
        self._show_placeholder()

    def _update_frame(self):
        """Callback periódico que obtiene el siguiente frame del generador."""
        if not self.is_recognizing:
            return
        try:
            frame = next(self.gen)
            # Redimensionar al tamaño del panel si la cámara tiene otra resolución
            frame = cv2.resize(frame, IMAGE_FRAME_SIZE)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

            for w in self.image_frame.winfo_children():
                w.destroy()
            lbl = ttk.Label(self.image_frame, image=tk_img)
            lbl.image = tk_img   # Evitar que el GC elimine la imagen
            lbl.pack(fill=tk.BOTH, expand=True)

            self.root.after(10, self._update_frame)   # ~100 fps tope de GUI
        except StopIteration:
            self.stop_recognition()

    # ------------------------------------------------------------------ Añadir rostro

    def _add_face_window(self):
        win = tk.Toplevel(self.root)
        win.title("Agregar Rostro")
        win.resizable(False, False)

        # Opciones de modo
        opt_frame = ttk.Frame(win)
        opt_frame.pack(pady=10)

        input_frame = ttk.Frame(win)
        input_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)

        ttk.Button(
            opt_frame, text="Nuevo Rostro",
            command=lambda: self._new_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            opt_frame, text="Seleccionar Existente",
            command=lambda: self._select_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)

        # Acciones
        act_frame = ttk.Frame(win)
        act_frame.pack(pady=10)

        ttk.Button(
            act_frame, text="Capturar y Entrenar",
            command=lambda: self._capture_and_train(win)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            act_frame, text="Cancelar",
            command=win.destroy
        ).pack(side=tk.LEFT, padx=5)

    def _new_face_mode(self, parent: ttk.Frame):
        for w in parent.winfo_children():
            w.destroy()

        ttk.Label(parent, text="Nombre de la persona:").pack(pady=5)
        entry = ttk.Entry(parent)
        entry.pack(pady=5)
        status = ttk.Label(parent, text="")
        status.pack(pady=5)

        def validate(_event=None):
            name = entry.get().strip()
            if not name:
                status.config(text="Nombre vacío", foreground="red")
                self.selected_name = None
            elif name in capture.get_people_list():
                status.config(text="Ya existe — se añadirán imágenes", foreground="orange")
                self.selected_name = name
            else:
                status.config(text="✓ Válido", foreground="green")
                self.selected_name = name

        entry.bind("<KeyRelease>", validate)

    def _select_face_mode(self, parent: ttk.Frame):
        for w in parent.winfo_children():
            w.destroy()

        ttk.Label(parent, text="Rostros registrados:").pack(pady=5)
        faces = capture.get_people_list()
        if not faces:
            ttk.Label(parent, text="No hay rostros registrados",
                      foreground="red").pack(pady=5)
            return

        lb = tk.Listbox(parent, height=5, width=24)
        for f in faces:
            lb.insert(tk.END, f)
        lb.pack(pady=5)

        status = ttk.Label(parent, text="")
        status.pack(pady=5)

        def on_select(_event=None):
            sel = lb.curselection()
            if sel:
                self.selected_name = lb.get(sel[0])
                status.config(text=f"✓ {self.selected_name}", foreground="green")

        lb.bind("<<ListboxSelect>>", on_select)

    def _capture_and_train(self, win: tk.Toplevel):
        if not self.selected_name:
            messagebox.showwarning("Advertencia",
                                   "Por favor selecciona o ingresa un nombre")
            return

        name = self.selected_name
        win.destroy()

        def task():
            try:
                capture.capture_faces(name)
                train.train_recognizer()
                # ← Tkinter no es thread-safe: siempre via root.after()
                self.root.after(0, lambda: messagebox.showinfo(
                    "Éxito", f"Rostro de '{name}' capturado y modelo entrenado ✅"))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Error durante captura/entrenamiento:\n{err}"))

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------ Cierre

    def _on_close(self):
        """Libera recursos antes de cerrar."""
        self.is_recognizing = False
        if self.gen:
            self.gen.close()
        self.root.destroy()


# --------------------------------------------------------------------------- #

def main():
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()