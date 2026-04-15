"""
Interfaz gráfica para el sistema de reconocimiento facial.

Permite:
- Iniciar reconocimiento facial en tiempo real
- Capturar imágenes de nuevos rostros para entrenar
- Gestionar personas registradas
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
    """Aplicación GUI principal para reconocimiento facial."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected_name: str | None = None
        self.is_recognizing = False
        self.recognition_generator = None

        self._setup_ui()

    def _setup_ui(self):
        """Configura la interfaz gráfica."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Área de video principal
        self.image_frame = tk.Frame(
            main_frame,
            width=IMAGE_FRAME_SIZE[0],
            height=IMAGE_FRAME_SIZE[1],
            bg="gray"
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        self._show_placeholder()

        # Panel de botones principales
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)

        self.btn_start = ttk.Button(
            button_frame, 
            text="Iniciar Reconocimiento",
            command=self.start_recognition
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(
            button_frame, 
            text="Detener",
            command=self.stop_recognition,
            state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, 
            text="Agregar Rostro",
            command=self._open_add_face_dialog
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, 
            text="Salir",
            command=self._on_close
        ).pack(side=tk.LEFT, padx=5)

    def _show_placeholder(self):
        """Muestra mensaje placeholder cuando no hay video activo."""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        ttk.Label(
            self.image_frame,
            text="El video aparecerá aquí",
            foreground="white",
            background="gray"
        ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def start_recognition(self):
        """Inicia el generador de reconocimiento y actualiza frames."""
        if self.is_recognizing:
            return
        try:
            self.recognition_generator = recognition.recognize()
            self.is_recognizing = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self._update_frame()
        except FileNotFoundError as error:
            messagebox.showerror("Error", str(error))
        except Exception as error:
            messagebox.showerror("Error", f"Error inesperado: {error}")

    def stop_recognition(self):
        """Detiene el reconocimiento y libera recursos."""
        self.is_recognizing = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.recognition_generator:
            self.recognition_generator.close()
            self.recognition_generator = None
        self._show_placeholder()

    def _update_frame(self):
        """Callback periódico que obtiene y visualiza el siguiente frame."""
        if not self.is_recognizing:
            return
        try:
            frame = next(self.recognition_generator)
            frame = cv2.resize(frame, IMAGE_FRAME_SIZE)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convertir a imagen PIL y luego a PhotoImage para tkinter
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Limpiar y actualizar frame
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            label = ttk.Label(self.image_frame, image=tk_image)
            label.image = tk_image  # Mantener referencia
            label.pack(fill=tk.BOTH, expand=True)

            self.root.after(10, self._update_frame)
        except StopIteration:
            self.stop_recognition()

    def _open_add_face_dialog(self):
        """Abre la ventana de diálogo para agregar nuevos rostros."""
        dialog_window = tk.Toplevel(self.root)
        dialog_window.title("Agregar Rostro")
        dialog_window.resizable(False, False)

        # Panel de opciones de modo
        mode_frame = ttk.Frame(dialog_window)
        mode_frame.pack(pady=10)

        input_frame = ttk.Frame(dialog_window)
        input_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)

        ttk.Button(
            mode_frame, 
            text="Nuevo Rostro",
            command=lambda: self._setup_new_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            mode_frame, 
            text="Seleccionar Existente",
            command=lambda: self._setup_select_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)

        # Panel de acciones
        action_frame = ttk.Frame(dialog_window)
        action_frame.pack(pady=10)

        ttk.Button(
            action_frame, 
            text="Capturar y Entrenar",
            command=lambda: self._capture_and_train_model(dialog_window)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="Cancelar",
            command=dialog_window.destroy
        ).pack(side=tk.LEFT, padx=5)

    def _setup_new_face_mode(self, parent_frame: ttk.Frame):
        """Configura UI para capturar un rostro nuevo."""
        # Limpiar frame
        for widget in parent_frame.winfo_children():
            widget.destroy()

        ttk.Label(parent_frame, text="Nombre de la persona:").pack(pady=5)
        name_entry = ttk.Entry(parent_frame)
        name_entry.pack(pady=5)
        status_label = ttk.Label(parent_frame, text="")
        status_label.pack(pady=5)

        def validate_name(_event=None):
            name = name_entry.get().strip()
            if not name:
                status_label.config(text="Nombre vacío", foreground="red")
                self.selected_name = None
            elif name in capture.get_people_list():
                status_label.config(text="Ya existe — se añadirán imágenes", foreground="orange")
                self.selected_name = name
            else:
                status_label.config(text="Valido", foreground="green")
                self.selected_name = name

        name_entry.bind("<KeyRelease>", validate_name)

    def _setup_select_face_mode(self, parent_frame: ttk.Frame):
        """Configura UI para seleccionar un rostro existente."""
        # Limpiar frame
        for widget in parent_frame.winfo_children():
            widget.destroy()

        ttk.Label(parent_frame, text="Rostros registrados:").pack(pady=5)
        existing_faces = capture.get_people_list()
        
        if not existing_faces:
            ttk.Label(
                parent_frame, 
                text="No hay rostros registrados",
                foreground="red"
            ).pack(pady=5)
            return

        # Listbox de rostros
        faces_listbox = tk.Listbox(parent_frame, height=5, width=24)
        for face_name in existing_faces:
            faces_listbox.insert(tk.END, face_name)
        faces_listbox.pack(pady=5)

        status_label = ttk.Label(parent_frame, text="")
        status_label.pack(pady=5)

        def on_selection_changed(_event=None):
            selection = faces_listbox.curselection()
            if selection:
                self.selected_name = faces_listbox.get(selection[0])
                status_label.config(
                    text=f"{self.selected_name}", 
                    foreground="green"
                )

        faces_listbox.bind("<<ListboxSelect>>", on_selection_changed)

    def _capture_and_train_model(self, dialog_window: tk.Toplevel):
        """Ejecuta captura de imágenes y entrenamiento en thread separado."""
        if not self.selected_name:
            messagebox.showwarning(
                "Advertencia", "Por favor selecciona o ingresa un nombre"
            )
            return

        selected_person = self.selected_name
        dialog_window.destroy()

        def task():
            """Tarea de fondo para captura y entrenamiento."""
            try:
                capture.capture_faces(selected_person)
                train.train_recognizer()
                self.root.after(0, lambda: messagebox.showinfo(
                    "Éxito", 
                    f"Rostro de '{selected_person}' capturado y modelo entrenado (OK)"
                ))
            except Exception as error:
                error_message = str(error)
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", 
                    f"Error durante captura/entrenamiento:\n{error_message}"
                ))

        # Ejecutar en thread daemon para no bloquear UI
        background_thread = threading.Thread(target=task, daemon=True)
        background_thread.start()

    def _on_close(self):
        """Libera recursos antes de cerrar la aplicación."""
        self.is_recognizing = False
        if self.recognition_generator:
            self.recognition_generator.close()
        self.root.destroy()


def main():
    """Punto de entrada principal de la aplicación."""
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()