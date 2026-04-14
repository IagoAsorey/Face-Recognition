"""
Interfaz gráfica para el sistema de reconocimiento facial.
Permite capturar nuevos rostros, entrenar el modelo y realizar reconocimiento en tiempo real.
"""
import threading
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import ttk, messagebox

from config import IMAGE_FRAME_SIZE
import faceCapture
import training
import faceRecognition


class FaceRecognitionApp:
    """Aplicación GUI para reconocimiento facial."""
    
    def __init__(self, root):
        """
        Inicializa la aplicación.
        
        Args:
            root (tk.Tk): Ventana principal de Tkinter.
        """
        self.root = root
        self.root.title("Face Recognition System")
        self.selected_name = None
        self.is_recognizing = False
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de usuario."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para video
        self.image_frame = tk.Frame(
            main_frame, 
            width=IMAGE_FRAME_SIZE[0], 
            height=IMAGE_FRAME_SIZE[1], 
            bg="gray"
        )
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.placeholder_label = ttk.Label(
            self.image_frame, 
            text="El video aparecerá aquí", 
            foreground="white", 
            background="gray"
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Frame para botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)
        
        ttk.Button(
            button_frame, 
            text="Iniciar Reconocimiento",
            command=self.start_recognition
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Agregar Rostro",
            command=self.add_face_window
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Salir",
            command=self.root.destroy
        ).pack(side=tk.LEFT, padx=5)
    
    def start_recognition(self):
        """Inicia el reconocimiento facial en tiempo real."""
        if self.is_recognizing:
            messagebox.showinfo("Info", "Ya está activo el reconocimiento")
            return
        
        try:
            self.is_recognizing = True
            self._update_frame()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.is_recognizing = False
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.is_recognizing = False
    
    def _update_frame(self):
        """Actualiza el frame de video."""
        try:
            if not self.is_recognizing:
                return
            
            frame = next(self.gen)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            video_label = ttk.Label(self.image_frame, image=tk_img)
            video_label.image = tk_img
            video_label.pack(fill=tk.BOTH, expand=True)
            
            self.root.after(15, self._update_frame)
        except StopIteration:
            self.is_recognizing = False
    
    def add_face_window(self):
        """Abre ventana para agregar un nuevo rostro."""
        window = tk.Toplevel(self.root)
        window.title("Agregar Rostro")
        
        # Frame de opciones
        option_frame = ttk.Frame(window)
        option_frame.pack(pady=10)
        
        ttk.Button(
            option_frame, 
            text="Nuevo Rostro",
            command=lambda: self._new_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            option_frame, 
            text="Seleccionar Existente",
            command=lambda: self._select_face_mode(input_frame)
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame para entrada
        input_frame = ttk.Frame(window)
        input_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame de acciones
        action_frame = ttk.Frame(window)
        action_frame.pack(pady=10)
        
        ttk.Button(
            action_frame,
            text="Capturar y Entrenar",
            command=lambda: self._capture_and_train(window)
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame,
            text="Cancelar",
            command=window.destroy
        ).pack(side=tk.LEFT, padx=5)
    
    def _new_face_mode(self, parent):
        """Permite crear un nuevo rostro."""
        for widget in parent.winfo_children():
            widget.destroy()
        
        ttk.Label(parent, text="Nombre de la persona:").pack(pady=5)
        
        entry = ttk.Entry(parent)
        entry.pack(pady=5)
        
        status_label = ttk.Label(parent, text="")
        status_label.pack(pady=5)
        
        def validate(event=None):
            name = entry.get().strip()
            existing = faceCapture.faces()
            
            if not name:
                status_label.config(text="Nombre vacío", foreground="red")
                self.selected_name = None
            elif name in existing:
                status_label.config(text="Ya existe", foreground="orange")
                self.selected_name = name
            else:
                status_label.config(text="✓ Válido", foreground="green")
                self.selected_name = name
        
        entry.bind("<KeyRelease>", validate)
    
    def _select_face_mode(self, parent):
        """Permite seleccionar un rostro existente."""
        for widget in parent.winfo_children():
            widget.destroy()
        
        ttk.Label(parent, text="Rostros registrados:").pack(pady=5)
        
        faces = faceCapture.faces()
        if not faces:
            ttk.Label(parent, text="No hay rostros registrados", foreground="red").pack(pady=5)
            return
        
        listbox = tk.Listbox(parent, height=5, width=20)
        for face in faces:
            listbox.insert(tk.END, face)
        listbox.pack(pady=5)
        
        status_label = ttk.Label(parent, text="")
        status_label.pack(pady=5)
        
        def on_select(event=None):
            selection = listbox.curselection()
            if selection:
                self.selected_name = listbox.get(selection[0])
                status_label.config(text=f"✓ Seleccionado: {self.selected_name}", foreground="green")
        
        listbox.bind("<<ListboxSelect>>", on_select)
    
    def _capture_and_train(self, window):
        """Captura imágenes y entrena el modelo."""
        if not self.selected_name:
            messagebox.showwarning("Advertencia", "Por favor selecciona o ingresa un nombre")
            return
        
        def task():
            try:
                faceCapture.capture_faces(self.selected_name)
                training.train_recognizer()
                messagebox.showinfo("Éxito", "Rostro capturado y modelo entrenado")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")
        
        threading.Thread(target=task, daemon=True).start()
        window.destroy()


def main():
    """Función principal."""
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()