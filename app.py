import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

class AnalizadorCancerDeMama:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de cáncer de mama")
        self.root.geometry("600x500")
        self.root.config(bg="#f4f4f9")
        
        # Definir colores personalizados para el tema
        self.bg_color = "#f4f4f9"
        self.button_color = "#6c7ae0"
        self.highlight_color = "#c5cae9"
        
        self.img = None
        self.imagen_path = None
        
        # Zona de arrastre
        self.frame_main = tk.Frame(self.root, bg=self.bg_color)
        self.frame_main.pack(pady=20, padx=30)
        
        # Botón de selección
        self.btn_seleccionar = tk.Button(self.frame_main, text="Seleccionar Imagen", command=self.seleccionar_imagen, bg=self.button_color, fg="white", relief="flat")
        self.btn_seleccionar.grid(row=0, column=0, pady=10, padx=20)
        
        # Entrada para la ruta de la imagen
        self.entry_ruta = tk.Entry(self.frame_main, width=50, relief="flat", bg=self.highlight_color)
        self.entry_ruta.grid(row=0, column=1, pady=10)
        
        # Zona de arrastrar/seleccionar imagen
        self.frame_drop = tk.Frame(self.frame_main, width=500, height=500, bg="lightgray", relief="ridge", bd=2)
        self.frame_drop.grid(row=1, column=0, columnspan=3, pady=10)
        self.frame_drop.drop_target_register(DND_FILES)
        self.frame_drop.dnd_bind("<<Drop>>", self.drop_image)
        
        # Etiqueta para el mensaje de arrastre
        self.label_drop = tk.Label(self.frame_drop, text="Arrastra una imagen aquí\nO haz clic en 'Seleccionar'", bg="lightgray")
        self.label_drop.pack(expand=True)
        
        # Vista previa de la imagen (se mostrará sobre la zona de drop)
        self.label_imagen = tk.Label(self.frame_main)
        
        # Botón para eliminar la imagen
        self.btn_eliminar = tk.Button(self.frame_main, text="Eliminar Imagen", command=self.eliminar_imagen, bg="red", fg="white", relief="flat")
        self.btn_eliminar.grid_forget()  # Inicialmente está oculto
        
        # Crear un círculo (se actualizará según el análisis)
        self.canvas_resultado = tk.Canvas(self.frame_main, width=50, height=50, bg=self.bg_color, bd=0, highlightthickness=0)
        self.canvas_resultado.grid(row=3, column=2, pady=10)
        self.circulo = self.canvas_resultado.create_oval(5, 5, 35, 35, fill="lightgray")
        
        # Etiqueta para mostrar el resultado del análisis
        self.label_resultado = tk.Label(self.frame_main, text="Resultado de la imagen", bg=self.bg_color, font=("Arial", 12))
        self.label_resultado.grid(row=3, column=1, pady=10)
        
        # Botón de analizar
        self.btn_analizar = tk.Button(self.frame_main, text="Analizar", command=self.analizar_imagen, bg=self.button_color, fg="white", relief="flat")
        self.btn_analizar.grid(row=3, column=0, pady=20)
        
    def seleccionar_imagen(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.cargar_imagen(file_path)
    
    def cargar_imagen(self, file_path):
        self.imagen_path = file_path
        self.entry_ruta.delete(0, tk.END)
        self.entry_ruta.insert(0, file_path)
        
        try:
            image = Image.open(file_path)
            image.thumbnail((200, 200))  # Redimensionamos la imagen
            self.img = ImageTk.PhotoImage(image)
            
            # Mostrar la imagen encima de la zona de arrastre
            self.label_imagen.config(image=self.img)
            self.label_imagen.grid(row=1, column=0, columnspan=3, pady=10)
            
            # Ocultar el mensaje de arrastre
            self.frame_drop.grid_forget()
            
            # Mostrar el botón para eliminar la imagen
            self.btn_eliminar.grid(row=0, column=0, pady=10)
            self.btn_seleccionar.grid_forget()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen\n{e}")
    
    def eliminar_imagen(self):
        self.imagen_path = None
        self.label_imagen.config(image='')  # Eliminar la imagen
        self.label_imagen.grid_forget()  # Eliminar la etiqueta de la imagen
        self.entry_ruta.delete(0, tk.END)  # Limpiar el campo de la ruta
        
        # Volver a mostrar el mensaje de arrastrar
        self.frame_drop.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.btn_eliminar.grid_forget()  # Ocultar el botón de eliminar
        self.btn_seleccionar.grid(row=0, column=0, pady=10)
    
    def analizar_imagen(self):
        image_path = self.entry_ruta.get()
        resultado, color = self.analize(image_path)
        
        # Cambiar el color del círculo según el resultado
        self.canvas_resultado.itemconfig(self.circulo, fill=color)  # Cambiar el color del círculo
        self.label_resultado.config(text=resultado)  # Cambiar el texto del resultado
    
    def analize(self, image_path):
        """Petición a la IA para análisis de la imagen"""
        result, color = "Mensaje de ejemplo", "white"
        if not image_path:
            return "No se ha seleccionado ninguna imagen", color
        # result, color = analisis_ia(image_path)
        return result, color
    
    def drop_image(self, event):
        """Permite arrastrar y soltar una imagen en la zona"""
        file_path = event.data.strip("{}")  # Elimina caracteres extras en Windows
        self.cargar_imagen(file_path)


if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Habilita arrastrar y soltar
    app = AnalizadorCancerDeMama(root)
    root.mainloop()