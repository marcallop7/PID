import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
from knn import predict_file_by_path
import threading

class AnalizadorCancerDeMama:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de cáncer de mama")
        self.root.geometry("900x800")
        self.root.config(bg="#f4f4f9")
        
        # Colores y configuración
        self.bg_color = "#f4f4f9"
        self.button_color = "#6c7ae0"
        self.highlight_color = "#c5cae9"
        self.text_color = "#333333"
        self.error_color = "#ff4444"
        self.benign_color = "#4CAF50"
        self.malignant_color = "#F44336"
        
        self.img = None
        self.imagenes_paths = []
        self.resultados = {"Benignas": 0, "Malignas": 0}
        self.resultados_colores = {}  
        self.miniaturas = []
        self.current_image_index = 0
        self.analizando = False
        
        # Estilo
        self.style = ttk.Style()
        self.style.configure("TButton", padding=10, relief="flat", background=self.button_color, foreground="white", font=("Arial", 12))
        self.style.map("TButton", background=[("active", "#5a6ccf")])
        self.style.configure("TEntry", padding=10, relief="flat", background=self.highlight_color, font=("Arial", 12))
        
        # Frame principal
        self.frame_main = tk.Frame(self.root, bg=self.bg_color)
        self.frame_main.pack(pady=20, padx=30, fill="both", expand=True)
        
        # Botones de selección
        self.btn_seleccionar = ttk.Button(self.frame_main, text="Seleccionar Imágenes", command=self.seleccionar_imagenes, style="TButton")
        self.style.configure("TButton", foreground="black")
        self.btn_seleccionar.grid(row=0, column=0, pady=10, padx=20, sticky="w")
        
        self.btn_seleccionar_individual = ttk.Button(self.frame_main, text="Seleccionar 1 Imagen", command=self.seleccionar_imagen, style="TButton")
        self.btn_seleccionar_individual.grid(row=0, column=1, pady=10, padx=20, sticky="w")
        
        # Entrada para la ruta
        self.entry_ruta = ttk.Entry(self.frame_main, width=50)
        self.entry_ruta.grid(row=1, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
        
        # Zona de vista previa principal
        self.frame_preview = tk.Frame(self.frame_main, width=500, height=300, bg=self.highlight_color, relief="ridge", bd=2)
        self.frame_preview.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")
        
        self.label_preview = tk.Label(self.frame_preview, text="Arrastra una o más imágenes aquí\nO haz clic en 'Seleccionar', para elegir imágenes", 
                                    bg=self.highlight_color, fg=self.text_color, font=("Arial", 14))
        self.label_preview.pack(expand=True)
        
        # Canvas para el carrete de miniaturas
        self.frame_carrete = tk.Frame(self.frame_main, bg=self.bg_color)
        self.frame_carrete.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.canvas_carrete = tk.Canvas(self.frame_carrete, bg=self.bg_color, height=120)
        self.scrollbar = ttk.Scrollbar(self.frame_carrete, orient="horizontal", command=self.canvas_carrete.xview)
        
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_carrete.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.frame_miniaturas = tk.Frame(self.canvas_carrete, bg=self.bg_color)
        self.canvas_carrete.create_window((0, 0), window=self.frame_miniaturas, anchor="nw")
        
        # Configurar scroll
        self.frame_miniaturas.bind("<Configure>", lambda e: self.canvas_carrete.configure(scrollregion=self.canvas_carrete.bbox("all")))
        self.canvas_carrete.configure(xscrollcommand=self.scrollbar.set)
        
        # Configurar arrastrar y soltar
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.drop_images)

        
        # Botón para eliminar
        self.btn_eliminar = ttk.Button(self.frame_main, text="Eliminar Imágenes", command=self.eliminar_imagenes, style="TButton")
        self.btn_eliminar.grid_forget()
        
        # Estadísticas
        self.frame_stats = tk.Frame(self.frame_main, bg=self.bg_color)
        self.frame_stats.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.label_total = tk.Label(self.frame_stats, text="Total imágenes: 0", bg=self.bg_color, font=("Arial", 12))
        self.label_total.pack(side=tk.LEFT, padx=10)
        
        self.label_benignas = tk.Label(self.frame_stats, text="Benignas: 0", bg=self.bg_color, fg=self.benign_color, font=("Arial", 12))
        self.label_benignas.pack(side=tk.LEFT, padx=10)
        
        self.label_malignas = tk.Label(self.frame_stats, text="Malignas: 0", bg=self.bg_color, fg=self.malignant_color, font=("Arial", 12))
        self.label_malignas.pack(side=tk.LEFT, padx=10)
        
        
        # Resultado individual
        self.frame_resultado = tk.Frame(self.frame_main, bg=self.bg_color)
        self.frame_resultado.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.label_resultado = tk.Label(self.frame_resultado, text="Resultado: ", 
                                      bg=self.bg_color, fg=self.text_color, font=("Arial", 12))
        self.label_resultado.pack(side=tk.LEFT, padx=10)
        
        self.canvas_resultado = tk.Canvas(self.frame_resultado, width=30, height=30, bg=self.bg_color, bd=0, highlightthickness=0)
        self.canvas_resultado.pack(side=tk.LEFT)
        self.circulo = self.canvas_resultado.create_oval(5, 5, 25, 25, fill="lightgray")
        
        # Frame para controles de análisis
        self.frame_controles = tk.Frame(self.frame_main, bg=self.bg_color)
        self.frame_controles.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Botón de analizar
        self.btn_analizar = ttk.Button(self.frame_controles, text="Iniciar Análisis", command=self.iniciar_analisis)
        self.btn_analizar.pack(side=tk.LEFT, padx=5)
        
        # Botón para pausar/reanudar
        self.btn_pausar = ttk.Button(self.frame_controles, text="Pausar", command=self.pausar_analisis, state=tk.DISABLED)
        self.btn_pausar.pack(side=tk.LEFT, padx=5)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frame_controles, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Configurar el grid
        self.frame_main.grid_columnconfigure(1, weight=1)
        self.frame_main.grid_rowconfigure(2, weight=1)
    
    def seleccionar_imagen(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.limpiar_resultados()
            self.imagenes_paths = [file_path]
            self.actualizar_interfaz()
    
    def seleccionar_imagenes(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_paths:
            self.limpiar_resultados()
            self.imagenes_paths = list(file_paths)
            self.actualizar_interfaz()
    
    def drop_images(self, event):
        """Maneja el evento de arrastrar y soltar imágenes"""
        file_paths = self.root.tk.splitlist(event.data)  # Obtener rutas correctamente
        
        # Filtrar solo archivos de imagen
        file_paths = [f.strip("{}") for f in file_paths if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

        if file_paths:
            self.limpiar_resultados()
            self.imagenes_paths = file_paths
            self.actualizar_interfaz()
    
    def actualizar_interfaz(self):
        if not self.imagenes_paths:
            return
            
        self.entry_ruta.delete(0, tk.END)
        if len(self.imagenes_paths) == 1:
            self.entry_ruta.insert(0, self.imagenes_paths[0])
        else:
            self.entry_ruta.insert(0, f"{len(self.imagenes_paths)} imágenes seleccionadas")
        
        # Limpiar miniaturas anteriores
        for widget in self.frame_miniaturas.winfo_children():
            widget.destroy()
        self.miniaturas = []
        
        # Crear miniaturas para el carrete
        for i, path in enumerate(self.imagenes_paths):
            try:
                img = Image.open(path)
                img.thumbnail((100, 100))
                
                photo = ImageTk.PhotoImage(img)
                self.miniaturas.append(photo)
                
                frame = tk.Frame(self.frame_miniaturas, bg=self.bg_color, bd=2, relief="ridge")
                frame.pack(side=tk.LEFT, padx=5, pady=5)
                
                label = tk.Label(frame, image=photo, bg=self.bg_color)
                label.image = photo
                label.pack()
                
                filename = path.split("/")[-1][:15] + ("..." if len(path.split("/")[-1]) > 15 else "")
                tk.Label(frame, text=filename, bg=self.bg_color, font=("Arial", 8)).pack()
                
                label.bind("<Button-1>", lambda e, idx=i: self.mostrar_imagen_seleccionada(idx))
            except Exception as e:
                print(f"No se pudo cargar la miniatura {path}: {e}")
        
        if self.imagenes_paths:
            self.mostrar_imagen_seleccionada(0)
        
        self.btn_eliminar.grid(row=0, column=2, pady=10, padx=10)
        self.actualizar_estadisticas()

    def actualizar_resultado(self, resultado, color, index):
        """Actualiza el resultado y guarda tanto el color como el texto original"""
        self.canvas_resultado.itemconfig(self.circulo, fill=color)
        
        # Determinar el texto según el resultado original
        if "benign" in resultado.lower():
            resultado_texto = "Benigno"
        else:
            resultado_texto = "Maligno"
        
        self.label_resultado.config(text=f"Resultado: {resultado_texto}")
        self.actualizar_estadisticas()

        # Guardar tanto el color como el texto original del resultado
        self.resultados_colores[self.imagenes_paths[index]] = (color, resultado)

        # Resaltar la miniatura analizada
        for i, widget in enumerate(self.frame_miniaturas.winfo_children()):
            if i == index:
                widget.config(bg=color)
            elif widget.cget("bg") not in [self.benign_color, self.malignant_color]:
                widget.config(bg=self.bg_color)
    
    def actualizar_resultado(self, resultado, color, index):
        """Actualiza el resultado en la interfaz y guarda el color correspondiente"""
        self.canvas_resultado.itemconfig(self.circulo, fill=color)
        
        # Determinar el texto según el resultado original
        if "benign" in resultado.lower():
            resultado_texto = "Benigno"
        else:
            resultado_texto = "Maligno"
        
        # Actualizar el texto del resultado en la interfaz
        self.label_resultado.config(text=f"Resultado: {resultado_texto}")
        self.actualizar_estadisticas()

        # Guardar el color y el resultado asociado a la imagen analizada
        self.resultados_colores[self.imagenes_paths[index]] = (color, resultado)

        # Resaltar la miniatura analizada
        for i, widget in enumerate(self.frame_miniaturas.winfo_children()):
            if i == index:
                widget.config(bg=color)
            elif widget.cget("bg") not in [self.benign_color, self.malignant_color]:
                widget.config(bg=self.bg_color)
    
    def actualizar_progreso(self, current):
        """Actualiza la barra de progreso"""
        progress = (current / len(self.imagenes_paths)) * 100
        self.progress_var.set(progress)
    
    def finalizar_analisis(self):
        self.analizando = False
        self.btn_analizar.config(state=tk.NORMAL)
        self.btn_pausar.config(state=tk.DISABLED, text="Pausar")
        if self.current_image_index >= len(self.imagenes_paths):
            messagebox.showinfo("Análisis completado", "Se han analizado todas las imágenes")
        
    def actualizar_estadisticas(self):
        total = len(self.imagenes_paths)
        self.label_total.config(text=f"Total imágenes: {total}")
        self.label_benignas.config(text=f"Benignas: {self.resultados['Benignas']}")
        self.label_malignas.config(text=f"Malignas: {self.resultados['Malignas']}")
    
    
    def mostrar_imagen_seleccionada(self, index):
        """Muestra la imagen seleccionada y actualiza el resultado correctamente"""
        self.current_image_index = index
        try:
            path = self.imagenes_paths[index]
            image = Image.open(path)
            image.thumbnail((400, 400))
            
            self.img = ImageTk.PhotoImage(image)
            self.label_preview.config(image=self.img, text="")
            self.label_preview.image = self.img
            
            # Resaltar la miniatura seleccionada
            for i, widget in enumerate(self.frame_miniaturas.winfo_children()):
                if i == index:
                    widget.config(bg="#6c7ae0")  # Resaltar seleccionado
                else:
                    widget.config(bg=self.bg_color)  # Resto normal
            
            # Si la imagen ya fue analizada, actualizar el círculo y el texto
            if path in self.resultados_colores:
                color, texto_original = self.resultados_colores[path]  # Ahora guardamos ambos
                self.canvas_resultado.itemconfig(self.circulo, fill=color)
                
                # Extraer el texto descriptivo del resultado original
                if "benign" in texto_original.lower():
                    resultado_texto = "Benigno"
                else:
                    resultado_texto = "Maligno"
                
                self.label_resultado.config(text=f"Resultado: {resultado_texto}")
            else:
                # Si no ha sido analizada, mostrar estado neutral
                self.canvas_resultado.itemconfig(self.circulo, fill="lightgray")
                self.label_resultado.config(text="Resultado: No analizado")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen\n{e}")

    
    
    def eliminar_imagenes(self):
        """Elimina todas las imágenes cargadas y limpia la interfaz"""
        # Limpiar la lista de rutas de imágenes
        self.imagenes_paths.clear()

        # Reiniciar los resultados
        self.resultados = {"Benignas": 0, "Malignas": 0}
        self.resultados_colores.clear()
        self.label_resultado.config(text="Resultado: ")
        
        # Eliminar todas las miniaturas del frame
        for widget in self.frame_miniaturas.winfo_children():
            widget.destroy()
        self.miniaturas.clear()

        # Restablecer la vista previa
        self.label_preview.config(image="", text="Arrastra una o más imágenes aquí\nO haz clic en 'Seleccionar', para elegir imágenes")
        self.label_preview.image = None
        self.img = None  # Asegurar que la imagen previa se elimina
        self.current_image_index = 0  # Reiniciar índice de imágenes

        # Limpiar la entrada de ruta
        self.entry_ruta.delete(0, tk.END)

        # Ocultar el botón de eliminar
        self.btn_eliminar.grid_forget()

        # Restablecer las estadísticas
        self.actualizar_estadisticas()

        # Restablecer el resultado individual
        self.canvas_resultado.itemconfig(self.circulo, fill="lightgray")
        self.label_resultado.config(text="Resultado: ")

        # Restablecer la barra de progreso
        self.progress_var.set(0)

    
    def iniciar_analisis(self):
        if not self.imagenes_paths:
            messagebox.showwarning("Advertencia", "No se han seleccionado imágenes para analizar")
            return
        
        self.limpiar_resultados()

        self.analizando = True
        # Resetear solo si no estamos en medio de un análisis pausado
        if self.current_image_index >= len(self.imagenes_paths):
            self.current_image_index = 0
            self.resultados = {"Benignas": 0, "Malignas": 0}
        
        self.btn_analizar.config(state=tk.DISABLED)
        self.btn_pausar.config(state=tk.NORMAL, text="Pausar")
        threading.Thread(target=self.analizar_secuencial, daemon=True).start()
    
    def limpiar_resultados(self):
        """Borra todos los resultados y restablece la interfaz"""
        self.resultados = {"Benignas": 0, "Malignas": 0}
        self.resultados_colores.clear()
        
        # Restablecer el círculo de resultado individual
        self.canvas_resultado.itemconfig(self.circulo, fill="lightgray")
        self.label_resultado.config(text="Resultado: ")

        # Restablecer estadísticas
        self.actualizar_estadisticas()

        # Reiniciar barra de progreso
        self.progress_var.set(0)

        # Quitar colores de las miniaturas
        for widget in self.frame_miniaturas.winfo_children():
            widget.config(bg=self.bg_color)


    def pausar_analisis(self):
        """Pausa o reanuda el análisis - Versión corregida"""
        if self.analizando:
            # Si está analizando, pausar
            self.analizando = False
            self.btn_pausar.config(text="Reanudar")
        else:
            # Si está pausado, reanudar
            self.analizando = True
            self.btn_pausar.config(text="Pausar")
            # Solo iniciamos un nuevo hilo si no hay uno activo
            if self.current_image_index < len(self.imagenes_paths):
                threading.Thread(target=self.analizar_secuencial, daemon=True).start()
        
    
    def analizar_secuencial(self):
        """Analiza las imágenes una por una con control de pausa"""
        while self.current_image_index < len(self.imagenes_paths):
            if not self.analizando:
                # Pausa activada, salir del bucle
                return
                
            path = self.imagenes_paths[self.current_image_index]
            
            # Actualizar interfaz
            self.root.after(0, lambda idx=self.current_image_index: self.mostrar_imagen_seleccionada(idx))
            
            try:
                # Analizar imagen
                resultado, color = predict_file_by_path(path)
                
                # Actualizar resultados
                if "benign" in resultado.lower():
                    self.resultados["Benignas"] += 1
                else:
                    self.resultados["Malignas"] += 1
                
                # Actualizar interfaz
                self.root.after(0, self.actualizar_resultado, resultado, color, self.current_image_index)
                
                time.sleep(0.5)  # Pequeña pausa para visualización
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"No se pudo analizar la imagen {path}\n{e}"))
            
            # Actualizar progreso
            self.current_image_index += 1
            self.root.after(0, self.actualizar_progreso, self.current_image_index)
        
        # Análisis completado
        self.root.after(0, self.finalizar_analisis)
    
    
    def analizar_imagen(self):
        if not self.imagenes_paths:
            messagebox.showwarning("Advertencia", "No se ha seleccionado ninguna imagen")
            return
        
        resultado, color = self.analize(self.imagenes_paths[0])
        self.canvas_resultado.itemconfig(self.circulo, fill=color)
        
        if "benign" in resultado.lower():
            resultado_texto = "Benigno"
        else:
            resultado_texto = "Maligno"
        
        self.label_resultado.config(text=f"Resultado: {resultado_texto}")
        self.resultados_colores[self.imagenes_paths[0]] = (color, resultado)
        
        if sum(self.resultados.values()) == 0:
            if "benign" in resultado.lower():
                self.resultados["Benignas"] = 1
            else:
                self.resultados["Malignas"] = 1
            self.actualizar_estadisticas()
    
    def analize(self, image_path):
        if not image_path:
            return "No se ha seleccionado ninguna imagen", self.error_color
        
        result, color = predict_file_by_path(image_path)
        
        if "benign" in result.lower():
            color = self.benign_color
        else:
            color = self.malignant_color
        
        return result, color


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AnalizadorCancerDeMama(root)
    root.mainloop() 