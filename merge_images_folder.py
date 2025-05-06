import os        # Para manejar rutas y directorios
import shutil    # Para copiar archivos

# Función que fusiona subcarpetas en una estructura plana manteniendo la etiqueta (benign/malignant)
def merge_subfolders(source_root, target_root):
    """
    Toma una estructura de carpetas anidadas (por ejemplo, por aumentos como x40, x100) 
    y fusiona las imágenes en una única carpeta por clase ('benign', 'malignant'), 
    renombrando los archivos para evitar colisiones.

    Parámetros:
    - source_root: ruta raíz del conjunto de datos original, donde hay subcarpetas por magnificación.
    - target_root: ruta donde se guardará la estructura fusionada.
    """
    
    for subfolder in os.listdir(source_root):  # Recorre subcarpetas (ej: x40, x100)
        subfolder_path = os.path.join(source_root, subfolder)
        
        if os.path.isdir(subfolder_path):  # Verifica que sea una carpeta
            for label in os.listdir(subfolder_path):  # Recorre las etiquetas (benign, malignant)
                label_path = os.path.join(subfolder_path, label)

                # Crea la carpeta destino si no existe (por clase)
                target_label_path = os.path.join(target_root, label)
                os.makedirs(target_label_path, exist_ok=True)

                # Recorre todas las imágenes dentro de la clase
                for img_file in os.listdir(label_path):
                    src_file = os.path.join(label_path, img_file)  # Ruta de la imagen original

                    # Renombra la imagen incluyendo la magnificación para evitar duplicados
                    new_name = f"{subfolder}_{img_file}"
                    dst_file = os.path.join(target_label_path, new_name)

                    # Copia la imagen al nuevo destino con el nuevo nombre
                    shutil.copy2(src_file, dst_file)


# ---------------------- #
# Ejemplo de uso práctico:
# Fusiona las imágenes de entrenamiento y prueba en una nueva carpeta sin subniveles por magnificación

merge_subfolders("./images/binary_scenario/train", "./images/binary_scenario_merged/train")
merge_subfolders("./images/binary_scenario/test", "./images/binary_scenario_merged/test")
