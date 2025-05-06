import pandas as pd              # Importa pandas para la manipulación de datos
import os                        # Importa os para manejar archivos y rutas

# Función que formatea un diccionario de métricas en un string
def format_metrics(metrics: dict):
    """
    Convierte un diccionario de métricas a una cadena de texto en formato de clave: valor.
    """
    return "\n".join(f"{key}: {value}" for key, value in metrics.items())  # Une las claves y valores con salto de línea

# Función que guarda las métricas de un modelo en un archivo CSV
def save_metrics_csv(model_name: str, metrics: dict, output_paht="metrics.csv"):
    """
    Guarda las métricas de un modelo en un archivo CSV. Si el archivo ya existe, actualiza las métricas
    para el modelo dado, o agrega una nueva entrada si no existe.

    Parámetros:
    - model_name: Nombre del modelo para usar como identificador.
    - metrics: Diccionario de métricas a guardar.
    - output_paht: Ruta del archivo CSV donde se guardarán las métricas. Por defecto es "metrics.csv".
    """
    
    # Verifica si el archivo CSV ya existe
    if os.path.exists(output_paht):
        df = pd.read_csv(output_paht)  # Lee el archivo CSV existente
    else:
        # Si no existe, crea un nuevo DataFrame con las columnas adecuadas
        columnas = ["archivo"] + list(metrics.keys())
        df = pd.DataFrame(columns=columnas)

    # Elimina cualquier fila con el modelo actual, para asegurarse de que se actualiza la información
    df = df[df["archivo"] != model_name]

    # Crea una nueva fila con el nombre del modelo y sus métricas
    nueva_fila = {"archivo": model_name, **metrics}
    # Concatenar la nueva fila al DataFrame
    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)

    # Ordena el DataFrame por el nombre del modelo y reinicia los índices
    df = df.sort_values("archivo").reset_index(drop=True)
    
    # Guarda el DataFrame actualizado en el archivo CSV
    df.to_csv(output_paht, index=False)
