import subprocess
import matplotlib.pyplot as plt
import sys
import pandas as pd
import re

EXECUTE_SCRIPTS = False

if(EXECUTE_SCRIPTS):
    # Obtén la ruta del intérprete de Python del entorno virtual
    python_executable = sys.executable

    scripts = ["test_cnn.py", "test_cnn_augmentation.py", "test_knn.py", "test_knn_augmentation.py"]

    for script in scripts:
        try:
            subprocess.run([python_executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar {script}: {e}")
            sys.exit(1)

# Continuar con el análisis si todos los scripts se ejecutan correctamente

# Leer métricas desde CSV
df = pd.read_csv('metricas.csv')

# Extraer categoría (40x, 100x, etc.) desde el nombre del archivo
def extraer_categoria(nombre):
    match = re.search(r'(\d+x|all)', nombre)
    return match.group(1) if match else "otro"

df["grupo"] = df["archivo"].apply(extraer_categoria)

# Clasificar modelo más detalladamente
def identificar_modelo(nombre):
    if "cnn_aug" in nombre:
        return "CNN_AUG"
    elif "cnn" in nombre:
        return "CNN"
    elif "knn_aug" in nombre:
        return "KNN_AUG"
    elif "knn" in nombre:
        return "KNN"
    else:
        return "Otro"
    
# Identificar tipo de modelo
df["modelo"] = df["archivo"].apply(identificar_modelo)

# Lista de grupos ordenados
orden_grupos = ["40x", "100x", "200x", "400x", "all"]

# Asegurar orden lógico
df["grupo"] = pd.Categorical(df["grupo"], categories=orden_grupos, ordered=True)
df = df.sort_values("grupo")

# Métricas a graficar
metricas = {
    "Accuracy": "Accuracy",
    "F1 Score": "F1 Score",
    "Precision": "Precision",
    "Recall": "Recall"
}

# Colores y marcadores por modelo
colores = {
    "CNN": "#1f77b4",        # Azul intenso
    "CNN_AUG": "#ff7f0e",    # Naranja
    "KNN": "#2ca02c",        # Verde
    "KNN_AUG": "#d62728"     # Rojo
}
marcadores = {
    "CNN": "o",       # Círculo
    "CNN_AUG": "^",   # Triángulo hacia arriba
    "KNN": "s",       # Cuadrado
    "KNN_AUG": "D"    # Diamante
}

# Crear una gráfica para cada métrica
for clave, titulo in metricas.items():
    plt.figure(figsize=(10, 6))

    for modelo in df["modelo"].unique():
        subset = df[df["modelo"] == modelo]
        plt.scatter(
            subset["grupo"], 
            subset[clave], 
            label=modelo, 
            color=colores[modelo], 
            marker=marcadores[modelo], 
            s=100  # tamaño de los puntos
        )

    plt.title(f'Comparación de {titulo} por tamaño de conjunto')
    plt.ylabel(titulo)
    plt.xlabel('Grupo (Tamaño de dataset)')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./outputs/metrics/comparation_metrics/{clave.lower().replace(" ", "_")}_comparativa_puntos.png')
    plt.close()

print("Gráficas de puntos generadas por grupo.")