import subprocess
import matplotlib.pyplot as plt
import sys
import numpy as np
    
# Obtén la ruta del intérprete de Python del entorno virtual
python_executable = sys.executable

scripts = ["test_cnn.py", "test_data_augmentation.py", "test_knn.py"]

for script in scripts:
    try:
        subprocess.run([python_executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar {script}: {e}")
        sys.exit(1)

# Continuar con el análisis si todos los scripts se ejecutan correctamente

# Leer métricas desde el archivo model_metrics.txt
metrics_file = 'model_metrics.txt'
models = []
precision = []
recall = []
accuracy = []
f1_score = []

with open(metrics_file, 'r') as file:
    for line in file:
        # Suponiendo que el archivo tiene el formato: Modelo, Precision, Recall, Accuracy, F1-Score
        parts = line.strip().split(',')
        if len(parts) == 5:
            models.append(parts[0].strip())
            precision.append(float(parts[1].strip()))
            recall.append(float(parts[2].strip()))
            accuracy.append(float(parts[3].strip()))
            f1_score.append(float(parts[4].strip()))

# Crear gráficas comparativas
# Gráfica de precisión
plt.figure(figsize=(8, 6))
plt.bar(models, precision, color=['blue', 'green', 'orange'])
plt.title('Comparación de Precisión')
plt.ylabel('Precisión')
plt.ylim(0, 1)
plt.savefig('precision_comparativa.png')
plt.pause(2)
plt.close()

# Gráfica de recall
plt.figure(figsize=(8, 6))
plt.bar(models, recall, color=['blue', 'green', 'orange'])
plt.title('Comparación de Recall')
plt.ylabel('Recall')
plt.ylim(0, 1)
plt.savefig('recall_comparativa.png')
plt.pause(2)
plt.close()

# Gráfica de accuracy
plt.figure(figsize=(8, 6))
plt.bar(models, accuracy, color=['blue', 'green', 'orange'])
plt.title('Comparación de Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('accuracy_comparativa.png')
plt.pause(2)
plt.close()

# Gráfica de F1-Score
plt.figure(figsize=(8, 6))
plt.bar(models, f1_score, color=['blue', 'green', 'orange'])
plt.title('Comparación de F1-Score')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.savefig('f1_score_comparativa.png')
plt.pause(2)
plt.close()

print("Gráficas generadas y guardadas como 'precision_comparativa.png', 'recall_comparativa.png', 'accuracy_comparativa.png' y 'f1_score_comparativa.png'.")