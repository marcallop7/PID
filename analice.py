import subprocess
import sys

# Obtén la ruta del intérprete de Python del entorno virtual
python_executable = sys.executable

scripts = ["test_cnn.py", "test_cnn_augmentation.py", "test_knn.py", "test_knn_augmentation.py"]

# Se ejecutan todos los scripts relacionados a test para de esta forma obtener todos los resultados de métricas
for script in scripts:
    try:
        subprocess.run([python_executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar {script}: {e}")
        sys.exit(1)