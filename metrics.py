import pandas as pd

import matplotlib.pyplot as plt

# Cargar el archivo CSV
file_path = 'metricas.csv'  # Cambia el nombre si es necesario
data = pd.read_csv(file_path)

# Crear gráficas comparativas
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['archivo'], data[metric], color='skyblue')
    plt.title(f'Comparación de {metric} entre modelos')
    plt.xlabel('Modelos')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Agregar los valores exactos sobre las barras
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'comparison_{metric.lower()}.png')  # Guardar la gráfica como archivo
    plt.show()

# Filtrar los modelos por tipo de augmentation
augmentations = ['_all', '_40X', '_100X', '_200X', '_400X']
for aug in augmentations:
    plt.figure(figsize=(10, 6))
    filtered_data = data[data['archivo'].str.contains(aug, case=False, na=False)]
    
    bar_width = 0.2
    x = range(len(filtered_data['archivo']))
    
    for i, metric in enumerate(metrics):
        bars = plt.bar(
            [pos + i * bar_width for pos in x],
            filtered_data[metric],
            bar_width,
            label=metric
        )
        
        # Agregar los valores exactos sobre las barras
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{bar.get_height():.2f}', ha='center', va='bottom')
    
    plt.title(f'Comparación de métricas para modelos con {aug}')
    plt.xlabel('Modelos')
    plt.ylabel('Valor de la métrica')
    plt.ylim(0, 1)
    plt.xticks([pos + (len(metrics) - 1) * bar_width / 2 for pos in x], filtered_data['archivo'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'comparison_{aug}.png')  # Guardar la gráfica como archivo
    plt.show()
    # Imprimir los valores de las métricas para los modelos filtrados
    print(f"Valores de métricas para modelos con {aug}:")
    print(filtered_data[['archivo'] + metrics])


print("Gráficas generadas y guardadas como archivos PNG.")