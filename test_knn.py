from knn import predict_folder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from save_metrics import save_metrics_csv, format_metrics

# Función para mostrar la matriz de confusión y las métricas de evaluación
def show_confusion_matrix_from_dicts(pred_benign_dict, pred_malign_dict, class_labels, model_name, show_metrics=True):
    # Construir listas de clases verdaderas (y_true) y predichas (y_pred)
    y_true = []
    y_pred = []

    # Procesar las predicciones de imágenes benignas
    y_true += ["benign"] * sum(pred_benign_dict.values())  # Añadir "benign" tantas veces como la cantidad de predicciones benignas
    y_pred += (["benign"] * pred_benign_dict["benign"]) + (["malignant"] * pred_benign_dict["malignant"])

    # Procesar las predicciones de imágenes malignas
    y_true += ["malignant"] * sum(pred_malign_dict.values())  # Añadir "malignant" tantas veces como la cantidad de predicciones malignas
    y_pred += (["benign"] * pred_malign_dict["benign"]) + (["malignant"] * pred_malign_dict["malignant"])

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels.keys()))

    # Crear figura con dos subgráficas: una para la matriz de confusión y otra para el texto con las métricas
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Mostrar la matriz de confusión en la primera subgráfica (ax_matrix)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  # `annot=True` para mostrar los valores en cada celda de la matriz
                xticklabels=class_labels.values(),  # Etiquetas de las clases en el eje X
                yticklabels=class_labels.values(),  # Etiquetas de las clases en el eje Y
                ax=ax_matrix)

    # Etiquetas para los ejes y título de la matriz de confusión
    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    # Calcular las métricas de evaluación
    acc = accuracy_score(y_true, y_pred)  # Exactitud
    f1 = f1_score(y_true, y_pred, pos_label="malignant", average='binary')  # F1 Score, para clase "malignant"
    prec = precision_score(y_true, y_pred, pos_label="malignant", average='binary')  # Precisión
    recall = recall_score(y_true, y_pred, pos_label="malignant", average='binary')  # Recall

    # Almacenar las métricas en un diccionario
    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }

    # Convertir las métricas a texto para mostrarlas
    metrics_text = format_metrics(metrics)
    
    # Mostrar las métricas en la segunda subgráfica (ax_text)
    ax_text.axis('off')  # Ocultar los ejes
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))  # Mostrar el texto con las métricas
    
    # Si `show_metrics` es True, mostramos las gráficas
    if(show_metrics):
        plt.tight_layout()  # Ajusta el diseño para que no haya superposición
        plt.show()

    # Guardar las métricas en un archivo CSV
    save_metrics_csv(model_name, metrics)


# Bloque principal para ejecutar el código
if __name__ == "__main__":
    show_metrics = False  # Cambia a True si deseas mostrar las gráficas y métricas

    # Iterar sobre diferentes valores de magnificación
    for magnificient in [40, 100, 200, 400, None]:
        if magnificient is not None:
            # Si hay magnificación, establecer las rutas a las carpetas de imágenes de benignos y malignos
            folder_path_benign = f"images\\binary_scenario\\test\\{magnificient}x\\benign"
            folder_path_malignant = f"images\\binary_scenario\\test\\{magnificient}x\\malignant"

            # Realizar las predicciones para las imágenes benignas y malignas
            predict_folder_benign = predict_folder(folder_path_benign, f"{magnificient}x")
            predict_folder_malignant = predict_folder(folder_path_malignant, f"{magnificient}x")
            
            # Nombre del modelo con la magnificación especificada
            model_name = f"modelo_knn_{magnificient}x"
        else: 
            # Si no hay magnificación, usar un conjunto de datos combinado
            folder_path_benign = f"images\\binary_scenario_merged\\test\\benign"
            folder_path_malignant = f"images\\binary_scenario_merged\\test\\malignant"
        
            # Realizar las predicciones para las imágenes benignas y malignas sin magnificación
            predict_folder_benign = predict_folder(folder_path_benign)
            predict_folder_malignant = predict_folder(folder_path_malignant)
            
            # Nombre del modelo para el conjunto de datos combinado
            model_name = "modelo_knn_all"

        # Etiquetas de las clases para la visualización
        class_labels = {"benign": "Benigno", "malignant": "Maligno"}

        # Mostrar la matriz de confusión y guardar las métricas
        show_confusion_matrix_from_dicts(predict_folder_benign, predict_folder_malignant, class_labels, model_name, show_metrics=show_metrics)
