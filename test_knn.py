from knn import predict_folder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def save_metrics_to_file(file_path, model_name, accuracy, precision, recall, f1_score):
    with open(file_path, "a") as f:
        f.write(f"{model_name}\t{accuracy}\t{precision}\t{recall}\t{f1_score}\n")

def show_confusion_matrix_from_dicts(pred_benign_dict, pred_malign_dict, class_labels, model_name):
    # Construir listas de clases verdaderas y predichas
    y_true = []
    y_pred = []

    # Procesar predicciones para benignos
    y_true += ["benign"] * sum(pred_benign_dict.values())
    y_pred += (["benign"] * pred_benign_dict["benign"]) + (["malignant"] * pred_benign_dict["malignant"])

    # Procesar predicciones para malignos
    y_true += ["malignant"] * sum(pred_malign_dict.values())
    y_pred += (["benign"] * pred_malign_dict["benign"]) + (["malignant"] * pred_malign_dict["malignant"])

    # Obtener matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels.keys()))

    # Crear figura con dos columnas: matriz y texto
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Mostrar matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.values(),
                yticklabels=class_labels.values(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="malignant", average='binary')
    prec = precision_score(y_true, y_pred, pos_label="malignant", average='binary')
    recall = recall_score(y_true, y_pred, pos_label="malignant", average='binary')

    # Mostrar métricas en el segundo subplot
    metrics_text = (f"Accuracy: {acc}\n"
                    f"F1 Score: {f1}\n"
                    f"Precision: {prec}\n"
                    f"Recall: {recall}")
    
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))

    plt.tight_layout()
    plt.show()

    # Guardar métricas en el archivo
    save_metrics_to_file("model_metrics.txt", model_name, acc, prec, recall, f1)


if __name__ == "__main__":
    for magnificient in [40, 100, 200, 400, None]:
        if magnificient is not None:
            folder_path_benign = f"images\\binary_scenario\\test\\{magnificient}x\\benign"
            folder_path_malignant = f"images\\binary_scenario\\test\\{magnificient}x\\malignant"

            predict_folder_benign = predict_folder(folder_path_benign, f"{magnificient}x")
            predict_folder_malignant = predict_folder(folder_path_malignant, f"{magnificient}x")
            model_name = f"knn_{magnificient}x"
        else: 
            folder_path_benign = f"images\\binary_scenario_merged\\test\\benign"
            folder_path_malignant = f"images\\binary_scenario_merged\\test\\malignant"
        
            predict_folder_benign = predict_folder(folder_path_benign)
            predict_folder_malignant = predict_folder(folder_path_malignant)
            model_name = "knn_all"

        class_labels = {"benign": "Benigno", "malignant": "Maligno"}

        # Mostrar matriz de confusión y guardar métricas
        show_confusion_matrix_from_dicts(predict_folder_benign, predict_folder_malignant, class_labels, model_name)