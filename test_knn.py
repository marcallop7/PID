from knn import predict_folder

def matriz_dispersion(res_benign, res_malignant):
    # Extraemos los valores
    TP = res_malignant['malignant']  # verdaderos positivos
    TN = res_benign['benign']        # verdaderos negativos
    FP = res_benign['malignant']     # falsos positivos
    FN = res_malignant['benign']     # falsos negativos

    # Mostrar matriz de dispersión
    print(f"{'':12}|{'benign':>10}   {'malignant':>10}")
    print("-" * 34)
    print(f"{'benign':12}|{TN:>10}   {FP:>10}")
    print(f"{'malignant':12}|{FN:>10}   {TP:>10}")
    print("\nMedidas de evaluación:")

    # Cálculos
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    # Mostrar métricas
    print(f"{'Accuracy':20}: {accuracy:.2f}")
    print(f"{'Precision (malignant)':20}: {precision:.2f}")
    print(f"{'Recall (malignant)':20}: {recall:.2f}")
    print(f"{'Specificity (benign)':20}: {specificity:.2f}")
    print(f"{'F1 Score':20}: {f1_score:.2f}")


predict_folder_benign = predict_folder(folder_path_benign)
predict_folder_malignant = predict_folder(folder_path_malignant)
matriz_dispersion(predict_folder_benign, predict_folder_malignant)