import pandas as pd
import os

def format_metrics(metrics: dict):
    return "\n".join(f"{key}: {value}" for key, value in metrics.items())

def save_metricas_csv(model_name: str, metrics: dict, output_paht="metricas.csv"):
    if os.path.exists(output_paht):
        df = pd.read_csv(output_paht)
    else:
        columnas = ["archivo"] + list(metrics.keys())
        df = pd.DataFrame(columns=columnas)

    df = df[df["archivo"] != model_name]

    nueva_fila = {"archivo": model_name, **metrics}
    df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)

    df = df.sort_values("archivo").reset_index(drop=True)
    
    df.to_csv(output_paht, index=False)