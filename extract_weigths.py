from tensorflow.keras.models import load_model # type: ignore
import os

# Rutas corregidas según tu sistema
rutas_modelos = [
    './modelo_cnn_40x.h5',
    './modelo_cnn_100x.h5',
    './modelo_cnn_200x.h5',
    './modelo_cnn_400x.h5',
    './modelo_cnn_all.h5'
]

carpeta_salida = 'pesos_guardados'
os.makedirs(carpeta_salida, exist_ok=True)

for ruta in rutas_modelos:
    if not os.path.exists(ruta):
        print(f'❌ Archivo no encontrado: {ruta}')
        continue

    try:
        modelo = load_model(ruta)
        nombre_modelo = os.path.splitext(os.path.basename(ruta))[0]
        ruta_pesos = os.path.join(carpeta_salida, f'{nombre_modelo}.weights.h5')
        modelo.save_weights(ruta_pesos)
        print(f'✅ Pesos guardados: {ruta_pesos}')

    except Exception as e:
        print(f'❌ Error procesando {ruta}: {e}')
