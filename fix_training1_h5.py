import h5py

# Modifica el archivo .h5 para eliminar 'groups' (si es necesario)
with h5py.File('models/training_1.h5', 'r+') as f:
    model_config = f.attrs['model_config']  # Ya está decodificado, no hace falta .decode()
    model_config = model_config.replace('"groups": 1,', '')  # Eliminar 'groups': 1
    f.attrs.modify('model_config', model_config)  # Modificar el archivo