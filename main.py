from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import os
import folium
import rasterio
from rasterio.transform import from_origin
import tempfile
from tensorflow.keras.models import load_model


## Mapa Imagen 
# Archivo modis
archivo_mod = "data/MOD13C1.A2025257.061.2025275163328.hdf"

if not os.path.exists(archivo_mod):
    raise FileNotFoundError(f" No se encontró el archivo: {archivo_mod}")

# Abrir archivo
hdf = SD(archivo_mod, SDC.READ)
# Mostrar datasets disponibles
datasets = hdf.datasets()
print("Datasets disponibles:")
for name, info in datasets.items():
    print(f" - {name}: dimensiones {info[0]}")

# Seleccionar dataset
ndvi_data = hdf.select('CMG 0.05 Deg 16 days NDVI')[:]
ndvi = np.array(ndvi_data, dtype=float)

# Limpiar datos
ndvi[ndvi == -3000] = np.nan
ndvi = ndvi / 10000.0

# Reducir resolución para evitar OOM 
ndvi = ndvi[::4, ::4]

# Cargar modelo IA
modelo_path = "data/ndvi_autoencoder_final.keras" 
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}")

print("Cargando modelo de predicción...")
modelo = load_model(modelo_path)
print("Modelo cargado correctamente.")

# Verificar entrada
modelo.summary()
print("Entrada esperada del modelo:", modelo.input_shape)

from tensorflow.image import resize

# Redimensionar NDVI para que coincida con la entrada del modelo
ndvi_pre = np.nan_to_num(ndvi[..., np.newaxis], nan=0.0)  # Añade canal y reemplaza NaN
ndvi_resized = resize(ndvi_pre, (128, 128)).numpy()

# Añadir dimensión batch
ndvi_input = ndvi_resized[np.newaxis, ...]  # (1, 128, 128, 1)
print("Forma final para predicción:", ndvi_input.shape)

# Predecir
print("Ejecutando predicción de crecimiento futuro...")
prediccion = modelo.predict(ndvi_input, verbose=1)[0, :, :, 0]

# Normalizar al rango 0–1
prediccion = (prediccion - np.min(prediccion)) / (np.max(prediccion) - np.min(prediccion))
print("Predicción completada.")

from tensorflow.image import resize

# Escalar la predicción al tamaño original del NDVI global
prediccion_upscaled = resize(
    prediccion[..., np.newaxis], (ndvi.shape[0], ndvi.shape[1])
).numpy().squeeze()

# Normalizar de nuevo
prediccion_upscaled = (prediccion_upscaled - np.min(prediccion_upscaled)) / (np.max(prediccion_upscaled) - np.min(prediccion_upscaled))

print("Predicción reescalada al tamaño del mapa:", prediccion_upscaled.shape)


# Mostrar estadísticas
print(f"NDVI: min={np.nanmin(ndvi):.3f}, max={np.nanmax(ndvi):.3f}")

# Plot de NDVI
plt.figure(figsize=(12,6))
plt.imshow(ndvi, cmap="YlGn", vmin=0, vmax=1)
plt.colorbar(label="NDVI")
plt.title("NDVI Global MOD13C1 - NASA MODIS, HDF4")
plt.xlabel("Longitud ")
plt.ylabel("Latitud ")
plt.tight_layout()
plt.savefig("ndvi_global.png")  # <-- Guarda la imagen
hdf.end()
print("Imagen guardada como ndvi_global.png")

## Mapa Interactivo
print("Creando mapa interactivo...")

# Definir umbral de crecimiento alto 
umbral = 0.8
mask_alto = np.where(ndvi > umbral * 0.9, ndvi, np.nan)

# Crear raster 
lon_res = 0.05 * 4
lat_res = 0.05 * 4
width = ndvi.shape[1]
height = ndvi.shape[0]
transform = from_origin(-180, 90, lon_res, lat_res)

# Calcular bounds globales
lat_min = 90 - height * lat_res
lat_max = 90
lon_min = -180
lon_max = -180 + width * lon_res
bounds = [
    [lat_min, lon_min],
    [lat_max, lon_max]
]

# Ajuste para bajar capas roja y morada 25 píxeles (restar para ir al sur)
shift_pixels = 25
lat_shift = shift_pixels * lat_res
bounds_shifted = [
    [lat_min - lat_shift, lon_min],
    [lat_max - lat_shift, lon_max]
]

# Crear mapa base con varias opciones
m = folium.Map(location=[23.5, -102], zoom_start=4, tiles="CartoDB dark_matter")

# Añadir diferentes fondos para comparar
folium.TileLayer("OpenStreetMap", name="Mapa estándar").add_to(m)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Satellite",
    name="Satélite"
).add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Fondo oscuro").add_to(m)

# Ajuste global de mapa
shift_background_pixels = -40   # negativo = subir el fondo, positivo = bajarlo
lat_shift_bg = shift_background_pixels * lat_res

# Recalcular límites desplazados para las capas base
lat_min_bg = lat_min - lat_shift_bg
lat_max_bg = lat_max - lat_shift_bg
bounds_bg = [
    [lat_min_bg, lon_min],
    [lat_max_bg, lon_max]
]

print(f"Mapa base desplazado {shift_background_pixels} píxeles verticalmente ({lat_shift_bg:.3f}°).")

# Capa verde (NDVI) - sin desplazamiento
folium.raster_layers.ImageOverlay(
    image=ndvi,
    bounds=bounds,
    colormap=lambda x: (0, x, 0, x),
    opacity=0.6,
    name='NDVI Vegetación',
).add_to(m)

# Capa morada (IA) - desplazada
folium.raster_layers.ImageOverlay(
    image=prediccion_upscaled,
    bounds=bounds_shifted,
    colormap=lambda x: (0.3, 0, 1, x**2),
    opacity=0.8,
    name='Predicción IA (cambio de vegetación futura)',
).add_to(m)

# Capa roja (crecimiento) - desplazada
mask_display = np.where(np.isnan(mask_alto), 0, mask_alto)
mask_norm = (mask_display - np.nanmin(mask_display)) / (np.nanmax(mask_display) - np.nanmin(mask_display))

folium.raster_layers.ImageOverlay(
    image=mask_norm,
    bounds=bounds_shifted,
    colormap=lambda x: (1, 0, 0, x**4),
    opacity=0.9,
    name='Crecimiento Exponencial (rojo)',
).add_to(m)

# Control de capas y fondos
folium.LayerControl(collapsed=False, position='topright').add_to(m)

# Guardar mapa interactivo
mapa_html = "ndvi_interactivo.html"
m.save(mapa_html)
print("Mapa interactivo guardado como ndvi_interactivo.html")
print("Abrir en navegador.")