import os
from pathlib import Path
from ultralytics import YOLO

print("================================")
print("       OBJECT VISION AI")
print("================================")

# Ubicacion del proyecto
BASE = Path(__file__).resolve().parent

# Banco de imagenes
CARPETA = BASE / "banco_imagenes"

# Carpeta donde se guardaran resultados
RESULTADOS = BASE / "resultados"

# Comprobar que existe el banco
if not CARPETA.exists():
    print("ERROR: No existe la carpeta banco_imagenes")
    input("Presiona ENTER para salir")
    raise SystemExit

# Buscar las 20 imagenes
extensiones = [".jpg", ".jpeg", ".png"]

imagenes = [
    archivo
    for archivo in CARPETA.iterdir()
    if archivo.suffix.lower() in extensiones
]

print("\nImagenes encontradas:", len(imagenes))

if len(imagenes) == 0:
    print("No hay imagenes en el banco.")
    input("Presiona ENTER para salir")
    raise SystemExit

# Cargar inteligencia artificial YOLO
print("\nCargando modelo de inteligencia artificial...")

modelo = YOLO("yolo26n.pt")

print("Modelo cargado correctamente.")

# Analizar todas las imagenes
print("\nIniciando reconocimiento de objetos...")

resultados = modelo.predict(
    source=[str(img) for img in imagenes],
    conf=0.25,
    save=True,
    project=str(BASE),
    name="resultados",
    exist_ok=True,
    device="cpu"
)

# Mostrar resultados
print("\n========== RESULTADOS ==========")

for resultado in resultados:

    nombre_imagen = Path(resultado.path).name

    print("\nImagen:", nombre_imagen)

    if resultado.boxes is None or len(resultado.boxes) == 0:
        print("No se detectaron objetos.")
        continue

    objetos = {}

    for caja in resultado.boxes:

        clase = int(caja.cls[0])
        nombre = resultado.names[clase]

        objetos[nombre] = objetos.get(nombre, 0) + 1

    for objeto, cantidad in objetos.items():

        print("Objeto:", objeto)
        print("Cantidad:", cantidad)

# Resumen
print("\n================================")
print("ANALISIS TERMINADO")
print("Total de imagenes:", len(imagenes))
print("Resultados guardados en:")
print(RESULTADOS)
print("================================")

input("\nPresiona ENTER para cerrar...")

