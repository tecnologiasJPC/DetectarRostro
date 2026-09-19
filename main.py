from pathlib import Path
from PIL import Image
import cv2
import sys

def recortar_rostro(ruta_imagen_entrada, ruta_imagen_salida):
    # 1. Cargar el detector de rostros preentrenado de OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 2. Cargar la imagen de entrada
    imagen = cv2.imread(ruta_imagen_entrada)

    if imagen is None:
        print(
            f"Error: No se pudo cargar la imagen desde '{ruta_imagen_entrada}'"
        )
        return

    # 3. Convertir la imagen a escala de grises (mejora la precisión de detección)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 4. Detectar rostros en la imagen
    rostros = face_cascade.detectMultiScale(
        gris,
        scaleFactor=1.1,  # Reduce la escala en cada paso
        minNeighbors=5,  # Cuántos vecinos debe tener cada rectángulo para conservarlo
        minSize=(30, 30),  # Tamaño mínimo del rostro
    )

    if len(rostros) == 0:
        print("No se detectó ningún rostro en la imagen.")
        return

    # 5. Tomar el primer rostro detectado (x, y, ancho, alto)
    x, y, w, h = rostros[0]

    # Opcional: Agregar un pequeño margen alrededor del rostro
    margen = 30
    alto_img, ancho_img, _ = imagen.shape

    y_inicio = max(0, y - margen)
    y_fin = min(alto_img, y + h + margen)
    x_inicio = max(0, x - margen)
    x_fin = min(ancho_img, x + w + margen)

    # 6. Recortar la región de interés (ROI)
    recorte = imagen[y_inicio:y_fin, x_inicio:x_fin]

    # 7. Guardar la imagen recortada
    cv2.imwrite(ruta_imagen_salida, recorte)
    print(f"¡Rostro recortado exitosamente y guardado en '{ruta_imagen_salida}'!")


def recortar_y_guardar_en_mismo_lugar(ruta_imagen, cantidad_pixeles=5):
    """
    Abre una imagen, le recorta un número de píxeles en cada borde
    y la guarda en la misma ubicación sobreescribiendo el archivo original.
    
    :param ruta_imagen: Ruta (str o Path) al archivo de imagen.
    :param cantidad_pixeles: Cantidad de píxeles a recortar por cada borde (por defecto 4).
    """
    ruta = Path(ruta_imagen)
    
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_imagen}")
    
    # Abrir la imagen y recortarla usando la sintaxis `with` para asegurar que el archivo de entrada se cierre correctamente
    with Image.open(ruta) as img:
        ancho, alto = img.size
        
        # Definir la caja de recorte (izquierda, superior, derecha, inferior)
        caja_recorte = (
            cantidad_pixeles,
            cantidad_pixeles,
            ancho - cantidad_pixeles,
            alto - cantidad_pixeles
        )
        
        # Realizar el recorte y cargar los datos en memoria
        img_recortada = img.crop(caja_recorte)
        img_recortada.load()
    
    # Guardar la imagen recortada en la misma ruta
    img_recortada.save(ruta)
    print(f"Imagen guardada correctamente en: {ruta.resolve()}")


def obtener_imagen_mas_reciente(ruta_carpeta):
    # Definir las extensiones de imagen permitidas
    extensiones_imagen = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    carpeta = Path(ruta_carpeta)
    
    # Validar que la ruta existe y es un directorio
    if not carpeta.is_dir():
        raise ValueError(f"La ruta '{ruta_carpeta}' no existe o no es una carpeta.")
    
    # Filtrar solo archivos que tengan una extensión de imagen
    imagenes = [
        archivo for archivo in carpeta.iterdir() 
        if archivo.is_file() and archivo.suffix.lower() in extensiones_imagen
    ]
    
    # Si no hay imágenes en la carpeta, retornar None
    if not imagenes:
        return None
    
    # Obtener el archivo con la fecha de modificación más reciente
    imagen_mas_reciente = max(imagenes, key=lambda archivo: archivo.stat().st_mtime)
    
    # Retornar solo el nombre del archivo (con extensión)
    return imagen_mas_reciente.name


# --- Ejemplo de uso ---
if __name__ == "__main__":

    # Sustituye con la ruta de tu carpeta
    ruta = "C:\\Users\\john_\\Downloads"
    # recortar_rostro(Path(ruta) /"16454700_008_e643.jpg", Path(ruta) /"rostro_recortado.jpg") 
    # sys.exit()
    
    try:
        resultado = obtener_imagen_mas_reciente(ruta)
        if resultado:
            print(f"La imagen más reciente es: {resultado}")
            recortar_y_guardar_en_mismo_lugar(Path(ruta) / resultado)
        else:
            print("No se encontraron imágenes en la carpeta.")
    except Exception as e:
        print(f"Error: {e}")
