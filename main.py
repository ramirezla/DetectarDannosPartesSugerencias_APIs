"""
API de Detección de Daños usando FastAPI y un modelo TensorFlow Keras.

Este módulo define una API REST para predecir daños en imágenes de vehículos.
Incluye la clase DamagePredictor para cargar el modelo, preprocesar imágenes,
realizar predicciones y formatear resultados.

Endpoints disponibles:
- GET /: Mensaje de bienvenida y descripción de endpoints.
- POST /predict: Recibe una imagen y devuelve predicciones de daños.
- GET /health: Verifica el estado de la API y carga del modelo.

Configuración:
- MODEL_PATH: Ruta al modelo entrenado.
- UPLOAD_FOLDER: Carpeta para almacenar imágenes subidas temporalmente.

"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import applications
import numpy as np
from typing import Dict, List, Union
from PIL import Image
import base64
from io import BytesIO
import pickle

app = FastAPI(title="API de Detección de Daños", version="1.0")

# Configuración de rutas
## MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos/final_model.keras")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos/final_model_fine_tuned_v2.keras")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "predecir")

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DamagePredictor:
    """
    Clase para cargar el modelo de detección de daños, preprocesar imágenes,
    realizar predicciones y formatear los resultados para la API.
    """
    def __init__(self, model_path: str):
        """
        Inicializa el predictor cargando el modelo desde la ruta especificada.
        Args:
            model_path (str): Ruta al archivo del modelo Keras.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)
        
        # Diccionarios de etiquetas (asegúrate que coincidan con el entrenamiento)
        self.label_maps = {
            'partes': {
                1: "Antiniebla delantero derecho",
                2: "Antiniebla delantero izquierdo",
                3: "Capó",
                4: "Cerradura capo",
                5: "Cerradura maletero",
                6: "Cerradura puerta",
                7: "Espejo lateral derecho",
                8: "Espejo lateral izquierdo",
                9: "Faros derecho",
                10: "Faros izquierdo",
                11: "Guardabarros delantero derecho",
                12: "Guardabarros delantero izquierdo",
                13: "Guardabarros trasero derecho",
                14: "Guardabarros trasero izquierdo",
                15: "Luz indicadora delantera derecha",
                16: "Luz indicadora delantera izquierda",
                17: "Luz indicadora trasera derecha",
                18: "Luz indicadora trasera izquierda",
                19: "Luz trasera derecho",
                20: "Luz trasera izquierdo",
                21: "Maletero",
                22: "Manija derecha",
                23: "Manija izquierda",
                24: "Marco de la ventana",
                25: "Marco de las puertas",
                26: "Moldura capó",
                27: "Moldura puerta delantera derecha",
                28: "Moldura puerta delantera izquierda",
                29: "Moldura puerta trasera derecha",
                30: "Moldura puerta trasera izquierda",
                31: "Parabrisas delantero",
                32: "Parabrisas trasero",
                33: "Parachoques delantero",
                34: "Parachoques trasero",
                35: "Puerta delantera derecha",
                36: "Puerta delantera izquierda",
                37: "Puerta trasera derecha",
                38: "Puerta trasera izquierda",
                39: "Rejilla, parrilla",
                40: "Rueda",
                41: "Tapa de combustible",
                42: "Tapa de rueda",
                43: "Techo",
                44: "Techo corredizo",
                45: "Ventana delantera derecha",
                46: "Ventana delantera izquierda",
                47: "Ventana trasera derecha",
                48: "Ventana trasera izquierda",
                49: "Ventanilla delantera derecha",
                50: "Ventanilla delantera izquierda",
                51: "Ventanilla trasera derecha",
                52: "Ventanilla trasera izquierda"
            },
            'dannos': {
                1: "Abolladura",
                2: "Deformación",
                3: "Desprendimiento",
                4: "Fractura",
                5: "Rayón",
            6: "Rotura"
            },
            'sugerencias': {
                1: "Reparar",
                2: "Reemplazar"
            }
        }

    def load_model(model_path):
    """
        Carga un modelo guardado en formato .keras o .h5 sin necesidad de reentrenar.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo del modelo no existe: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo cargado desde {model_path}")
        return model

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesa la imagen para el modelo EfficientNet.
        Args:
            image_path (str): Ruta a la imagen a procesar.
        Returns:
            np.ndarray: Imagen preprocesada lista para la predicción.
        """
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    # def predict(self, image_path: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Realiza la predicción principal de daños en la imagen.
        Args:
            image_path (str): Ruta a la imagen para predecir.
        Raises:
            FileNotFoundError: Si la imagen no existe en la ruta dada.
        Returns:
            Dict[str, List[Dict[str, Union[str, float]]]]: Diccionario con las predicciones
            para cada categoría ('partes', 'dannos', 'sugerencias'), cada una con las
            top 3 etiquetas y sus probabilidades.
        """
        # if not os.path.exists(image_path):
            # raise FileNotFoundError(f"Archivo no encontrado: {image_path}")
        
        # img_array = self.preprocess_image(image_path)
        # predictions = self.model.predict(img_array)
        
        # return self._format_predictions(predictions)

    def predict(image_path, model, mlb_partes, mlb_danos, mlb_sugerencias):
        """
        Realiza predicción usando el modelo cargado para una imagen dada.
        """
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)

        partes_probs = predictions[0][0]
        dannos_probs = predictions[1][0]
        sugerencias_probs = predictions[2][0]

        def get_top_predictions(classes, probs, label_dict, top_n=3):
            top_items = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:top_n]
            return [(label_dict.get(int(cls), f"Clase_{int(cls)}"), float(prob)) for cls, prob in top_items]

        partes_pred = get_top_predictions(mlb_partes.classes_, partes_probs, label_to_cls_piezas)
        dannos_pred = get_top_predictions(mlb_danos.classes_, dannos_probs, label_to_cls_danos)
        sugerencias_pred = get_top_predictions(mlb_sugerencias.classes_, sugerencias_probs, label_to_cls_sugerencias)

        return {
            'partes': partes_pred,
            'dannos': dannos_pred,
            'sugerencias': sugerencias_pred
        }

    def _format_predictions(self, predictions: List[np.ndarray]) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Formatea las predicciones para la API con nombres legibles.
        Args:
            predictions (List[np.ndarray]): Lista de arrays con probabilidades para cada categoría.
        Returns:
            Dict[str, List[Dict[str, Union[str, float]]]]: Diccionario con las predicciones formateadas.
        """
        results = {}
        
        # Mapeo de categorías a sus respectivos diccionarios
        categories = ['partes', 'dannos', 'sugerencias']
        
        for i, category in enumerate(categories):
            probs = predictions[i][0]
            label_dict = self.label_maps[category]
            
            # Obtener los índices de las top 3 predicciones (de mayor a menor probabilidad)
            top_indices = np.argsort(probs)[::-1][:3]
            
            category_predictions = []
            for idx in top_indices:
                # Los índices del modelo comienzan en 0, pero tus diccionarios en 1
                class_id = idx + 1
                class_name = label_dict.get(class_id, f"Clase_{class_id}")
                
                category_predictions.append({
                    "label": class_name,
                    "probability": float(probs[idx])
                })
            
            results[category] = category_predictions
        
        return results

# Cargar modelo al iniciar
# try:
    # predictor = DamagePredictor(MODEL_PATH)
# except Exception as e:
    # raise RuntimeError(f"Error cargando modelo: {str(e)}")

 # Cargar modelo
model_path = "final_model_fine_tuned_v2.keras"  # Cambiar si se usa otro archivo
model = load_model(model_path)

# Cargar MultiLabelBinarizer guardados (adaptar rutas)
with open("mlb_partes.pkl", "rb") as f:
    mlb_partes = pickle.load(f)
with open("mlb_dannos.pkl", "rb") as f:
    mlb_dannos = pickle.load(f)
with open("mlb_sugerencias.pkl", "rb") as f:
    mlb_sugerencias = pickle.load(f)

def allowed_file(filename: str) -> bool:
    """
    Verifica si el archivo tiene una extensión permitida.
    Args:
        filename (str): Nombre del archivo.
    Returns:
        bool: True si la extensión es permitida, False en caso contrario.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/")
async def root():
    """
    Endpoint raíz que devuelve un mensaje de bienvenida y los endpoints disponibles.
    Returns:
        dict: Mensaje de bienvenida y descripción de endpoints.
    """
    return {
        "message": "Bienvenido a la API de Detección de Daños",
        "endpoints": {
            "predict": "/predict (POST - Sube una imagen para predecir daños)",
            "health": "/health (GET - Verifica el estado de la API)"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para predecir daños en una imagen subida.
    Args:
        file (UploadFile): Archivo de imagen subido.
    Returns:
        dict: Diccionario con la predicción y la imagen codificada en base64.
    Raises:
        HTTPException: Si ocurre un error durante la predicción.
    """
    try:
        # Guardar archivo temporalmente
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Realizar predicción
        result = predictor.predict(file_path)
        
        # Opcional: devolver imagen en base64
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return {
            "prediction": result,
            "image_base64": encoded_image
        }
        
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API y la carga del modelo.
    Returns:
        dict: Estado de la API y confirmación de carga del modelo.
    """
    return {"status": "OK", "model_loaded": True}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000))) 
