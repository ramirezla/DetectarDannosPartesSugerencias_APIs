# api_dannos/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import os
import numpy as np
from typing import Dict, List, Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: todos, 1: info, 2: warnings, 3: errors  Suprimir errores de tensorflow
import tensorflow as tf
from tensorflow.keras import applications

# import base64
# from io import BytesIO
# from PIL import Image

# Configuración
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelos/final_model.keras")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "predecir")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Usar rutas absolutas o relativas al directorio actual
# current_dir = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(current_dir, "../modelos/final_model.keras")
# UPLOAD_FOLDER = os.path.join(current_dir, "../predecir")

# app = FastAPI(
#     title="API de Predicción de Daños",
#     version="1.0.0"
# )
# port = int(os.environ.get("PORT", 10000))

class DamagePredictor:
    def __init__(self, model_path: str):
        """Inicializa el predictor con el modelo cargado"""
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

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocesamiento para EfficientNet"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Predicción principal"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Archivo no encontrado: {image_path}")
        
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        
        return self._format_predictions(predictions)

    def _format_predictions(self, predictions: List[np.ndarray]) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Formatea las predicciones para la API con nombres correctos"""
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

# Cargar modelo al iniciar # Cargar modelo
try:
    predictor = DamagePredictor(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando modelo: {str(e)}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint para predecir daños a partir de una imagen"""
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="Tipo de archivo no permitido. Use .png, .jpg o .jpeg")
    
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        # Guardar archivo temporal
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Realizar predicción
        result = predictor.predict(file_path)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(500, detail=f"Error al procesar la imagen: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}