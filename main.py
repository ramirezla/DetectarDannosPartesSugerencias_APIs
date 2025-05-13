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
import tempfile
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import base64
import json

app = FastAPI(title="API de Detección de Daños", version="3.2")

# Diccionarios de mapeo (debe adaptarse si se usan otros)
label_to_cls_piezas = {
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
}

label_to_cls_danos = {
    1: "Abolladura",
    2: "Deformación",
    3: "Desprendimiento",
    4: "Fractura",
    5: "Rayón",
    6: "Rotura"
}

label_to_cls_sugerencias = {
    1: "Reparar",
    2: "Reemplazar"
}

MODEL_PATH = "modelos/final_model_fine_tuned_v2.keras"
MLB_PARTES_PATH = "mlb_partes.pkl"
MLB_DANNOS_PATH = "mlb_dannos.pkl"
MLB_SUGERENCIAS_PATH = "mlb_sugerencias.pkl"

# Variables globales para modelo y binarizadores
model = None
mlb_partes = None
mlb_danos = None
mlb_sugerencias = None

# Load model and MultiLabelBinarizers on startup
@app.on_event("startup")
def load_resources():
    global model, mlb_partes, mlb_danos, mlb_sugerencias
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    if not os.path.exists(MLB_PARTES_PATH) or not os.path.exists(MLB_DANNOS_PATH) or not os.path.exists(MLB_SUGERENCIAS_PATH):
        raise RuntimeError("Archivos de MultiLabelBinarizer no encontrados")
    with open(MLB_PARTES_PATH, "rb") as f:
        mlb_partes = pickle.load(f)
    with open(MLB_DANNOS_PATH, "rb") as f:
        mlb_danos = pickle.load(f)
    with open(MLB_SUGERENCIAS_PATH, "rb") as f:
        mlb_sugerencias = pickle.load(f)

def preprocess_image(image_path, img_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path, model, mlb_partes, mlb_danos, mlb_sugerencias):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)

    partes_probs = predictions[0][0]
    dannos_probs = predictions[1][0]
    sugerencias_probs = predictions[2][0]

    def get_top_predictions(classes, probs, label_dict, top_n=2):    # top-n=3
        top_items = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:top_n]
        return [(label_dict.get(int(cls), f"Clase_{int(cls)}"), float(prob)) for cls, prob in top_items]

    partes_pred = get_top_predictions(mlb_partes.classes_, partes_probs, label_to_cls_piezas)
    dannos_pred = get_top_predictions(mlb_danos.classes_, dannos_probs, label_to_cls_danos)
    sugerencias_pred = get_top_predictions(mlb_sugerencias.classes_, sugerencias_probs, label_to_cls_sugerencias)

    # Opcional: devolver imagen en base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    return {
        'partes': partes_pred,
        'dannos': dannos_pred,
        'sugerencias': sugerencias_pred,
        "image_base64": encoded_image
    }

def predict_thresholds(image_path, model, mlb_partes, mlb_danos, mlb_sugerencias, thresholds_partes, img_size=(224, 224)):
    img_array = preprocess_image(image_path, img_size)
    predictions = model.predict(img_array)

    partes_probs = predictions[0][0]
    dannos_probs = predictions[1][0]
    sugerencias_probs = predictions[2][0]

    # Aplicar umbrales personalizados para partes
    partes_pred = []
    for i, cls in enumerate(mlb_partes.classes_):
        cls_name = str(cls)
        threshold = thresholds_partes.get(cls_name, 0.5)  # usar 0.5 si no está definido
        prob = float(partes_probs[i])
        above_thresh = bool(prob >= threshold)
        partes_pred.append({
            "class": cls_name,
            "probability": prob,
            "above_threshold": above_thresh
        })

    # Para daños y sugerencias se usa umbral fijo 0.5 (puede extenderse si se desea)
    dannos_pred = []
    for i, cls in enumerate(mlb_danos.classes_):
        prob = float(dannos_probs[i])
        above_thresh = bool(dannos_probs[i] >= 0.5)
        dannos_pred.append({
            "class": str(cls),
            "probability": prob,
            "above_threshold": above_thresh
        })

    sugerencias_pred = []
    for i, cls in enumerate(mlb_sugerencias.classes_):
        prob = float(sugerencias_probs[i])
        above_thresh = bool(sugerencias_probs[i] >= 0.5)
        sugerencias_pred.append({
            "class": str(cls),
            "probability": prob,
            "above_threshold": above_thresh
        })

    # Logging types for debugging serialization issues
    logging.debug("partes_pred types: %s", [(type(item["above_threshold"]), type(item["probability"])) for item in partes_pred])
    logging.debug("dannos_pred types: %s", [(type(item["above_threshold"]), type(item["probability"])) for item in dannos_pred])
    logging.debug("sugerencias_pred types: %s", [(type(item["above_threshold"]), type(item["probability"])) for item in sugerencias_pred])

    return {
        'partes': partes_pred,
        'dannos': dannos_pred,
        'sugerencias': sugerencias_pred
    }

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
            "predict_thresholds": "/predict_thresholds (POST - Sube una imagen para predecir con umbrales personalizados)",
            "health": "/health (GET - Verifica el estado de la API)"
        }
    }

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API y la carga del modelo.
    Returns:
        dict: Estado de la API y confirmación de carga del modelo.
    """
    return {"status": "OK", "model_loaded": True}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            results = predict(tmp.name, model, mlb_partes, mlb_danos, mlb_sugerencias)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_thresholds")
async def predict_thresholds_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    try:
        # Cargar umbrales personalizados (puedes ajustar la ruta o cargar desde otro lugar)
        thresholds_path = "optimal_thresholds_partes.json"
        with open(thresholds_path, "r") as f:
            thresholds_partes = json.load(f)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            results = predict_thresholds(tmp.name, model, mlb_partes, mlb_danos, mlb_sugerencias, thresholds_partes)
        return ORJSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
