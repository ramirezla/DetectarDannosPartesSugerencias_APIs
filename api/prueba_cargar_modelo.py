from predecir import DamagePredictor
import os

model_path = os.path.join("../modelos/", "final_model.keras")
try:
    predictor = DamagePredictor(model_path)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {str(e)}")