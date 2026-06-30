import mlflow
from mlflow import MlflowClient

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Definir origen fijo (DEV) y destino fijo (QA)
origen = "dev_catalog.default.house_model"
destino = "main_qa.default.house_model_qa"

print(f"🤖 Buscando modelo @champion en {origen}...")

try:
    # 3. Obtener la versión con el alias @champion
    version_champion = client.get_model_version_by_alias(origen, "champion")
    numero_version = version_champion.version
    
    # 4. Ruta directa del modelo en DEV
    uri_modelo_origen = f"models:/{origen}/{numero_version}"
    
    # 5. Registrar directamente en QA usando la URI de origen
    print(f"🚀 Promoviendo directamente a: {destino}")
    mlflow.register_model(model_uri=uri_modelo_origen, name=destino)
    
    print("🎉 ¡Promoción a QA completada con éxito!")

except Exception as e:
    print(f"❌ Error: {e}")