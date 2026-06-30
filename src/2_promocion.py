import os
import mlflow
from mlflow import MlflowClient

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Configuración Dinámica
catalogo_origen = os.getenv("ORIGIN_CATALOG", "dev_catalog")
catalogo_destino = os.getenv("ENV_CATALOG", "qa_catalog")
nombre_modelo = os.getenv("MODEL_NAME", "house_model")

origen = f"{catalogo_origen}.default.{nombre_modelo}"
destino = f"{catalogo_destino}.default.{nombre_modelo}_qa"

print(f"🤖 Buscando el modelo @champion en {origen}...")

try:
    # 3. Buscar la versión campeón en el origen
    version_champion = client.get_model_version_by_alias(origen, "champion")
    numero_version = version_champion.version
    print(f"🏆 Encontrada la versión {numero_version}.")

    # 4. Obtener la información de la versión
    informacion_version = client.get_model_version(origen, numero_version)
    
    # EXTRAEMOS EL RUN ID ORIGINAL (Donde nació el modelo antes de UC)
    run_id_original = informacion_version.run_id
    
    # 5. CONSTRUIMOS LA RUTA FÍSICA INMUTABLE DEL EXPERIMENTO
    # Esto apunta directamente al almacenamiento raíz (S3/Azure Blob/DBFS) 
    # y no al ID lógico 'm-...' de Unity Catalog que causa el bloqueo.
    ruta_fisica_inmutable = f"runs:/{run_id_original}/model"
    
    print(f"📦 Ruta física inmutable del experimento detectada: {ruta_fisica_inmutable}")
    print(f"🚀 Promoviendo y registrando versión en el destino de forma genérica: {destino}...")
    
    # 6. Registrar la versión usando la ruta del Run original
    # Es 100% compatible con cualquier framework (LangChain, Sklearn, etc.)
    nueva_version = client.create_model_version(
        name=destino,
        source=ruta_fisica_inmutable
    )
    
    print(f"🎉 ¡Promoción completada con éxito! Nueva versión creada en QA: {nueva_version.version}")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")