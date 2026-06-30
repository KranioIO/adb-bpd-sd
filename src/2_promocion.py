import os
import mlflow
from mlflow import MlflowClient

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Configuración Dinámica (Se adapta a cualquier modelo y catálogo)
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

    # 4. Obtener la ruta de almacenamiento físico original del modelo
    informacion_modelo = client.get_model_version(origen, numero_version)
    ruta_fisica_origen = informacion_modelo.source
    
    print(f"📦 Ruta física de origen detectada: {ruta_fisica_origen}")
    print(f"🚀 Clonando y registrando versión en el destino: {destino}...")
    
    # 5. CREAR LA VERSIÓN DIRECTAMENTE EN EL MODEL REGISTRY DE DESTINO
    # Esto es 100% genérico. No importa si es Sklearn o un Agente LangChain.
    # Toma los archivos de la ruta física y los registra de forma limpia en QA.
    nueva_version = client.create_model_version(
        name=destino,
        source=ruta_fisica_origen
    )
    
    print(f"🎉 ¡Promoción completada con éxito! Nueva versión creada: {nueva_version.version}")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")