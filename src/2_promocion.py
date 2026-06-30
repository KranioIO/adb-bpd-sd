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
    print(f"🏆 Encontrada la versión {numero_version} en {origen}.")

    # 4. COPIAR LA VERSIÓN DIRECTAMENTE ENTRE CATÁLOGOS (La solución nativa)
    # Este comando es 100% genérico. Copia cualquier framework (LangChain, Sklearn, etc.)
    # porque Databricks duplica el artefacto directamente en el almacenamiento de la nube.
    print(f"🚀 Copiando físicamente el modelo vía Unity Catalog hacia: {destino}...")
    
    nueva_version = client.copy_model_version(
        src_model_name=origen,
        src_model_version=numero_version,
        dst_model_name=destino
    )
    
    print(f"🎉 ¡PROMOCIÓN COMPLETADA CON ÉXITO! Nueva versión creada en QA: {nueva_version.version}")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")