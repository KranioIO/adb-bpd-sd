import os
import shutil
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

    # 4. Obtener la firma original del modelo
    ruta_modelo_uc = f"models:/{origen}/{numero_version}"
    firma_original = mlflow.models.get_model_info(ruta_modelo_uc).signature
    print("✍️ Firma original del modelo recuperada con éxito.")

    # 5. Descargar los artefactos originales a la máquina de GitHub Actions
    print("⏳ Descargando artefactos físicamente desde el origen...")
    carpeta_temporal = mlflow.artifacts.download_artifacts(artifact_uri=ruta_modelo_uc)
    
    # 6. LA MAGIA: Romper el rastreo eliminando el archivo MLmodel original
    # Esto evita que arrastre el ID oculto de DEV ('m-...') o el Run ID viejo
    archivo_mlmodel_viejo = os.path.join(carpeta_temporal, "MLmodel")
    if os.path.exists(archivo_mlmodel_viejo):
        os.remove(archivo_mlmodel_viejo)
        print("🧹 Eliminada la metadata vieja con IDs bloqueados de DEV.")

    # Apuntar a los archivos puros del modelo (la subcarpeta 'model')
    ruta_archivos_puros = os.path.join(carpeta_temporal, "model")
    if not os.path.exists(ruta_archivos_puros):
        ruta_archivos_puros = carpeta_temporal

    artefactos_genericos = {"model_files": ruta_archivos_puros}
    
    # 7. Registrar limpiamente en el Model Registry de QA usando pyfunc universal
    print(f"🚀 Promoviendo una copia limpia y genérica hacia: {destino}")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            artifacts=artefactos_genericos, 
            signature=firma_original,
            registered_model_name=destino,
            python_model=mlflow.pyfunc.PythonModel() # Contenedor universal genérico
        )
    
    print(f"🎉 ¡Promoción completada con éxito hacia: {destino}!")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")