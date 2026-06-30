import mlflow
from mlflow import MlflowClient

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Origen y destino fijos (Sin condiciones)
origen = "dev_catalog.default.house_model_qa"
destino = "qa_catalog.default.house_model_qa"  # Cambia esto al catálogo que necesites probar

print(f"🤖 Buscando el modelo @champion en {origen}...")

try:
    # 3. Buscar la versión campeón en el origen
    version_champion = client.get_model_version_by_alias(origen, "champion")
    numero_version = version_champion.version
    print(f"🏆 Encontrada la versión {numero_version}.")

    # 4. Obtener la RUTA y la FIRMA original (Válido para cualquier tipo de modelo)
    ruta_modelo_uc = f"models:/{origen}/{numero_version}"
    firma_original = mlflow.models.get_model_info(ruta_modelo_uc).signature
    print("✍️ Firma original del modelo recuperada con éxito.")

    # 5. Descargar los artefactos originales a la máquina de GitHub Actions
    carpeta_temporal = mlflow.artifacts.download_artifacts(artifact_uri=ruta_modelo_uc)
    
    # 6. Registrar usando 'pyfunc' (El sabor universal de MLflow)
    print(f"🚀 Promoviendo modelo de forma genérica a: {destino}")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            data_path=carpeta_temporal, # Le pasamos la carpeta descargada tal cual
            signature=firma_original,   # Inyectamos su firma original
            registered_model_name=destino
        )
    
    print(f"🎉 ¡Promoción completada con éxito hacia: {destino}!")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")