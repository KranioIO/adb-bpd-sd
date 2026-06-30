import os
import mlflow
from mlflow import MlflowClient

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Origen y destino fijos
origen = "dev_catalog.default.house_model"
destino = "qa_catalog.default.house_model_qa"

print(f"🤖 Buscando el modelo @champion en {origen}...")

try:
    # 3. Buscar la versión campeón en el origen
    version_champion = client.get_model_version_by_alias(origen, "champion")
    numero_version = version_champion.version
    print(f"🏆 Encontrada la versión {numero_version}.")

    # 4. Obtener la RUTA y la FIRMA original
    ruta_modelo_uc = f"models:/{origen}/{numero_version}"
    firma_original = mlflow.models.get_model_info(ruta_modelo_uc).signature
    print("✍️ Firma original del modelo recuperada con éxito.")

    # 5. Descargar los artefactos originales
    carpeta_temporal = mlflow.artifacts.download_artifacts(artifact_uri=ruta_modelo_uc)
    
    # 6. LA SOLUCIÓN: Apuntar a la subcarpeta 'model' interna para aislar los archivos puros
    # Esto evita arrastrar el archivo 'MLmodel' viejo que tiene el ID bloqueado de DEV
    ruta_archivos_puros = os.path.join(carpeta_temporal, "model")
    if not os.path.exists(ruta_archivos_puros):
        ruta_archivos_puros = carpeta_temporal # Por si no tuviera subcarpeta

    # Mapeamos los archivos puros como un diccionario de artefactos genérico
    artefactos_genericos = {"model_files": ruta_archivos_puros}
    
    # 7. Registrar usando un modelo Wrapper genérico (PythonModel base)
    print(f"🚀 Promoviendo de forma 100% limpia y genérica a: {destino}")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            artifacts=artefactos_genericos, # <--- Pasamos los archivos puros aquí
            signature=firma_original,
            registered_model_name=destino,
            python_model=mlflow.pyfunc.PythonModel() # <--- Un modelo vacío genérico que solo sirve de contenedor
        )
    
    print(f"🎉 ¡Promoción completada con éxito hacia: {destino}!")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")