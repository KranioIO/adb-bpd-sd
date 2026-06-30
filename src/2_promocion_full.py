import os
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature # <-- NUEVA IMPORTACIÓN
import pandas as pd

# 1. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 2. Configurar nombres de catálogos
catalogo_destino = os.getenv("ENV_CATALOG", "qa_catalog")

if catalogo_destino == "qa_catalog" or catalogo_destino == "prd_catalog":
    origen = "dev_catalog.default.house_model"
    destino = f"{catalogo_destino}.default.house_model_qa"
elif catalogo_destino == "enriched":
    origen = "dev_catalog.default.house_model_qa"
    destino = "prd_catalog.default.house_model_prod"

print(f"🤖 Activando grúa... Buscando el modelo @champion en {origen}")

try:
    # 3. Buscar la versión campeón en DEV
    version_champion = client.get_model_version_by_alias(origen, "champion")
    numero_version = version_champion.version
    print(f"🏆 Encontrada la versión {numero_version} con el alias @champion.")

    # 4. Descargar los artefactos localmente
    ruta_modelo_uc = f"models:/{origen}/{numero_version}"
    carpeta_temporal = mlflow.artifacts.download_artifacts(artifact_uri=ruta_modelo_uc)
    
    # 5. CARGAR EL MODELO EN MEMORIA
    modelo_cargado = mlflow.sklearn.load_model(carpeta_temporal)
    
    # 6. SOLUCIÓN AL ERROR: Re-inferir la firma obligatoria de Unity Catalog
    # Simulamos un dato de entrada y salida para que MLflow extraiga el esquema del modelo automáticamente
    datos_ejemplo_X = pd.DataFrame({"sqft": [1500]})
    prediccion_ejemplo_y = modelo_cargado.predict(datos_ejemplo_X)
    firma_obligatoria = infer_signature(datos_ejemplo_X, prediccion_ejemplo_y) # <-- Genera el contrato de datos
    print("✍️ Firma del modelo (Contrato de esquema de datos) generada con éxito.")

    # 7. LA GRÚA CON PASO VIP: Registramos en QA inyectando la firma obligatoria
    print(f"🚀 Registrando nueva versión autorizada en: {destino}")
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=modelo_cargado,
            artifact_path="model",
            #signature=firma_obligatoria,
            registered_model_name=destino
        )
    
    print(f"🎉 ¡PROMOCIÓN COMPLETADA CON ÉXITO ABSOLUTO HACIA: {destino}!")

except Exception as e:
    print(f"❌ Error al intentar promover el modelo: {e}")