# databricks: ignore
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient
import os

# 1. Parámetros dinámicos vía Widgets (Inyectados por el Bundle en Jobs)
dbutils.widgets.text("catalog", "dev_catalog")
catalog = dbutils.widgets.get("catalog")

# Ruta del Volumen para los archivos Parquet
output_path = f"/Volumes/{catalog}/default/model_outputs"

# 2. Simular datos de entrenamiento
data = pd.DataFrame({
    "sqft": [1000, 2000, 3000, 4000, 5000], 
    "price": [200, 400, 600, 800, 1000]
})
X = data[["sqft"]]
y = data["price"]

# 3. Configurar MLflow para Unity Catalog
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run() as run:
    # Entrenamiento
    model = LinearRegression().fit(X, y)
    
    # Calcular métricas para decidir si es "Ganador"
    predictions_train = model.predict(X)
    mse = mean_squared_error(y, predictions_train)
    mlflow.log_metric("mse", mse)
    
    # Ejemplo de entrada para la Firma (Signature) - Requerido por UC
    input_example = X.head(5)
    
    # Registro del modelo en Unity Catalog
    model_name = f"{catalog}.default.house_model"
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        input_example=input_example, 
        registered_model_name=model_name
    )
    
    # --- LÓGICA DE MODELO GANADOR (CHAMPION) ---
    client = MlflowClient()
    model_version = model_info.registered_model_version
    
    # Umbral de calidad (Si el error es bajo, lo promovemos a Champion)
    if mse < 100:
        print(f"✅ ¡Modelo Ganador! Versión {model_version} promovida a @champion")
        client.set_registered_model_alias(model_name, "champion", model_version)
    else:
        print(f"⚠️ Modelo versión {model_version} no superó el umbral de calidad.")

    # 4. Generar Predicciones de prueba (incluyendo un valor negativo para la Alerta)
    new_data = pd.DataFrame({"sqft": [1500, 2500, -500]}) 
    predictions = model.predict(new_data)
    
    results_df = new_data.copy()
    results_df["price_predicted"] = predictions
    results_df["execution_date"] = pd.Timestamp.now()

    # 5. Persistencia de Resultados
    spark_df = spark.createDataFrame(results_df)

    # Opción A: Guardar como archivo Parquet en el Volumen
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.default.model_outputs")

    file_destination = f"{output_path}/house_predictions.parquet"
    spark_df.write.mode("overwrite").parquet(file_destination)
    
    # Guardar en Tabla (Para Alertas SQL y Dashboards)
    table_destination = f"{catalog}.default.ma_table_scores"
    spark_df.write.mode("overwrite").saveAsTable(table_destination)

print(f"🚀 Proceso finalizado en el catálogo: {catalog}")