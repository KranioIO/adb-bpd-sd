# databricks: ignore
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# 1. Parámetros dinámicos vía Widgets
dbutils.widgets.text("catalog", "dev_catalog")
catalog = dbutils.widgets.get("catalog")

# Definimos la ruta del Volumen para los outputs (asegúrate que el Volumen existe)
# Formato: /Volumes/<catalog>/<schema>/<volume_name>/
output_path = f"/Volumes/{catalog}/default/model_outputs"

# 2. Simular datos de entrenamiento
data = pd.DataFrame({"sqft": [1000, 2000, 3000, 4000], "price": [200, 400, 600, 800]})
X = data[["sqft"]]
y = data["price"]

# 3. MLflow: Entrenamiento y Registro en Unity Catalog
mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run() as run:
    model = LinearRegression().fit(X, y)
    
    # Log del modelo
    mlflow.sklearn.log_model(
        model, 
        "model", 
        registered_model_name=f"{catalog}.default.house_model"
    )
    
    # 4. Generar Predicciones para testing
    # Creamos un pequeño set de datos nuevos
    new_data = pd.DataFrame({"sqft": [1500, 2500, -500]}) # Incluimos un valor negativo para disparar la alerta
    predictions = model.predict(new_data)
    
    # Crear DataFrame de resultados
    results_df = new_data.copy()
    results_df["price_predicted"] = predictions
    results_df["execution_date"] = pd.Timestamp.now()

    # 5. Guardar resultados en PARQUET dentro del Volumen
    # Convertimos a Spark DataFrame para guardar como tabla o archivo
    spark_df = spark.createDataFrame(results_df)
    
    # Opción A: Guardar como archivo Parquet en el Volumen
    file_destination = f"{output_path}/house_predictions.parquet"
    spark_df.write.mode("overwrite").parquet(file_destination)
    
    # Opción B: Crear/Actualizar una tabla para la Alerta SQL
    # Es más fácil para la Alerta consultar una tabla que un archivo suelto
    spark_df.write.mode("overwrite").saveAsTable(f"{catalog}.default.ma_table_scores")

print(f"Proceso completado. Modelo registrado y predicciones guardadas en {catalog}.default.ma_table_scores")