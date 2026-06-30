import os
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import SparkSession

# 1. Inicializar Spark y configurar MLflow para Unity Catalog
spark = SparkSession.builder.getOrCreate()
mlflow.set_registry_uri("databricks-uc")

# 2. TU MEJORA: Leer el catálogo directamente desde la variable de entorno del clúster
# Si el script corre en local o la variable no está configurada, hace fallback a "dev_catalog"
catalogo_actual = os.getenv("ENV_CATALOG", "dev_catalog")

print(f"🔍 [INFERENCIA] Detectado entorno operativo desde el clúster: {catalogo_actual}")

# =====================================================================
# CONFIGURACIÓN DINÁMICA DE RUTAS SEGÚN EL ENTORNO
# =====================================================================
if catalogo_actual in ["dev_catalog", "enriched_dev"]:
    nombre_modelo = f"{catalogo_actual}.default.house_model"
    tabla_destino = f"{catalogo_actual}.default.predicciones_diarias_dev"
    
elif catalogo_actual in ["qa_catalog", "enriched_qa"]:
    nombre_modelo = f"{catalogo_actual}.default.house_model_qa"
    tabla_destino = f"{catalogo_actual}.default.predicciones_diarias_qa"
else:
    raise ValueError(f"❌ Error: Ambiente no reconocido por la infraestructura: {catalogo_actual}")

# =====================================================================
# EJECUCIÓN DEL PIPELINE DE PREDICCIÓN DIARIA
# =====================================================================
print(f"🔌 Cargando el cerebro del modelo: {nombre_modelo}")

try:
    # A. Cargamos la última versión del modelo disponible en este catálogo
    modelo_cargado = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}/latest")
    
    # B. SIMULACIÓN: Datos de entrada reales de hoy en el banco
    print("📈 Recibiendo nuevas solicitudes de vivienda...")
    datos_nuevos_hoy = pd.DataFrame({
        "sqft": [1200, 2800, 4200]
    })
    
    # C. El modelo ejecuta la inferencia
    print("🧠 Calculando predicciones de precios...")
    predicciones = modelo_cargado.predict(datos_nuevos_hoy)
    
    # D. Estructuramos el dataframe de salida
    datos_nuevos_hoy["price_predicted"] = predicciones
    datos_nuevos_hoy["execution_date"] = pd.Timestamp.now()
    
    # E. PERSISTENCIA: Guardamos los resultados en la tabla definitiva de este entorno
    print(f"💾 Guardando resultados en la tabla: {tabla_destino}")
    spark_df = spark.createDataFrame(datos_nuevos_hoy)
    spark_df.write.mode("append").saveAsTable(tabla_destino)
    
    print(f"🎉 ¡Éxito absoluto! Muestra de las predicciones en {catalogo_actual}:")
    display(spark_df)

except Exception as e:
    print(f"❌ Error en el proceso de predicción: {e}")
    print(f"Verifica que el modelo '{nombre_modelo}' exista realmente en este catálogo.")