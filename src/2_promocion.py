import mlflow

client = mlflow.DatabricksWorkspaceClient()

# Definimos las rutas usando tu modelo real de casas
origen_dev = "dev_catalog.default.house_model"
destino_qa = "qa_catalog.default.house_model_qa" # El catálogo de QA que elijas

# LA GRÚA BUSCA EL CAMPEÓN: En lugar de buscar la última versión a ciegas, 
# tu grúa puede ir directamente a buscar la versión que tenga la medalla @champion
version_champion = client.get_model_version_by_alias(origen_dev, "champion")

# Clonamos el archivo hacia el catálogo de QA sin tocar una sola matemática
mlflow.register_model(
    model_uri=version_champion.source, # Viaja el archivo binario exacto de hace meses
    name=destino_qa
)

print(f"🚀 ¡Éxito! El modelo de casas con sello @champion ha sido clonado en QA.")