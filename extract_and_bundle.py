import json
import os
import re
import shutil

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DIR = "databricks_raw"
DEST_RESOURCES = "resources"
# Carpetas para los archivos de código/definición (Contenido)
PATHS = {
    "dashboard": "dashboards",
    "alert": "alerts_sql",   # Guardaremos el SQL de la alerta aquí
    "query": "queries",      # SQL de queries puras
    "job": "jobs_config"     # Copia limpia del JSON del job
}

REPLACEMENTS = {
    "dev_catalog": "${var.catalog_name}",
    "af6acdfdf7d45afd": "${var.warehouse_id}"
}

def sanitize(name):
    return re.sub(r'\W+', '_', name.lower()).strip('_')

def clean_text(text):
    if not text: return ""
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except: return 

    display_name = data.get("display_name", data.get("name", os.path.basename(file_path).split('.')[0]))
    clean_id = sanitize(display_name)
    
    # 1. CASO DASHBOARD
    if "serialized_dashboard" in data:
        print(f"📊 Dashboard: {clean_id}")
        content = clean_text(data["serialized_dashboard"])
        os.makedirs(PATHS["dashboard"], exist_ok=True)
        with open(f"{PATHS['dashboard']}/{clean_id}.json", "w") as f:
            f.write(content)
        
        save_yaml("dashboards", clean_id, f"""
    {clean_id}:
      display_name: "{display_name}"
      file_path: ../../{PATHS['dashboard']}/{clean_id}.json
      warehouse_id: ${{var.warehouse_id}}""")

    # 2. CASO ALERTA (Ahora extraemos el SQL a un archivo aparte para mayor limpieza)
    elif "evaluation" in data:
        print(f"🔔 Alerta: {clean_id}")
        query = clean_text(data.get("query_text", ""))
        os.makedirs(PATHS["alert"], exist_ok=True)
        with open(f"{PATHS['alert']}/{clean_id}.sql", "w") as f:
            f.write(query)
            
        save_yaml("alerts", clean_id, f"""
    {clean_id}:
      display_name: "{display_name}"
      warehouse_id: ${{var.warehouse_id}}
      query_text: |
{re.sub('^', '        ', query, flags=re.M)}""") # Indentación automática del SQL

    # 3. CASO JOB
    elif "settings" in data or "tasks" in data:
        print(f"⚙️ Job: {clean_id}")
        # Limpiamos el JSON del Job (rutas de notebooks, etc)
        job_json = clean_text(json.dumps(data, indent=2))
        os.makedirs(PATHS["job"], exist_ok=True)
        with open(f"{PATHS['job']}/{clean_id}.json", "w") as f:
            f.write(job_json)
        
        save_yaml("jobs", clean_id, f"""
    {clean_id}:
      name: "{display_name}"
      # Aquí el bundle suele preferir la definición inline en el YAML, 
      # pero podemos usar el JSON como referencia o volcarlo aquí.
      tasks: 
        # (Lógica simplificada para el ejemplo)
        - task_key: "task_1"
          notebook_task:
            notebook_path: "../../src/train_model.py"
""")

    # 4. CASO QUERY SQL
    elif "query_text" in data:
        print(f"🔍 Query: {clean_id}")
        sql = clean_text(data["query_text"])
        os.makedirs(PATHS["query"], exist_ok=True)
        with open(f"{PATHS['query']}/{clean_id}.sql", "w") as f:
            f.write(sql)
        save_yaml("queries", clean_id, f"""
    {clean_id}:
      display_name: "{display_name}"
      query_text: |
{re.sub('^', '        ', sql, flags=re.M)}""")

def save_yaml(category, clean_id, content):
    folder = f"{DEST_RESOURCES}/{category}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{clean_id}.yml", "w") as f:
        f.write(f"resources:\n  {category}:{content}")

if __name__ == "__main__":
    # Limpieza total de carpetas destino
    for d in [DEST_RESOURCES] + list(PATHS.values()):
        if os.path.exists(d): shutil.rmtree(d)
    
    if os.path.exists(RAW_DIR):
        for file in os.listdir(RAW_DIR):
            if file.endswith(".json") or file.endswith(".yml"):
                process_file(os.path.join(RAW_DIR, file))
        print("\n✨ ¡Fábrica completa! Estructura generada para Dashboards, Alertas, Jobs y Queries.")