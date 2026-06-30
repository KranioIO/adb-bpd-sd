import os
import yaml
from pathlib import Path

# 1. Detectar de forma automática la raíz del proyecto
ROOT_DIR = Path(__file__).parent

# 2. Leer la variable de entorno (por defecto dev_local si estamos en tu PC)
env = os.getenv("APP_ENV", "dev_local")

# 3. Armar la ruta al archivo YAML correspondiente
config_file_path = ROOT_DIR / "config" / f"{env}.yml"

# 4. Cargar el archivo con manejo de errores simple
try:
    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_file_path}")

# 5. Clase limpia para acceder con punto (.) en vez de corchetes
class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

# Instancia global para importar
cfg = Config(**config_data)