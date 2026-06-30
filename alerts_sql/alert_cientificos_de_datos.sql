SELECT 
  execution_date,
  sqft,
  price_predicted,
  -- Creamos una etiqueta de estado para el Dashboard
  CASE 
    WHEN price_predicted < 0 THEN 'Error: Precio Negativo'
    WHEN price_predicted > 1000 THEN 'Alerta: Precio muy alto'
    ELSE 'OK'
  END AS validation_status
FROM ${var.catalog_name}.default.ma_table_scores
ORDER BY execution_date DESC