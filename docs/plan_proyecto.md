# Plan de proyecto y criterios de calidad

## Objetivo verificable

El 20/11/2026, una persona debe poder escribir una preferencia turística en español y recibir, en
menos de tres segundos después del primer arranque, cinco destinos más relacionados y cinco menos
relacionados. La demostración debe funcionar sin internet y cada recomendación debe poder explicarse
con información del train de RestMex 2025.

## Alcance del desafío

- Fuente permitida: `data/Rest-Mex_2025_train_clean.csv`.
- Inventario: 208,051 reseñas, 40 pueblos, 19 regiones y tres tipos de establecimiento.
- Entrada: texto libre, incluidas preferencias negativas como “sin playa”.
- Salida: dos listas sin duplicados ni intersección, ordenadas por afinidad relativa.
- Despliegue: proceso y pesos alojados en el equipo de exposición.

## Decisiones técnicas

### Perfiles balanceados

El corpus está muy desbalanceado: Tulum tiene 45,345 reseñas y Tapalpa 725. Sumar coincidencias
favorecería a los pueblos con mayor volumen. El constructor promedia los vectores dentro de cada
pueblo y normaliza cada perfil antes del ranking. Las polaridades altas reciben mayor peso, pero las
bajas no se eliminan por completo porque también describen servicios y actividades disponibles.

### Recuperación híbrida

La señal TF-IDF conserva vocabulario preciso del corpus. La recuperación de reseñas representativas
evita que actividades pequeñas desaparezcan en un promedio. Multilingual E5 agrega comprensión de
paráfrasis. El motor normaliza las señales sobre los 40 candidatos y las combina, con expansión
léxica controlada para conceptos turísticos como playa, cenotes, viñedos y aguas termales.

### Interpretación del último grupo

“Menos relacionado” significa menor afinidad con la intención escrita. No significa peor destino,
peor servicio ni menor calificación. La interfaz lo aclara para evitar una interpretación dañina.

## División de trabajo sugerida

| Frente | Responsable sugerido | Entregable | Criterio de cierre |
|---|---|---|---|
| Datos y trazabilidad | Ingeniería de datos | Auditoría, manifiesto y constructor | Hash, conteos y reconstrucción determinista |
| Modelado | Ingeniería de ML | Ranking híbrido y ajuste de pesos | Pruebas ciegas y análisis de errores |
| Producto | Desarrollo | Interfaz, explicaciones y manejo de fallos | Flujo completo sin terminal ni internet |
| Evaluación y presentación | Dueño de calidad | Consultas, rúbrica y demostración | Ensayo cronometrado en el equipo final |

Si el equipo tiene menos integrantes, una persona puede asumir más de un frente. Cada frente debe
tener una segunda persona capaz de arrancarlo durante la exposición.

## Calendario desde el 24/10/2026

### 24 al 30 de octubre

- Congelar el esquema de datos y conservar el hash del CSV.
- Revisar resultados del baseline en al menos 30 consultas escritas por el equipo.
- Etiquetar, antes de ver el ranking, hasta cinco destinos aceptables por consulta.
- Clasificar errores en vocabulario, intención negativa, confusión geográfica o falta de evidencia.

### 31 de octubre al 7 de noviembre

- Ajustar pesos con las consultas etiquetadas, sin optimizar consulta por consulta.
- Medir Precision@5, Hit@1, latencia p50/p95 y estabilidad entre ejecuciones.
- Probar consultas cortas, faltas ortográficas, acentos y frases con exclusiones.
- Congelar una versión candidata y guardar un reporte de evaluación.

### 8 al 12 de noviembre

- Preparar el segundo avance del 13/11/2026 con arquitectura, resultados parciales y dos errores
  conocidos.
- Ejecutar la aplicación con Wi-Fi desactivado.
- Medir carga inicial y tiempo por consulta en el equipo que se llevará a clase.

### 14 al 19 de noviembre

- Realizar dos ensayos completos con tres consultas desconocidas para quien opera la interfaz.
- Grabar un video corto como respaldo visual, sin sustituir la demostración funcional.
- Copiar proyecto, entorno, índices y modelo a una segunda ubicación recuperable.
- Congelar cambios 24 horas antes de la evaluación.

## Rúbrica interna

| Dimensión | Medición | Umbral de aceptación |
|---|---|---|
| Relevancia | Precision@5 sobre consultas ciegas | Al menos 0.70 |
| Primer resultado | Hit@1 | Al menos 0.80 |
| Contrato | Cinco superiores y cinco inferiores | 100% de consultas |
| Estabilidad | Mismo orden con la misma consulta | 100% |
| Rendimiento | Latencia tras carga | p95 menor a 3 s |
| Operación | Prueba sin Wi-Fi | 100% |
| Explicabilidad | Evidencia visible del train | En cada resultado |

## Guion de demostración

1. Mostrar el manifiesto y explicar que el índice solo leyó train.
2. Ejecutar una consulta de playa, una cultural y una con una exclusión.
3. Abrir la evidencia de un resultado para conectar el ranking con el corpus.
4. Cambiar a los cinco menos relacionados y aclarar su significado.
5. Cerrar con métricas del conjunto ciego y una limitación real.

## Riesgos y contingencias

- **El modelo denso no carga:** reiniciar en modo léxico; el contrato de salida sigue funcionando.
- **No hay red:** el modelo y los índices ya están dentro del proyecto.
- **El puerto está ocupado:** usar `streamlit run app.py --server.port 8502`.
- **Una consulta es ambigua:** mostrar coincidencias y pedir una preferencia más específica.
- **Resultados discutibles:** conservar la consulta exacta, registrar el caso y compararlo con la
  rúbrica humana; no cambiar pesos durante la exposición.

