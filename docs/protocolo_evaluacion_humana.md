# Protocolo de evaluación humana ciega

Documento operativo para quienes coordinan y para quienes evalúan. El diseño y sus fundamentos
están en `docs/metodologia.md` §5.

## Roles

- **Coordinación** (1 persona): prepara el entorno, asigna particiones y no evalúa la partición
  test.
- **Evaluadores** (se recomiendan 3 o más): no participaron en el ajuste del modelo, o al menos no
  han visto los rankings de las consultas que evaluarán.

## Antes de la primera ronda

1. Ejecuta `make web-api` y `make web`, o `docker compose up -d`, y abre la sección
   **Evaluación → Laboratorio**.
2. Revisa `eval/query_bank.json`. Si el equipo quiere añadir consultas propias, agrégalas **antes**
   de abrir la ronda. Su partición la asigna el hash del `id`; no la elijas a mano.
3. Congela el banco: anota en el acta el hash SHA-256 de `eval/query_bank.json`
   (`shasum -a 256 eval/query_bank.json`).

## Instrucciones para cada evaluador (léelas en voz alta)

1. Escribe un identificador estable (iniciales o seudónimo) y úsalo en todas las sesiones.
2. Elige la partición que te asignaron. **dev** y **calibration** sirven para mejorar el sistema;
   **test** es la evaluación final y se hace una sola vez.
3. Para cada destino califica su compatibilidad **con la consulta**, no su calidad general:
   - 0 · nada compatible: contradice lo pedido o no tiene relación.
   - 1 · poco compatible: cumple algo menor.
   - 2 · compatible: cumple lo principal.
   - 3 · muy compatible: cumple casi todo lo pedido.
4. Si la consulta tiene una exclusión (“sin playa”), marca si el destino **cumple** o **incumple**,
   o “no sé”.
5. Marca **evidencia insuficiente** cuando no puedas juzgar con lo mostrado y tu conocimiento.
   Es preferible marcarla a adivinar.
6. No busques el ranking del sistema durante la sesión. Las posiciones solo aparecen en el reporte
   agregado.
7. Atajos: teclas 0–3 en el destino enfocado. Una consulta tarda normalmente entre 2 y 4 minutos.

## Diseño de la ronda

- **Acuerdo:** al menos 10 consultas deben ser juzgadas por 2 o más evaluadores (unos 30 ítems
  compartidos) para estimar el alfa de Krippendorff. Interpretación orientativa: α ≥ 0.667 es
  aceptable para conclusiones tentativas; α ≥ 0.8, fiable.
- **Cobertura:** idealmente todas las consultas de test con 2 o más evaluadores.
- **Orden de trabajo:** dev → ajustes del equipo → calibration → congelar el modelo → test,
  **una vez**.

## Después de la ronda

1. Exporta los juicios: `GET /api/lab/export?format=csv` (enlace en el laboratorio).
2. Revisa el reporte (`GET /api/lab/report` o el panel): nDCG@5, P@5, relevancia media del
   bottom 5, errores de restricción, falsos negativos y acuerdo.
3. Registra los errores por tipo: vocabulario, negación, confusión geográfica, falta de evidencia,
   sesgo por volumen.
4. Cualquier cambio del modelo se valida en **calibration**. Si se cambia algo después de ver
   **test**, test deja de ser una evaluación final; documéntalo como tal.

## Qué no hacer

- No inventar ni completar juicios ajenos.
- No editar `eval/judgments.jsonl` a mano. Si alguien se equivoca, vuelve a guardar la consulta:
  prevalece el último juicio.
- No usar test para elegir pesos.
