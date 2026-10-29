# Integración SAM-6D

Este directorio contiene la interfaz imagen + CAD (`sam6d_ui/`), el lanzador
`sam6d_gui.py`, la suite (`sam6d_suite.py`), su informe (`sam6d_suite_report.py`),
el diagnóstico de scores (`score_all_cads.py`) y la preparación de datos (`prepare.py`).
`config.json` referencia directamente los CAD, las fotos y el catálogo de PDDL.

```bash
bash SAM6D/run.sh ui --port 8770
bash SAM6D/run.sh sam doctor
bash SAM6D/run.sh suite report --output SAM6D/pipeline/prepared/sam6d/suites/gpu_all_22 --export-dir SAM6D/results/dashboard
```

[Guía de interfaz y configuración](sam6d_ui/README.md).

`prepared/manifest.json` tiene las rutas actuales a los originales.
`prepared/manifest_at_execution.json` conserva los bytes y el hash del manifest
usado por las inferencias históricas; no se han recalculado sus scores al moverlas.
El diagnóstico histórico resuelve ese manifest en su nueva ubicación sin alterar sus hashes.
Los dos enlaces `prepared/asap` y `prepared/asap_dense` apuntan a las únicas mallas
de simulación de `ASAP/prepared/`; conservan el formato compartido del manifest.

ASAP tiene su código, catálogo, resultados y dashboard en `ASAP/`.
`run_runtime.sh` solo configura el entorno compartido y despacha los comandos.


## Interfaz común SAM-6D / FoundationPose y RealSense

La interfaz permite seleccionar SAM-6D/PEM o FoundationPose, ver el directo de una
Intel RealSense conectada al servidor, capturar RGB-D y grabar vídeo con profundidad
y marcas de tiempo. FoundationPose permite seguir las grabaciones.

```bash
cd /home/raca/projects
bash SAM6D/run.sh ui --port 8770
```

Los nuevos trabajos se separan en `SAM6D/results/interface/jobs/sam6d/` y
`SAM6D/results/interface/jobs/foundationpose/`; las capturas y grabaciones están en
`SAM6D/results/interface/camera/`. Las suites conservan sus carpetas.

Consulta [la guía de la interfaz](sam6d_ui/README.md) para arquitectura,
instalación, unidades, cámaras, seguimiento y estructura de archivos.


## Procedencia y mapa de documentos

Las herramientas de esta carpeta son nuestra integración del ensamblaje. Los modelos
de terceros y sus repositorios oficiales, los cambios locales dentro de la copia
SAM-6D y el origen institucional de FoundationPose se explican en
[Procedencia](../README.md#procedencia-modelos-oficiales-y-desarrollo-del-proyecto).

El [mapa de documentación y carpetas](../README.md#estructura-de-la-documentación)
indica qué README consultar y distingue configuraciones, manifests, resultados,
informes, logs y copias históricas. Para usar la web y consultar cada parámetro,
seguir [el manual de interfaz](sam6d_ui/README.md).
