# Adaptaciones FoundationPose de SDU

El código oficial está en el submódulo `../../models/FoundationPoseOfficial`, fijado a
`a1b694b83e633c2cb6115b9063d940a687759392` de NVlabs/FoundationPose.
Esta carpeta conserva por separado la adaptación de SDU en
`f4b04cd30bdae0b72a0a548192965782b5553587`:

- `files/`: archivos añadidos, con los mismos bytes y nombres que la revisión SDU.
- `sdu.patch`: cambios de SDU en archivos oficiales, sin los archivos añadidos.
- `provenance.json`: repositorios, revisiones, inventario y hashes de compatibilidad.
- `LICENSE.upstream`: licencia NVIDIA conservada. La separación no cambia las licencias.
- `runtime.py` y `run.py`: nuestros adaptadores para usar la separación.

`files/UR5eGripper/` conserva los recursos locales del gripper. No estaban
versionados en el checkout SDU; esa procedencia está diferenciada en el JSON.
Los tres JSON personales de agarre se conservan localmente e ignoran en Git.

CADTracking utiliza el algoritmo oficial y aplica el cambio de dtype/dispositivo
de `Utils.py` en memoria. El archivo oficial no se modifica. Una copia derivada
de ese archivo se regenera en `.cache/foundationpose_sdu/` y se valida por hash.
Las diferencias de documentación, entorno y gitignore se conservan en el parche,
pero no se aplican al runtime: su instalación sigue la guía de CADTracking.

Para ejecutar manualmente una herramienta SDU, con el Python de FoundationPose y
desde la raíz de CADTracking:

```bash
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py grasp_config_ui.py
```

Los scripts se han conservado tal cual; sus opciones y dependencias específicas
son las de SDU. No se han ejecutado demos ni comandos del robot durante esta separación.
Los pesos permanecen fuera de Git, en `checkpoints/foundationpose/`, enlazados desde
el submódulo. Clonar el repositorio ya no requiere acceso a GitLab SDU.


## Comprobación de arranque

`--check` debe ir antes del nombre del script. Carga sus importaciones sin ejecutar
el bloque principal: no abre la cámara ni conecta al robot.

```bash
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py --check run_video_demo.py
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py --check --teleop-root ../PDDL/DINOBOT/alignment_and_replay-main run_single_pose_robot_demo.py
```

La demo robótica depende del paquete externo `teleop_setup`. `--teleop-root`
selecciona la carpeta que lo contiene; también puede configurarse mediante
`SDU_TELEOP_ROOT`. Este paquete no se modifica ni se copia a CADTracking.
Sin `--check`, la demo original conecta y envía un movimiento al robot al arrancar.
Los argumentos posteriores al nombre del script, incluido `--help`, se pasan al
programa SDU. Para la demo de vídeo hay que indicar `--mesh_file` y `--mesh_scale`
si no están presentes sus datos originales de ejemplo.

Se comprobó el arranque con el CAD control_panel: ventana de agarres en pantalla
virtual, carga de modelos de vídeo y recepción RGB-D RealSense a 640×480. La demo
robótica se verificó únicamente con `--check`, sin conexión ni movimientos.
