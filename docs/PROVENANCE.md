# Procedencia de los modelos y adaptaciones

CADTracking integra dos proyectos externos. Su código se conserva en submódulos:
el repositorio principal fija una revisión concreta de cada uno, sin copiar sus
archivos, pesos ni historial dentro de la integración.

## SAM-6D: oficial y adaptaciones para CADTracking

- Proyecto oficial: https://github.com/JiehongLin/SAM-6D.
- Fork que conserva nuestras adaptaciones: https://github.com/raucal/SAM-6D.
- Rama de integración: `cadtracking`.
- Revisión fijada: `ad2365a18b62b3fa86b61b9fb96da8c74d4272f1`.
- Base oficial de esa línea: `1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`.

Los commits `36865b0` y `ad2365a` añaden/adaptan preparación e inferencia del
proyecto, suites, compatibilidad de modelos y visualización de poses PEM.
El algoritmo SAM-6D y sus modelos proceden de los autores originales; nuestra
aportación es la integración y esas adaptaciones, no el método publicado.

## FoundationPose oficial y adaptación SDU separada

- Submódulo oficial: `models/FoundationPoseOfficial`.
- Repositorio: https://github.com/NVlabs/FoundationPose.
- Revisión compatible: `a1b694b83e633c2cb6115b9063d940a687759392`.
- Adaptaciones: `integrations/foundationpose_sdu/`.
- Origen de esas adaptaciones: https://gitlab.sdu.dk/robot_learning/vision/foundation_pose.
- Revisión SDU de referencia: `f4b04cd30bdae0b72a0a548192965782b5553587`.

Los cinco archivos añadidos por SDU se conservan sin editar en `files/`: su
`README.md`, `real_sense_interface.py`, `run_video_demo.py`,
`run_single_pose_robot_demo.py` y `grasp_config_ui.py`. Los cambios sobre archivos
oficiales están en `sdu.patch` (`Utils.py`, entorno, gitignore y documentación).
`provenance.json` conserva revisiones, inventario y hashes. El algoritmo
`estimater.py` coincide con la base oficial.

Los recursos `files/UR5eGripper/` proceden de la instalación local; no estaban
versionados en el checkout SDU. Esa diferencia de procedencia queda registrada.
Los tres JSON personales de agarres permanecen locales y no se publican.

`runtime.py`, `run.py` y el puente `pipeline/foundationpose_sdu.py` son adaptadores
nuestros. Aplican en memoria el cambio SDU de dtype/dispositivo de `Utils.py`,
validando el archivo oficial y el resultado del parche por hash. El submódulo
oficial permanece intacto. La copia derivada es regenerable en `.cache/`.
Las instrucciones de uso están en [la carpeta SDU](../integrations/foundationpose_sdu/README.md).

Los pesos no se publican y se conservan en `checkpoints/foundationpose/`. La antigua
carpeta local `models/FoundationPose/` está protegida por permisos del sistema;
queda ignorada, sin utilizarse, para no borrar sus archivos. No forma parte de un
clon nuevo ni es una dependencia del runtime reorganizado.

Ambos submódulos apuntan ahora a GitHub y clonar CADTracking no necesita acceso al
GitLab de SDU. Las adaptaciones conservan la atribución y la licencia original
en `LICENSE.upstream`; la separación no cambia sus condiciones.

## Código propio de CADTracking

`pipeline/sam6d_ui/` implementa nuestra interfaz, integración de SAM/PEM y
FoundationPose, procesos residentes, captura, seguimiento y recuperación.
`pipeline/suites/`, `pipeline/tools/`, `config/` y `run.sh` contienen nuestras
suites, herramientas, configuración y lanzador. Esta capa está fuera de los
submódulos y se versiona en Tracking-CAD.

Los checkpoints se obtienen aparte según las instrucciones y licencias de sus
autores. Los archivos `LICENSE` de los submódulos se conservan en sus repositorios;
la integración no cambia las condiciones de esos proyectos.

## Clonar y recuperar las revisiones exactas

Los submódulos se descargan desde GitHub:

```bash
git clone --recurse-submodules https://github.com/raucal/Tracking-CAD.git
cd Tracking-CAD
```

En un clon existente:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

No usar `git submodule update --remote` para reproducir la instalación: ese comando
puede avanzar las revisiones. El commit del repositorio principal fija los hashes
exactos; el campo `branch` solo identifica la rama de desarrollo.

Los entornos, pesos, extensiones compiladas y datos preparados se instalan aparte:
consultar [la guía de instalación](../installation/README.md).
