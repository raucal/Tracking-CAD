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

## FoundationPose: NVIDIA y la variante de SDU

- Proyecto oficial: https://github.com/NVlabs/FoundationPose.
- Variante SDU utilizada: https://gitlab.sdu.dk/robot_learning/vision/foundation_pose.
- Revisión SDU fijada: `f4b04cd30bdae0b72a0a548192965782b5553587`.
- Base oficial identificada en ese historial: `a1b694b83e633c2cb6115b9063d940a687759392`.

Esta variante conserva el historial de NVIDIA y añade herramientas presentes
en el repositorio de SDU:

| Archivo o componente | Función |
| --- | --- |
| `real_sense_interface.py` | Captura con Intel RealSense. |
| `run_video_demo.py` | Demo para vídeo/captura. |
| `run_single_pose_robot_demo.py` | Integración de pose y demostración robótica. |
| `grasp_config_ui.py` | Interfaz de configuración de agarres y transformaciones. |
| `UR5eGripper/` | Recursos del robot y gripper utilizados por las herramientas. |
| `Utils.py` | Ajuste de dtype y dispositivo al calcular ventanas de proyección. |
| Archivos de instalación y documentación | Adaptación del entorno y sus instrucciones. |

Frente a esa base oficial, `estimater.py` no presenta cambios en esta revisión.
Las herramientas de SDU no se atribuyen a CADTracking. Las configuraciones locales
no versionadas de agarre (`grasp_config.json`, `grasp_cfd_base_config.json` y
`grasp_peg_config.json`) permanecen locales y no forman parte del commit fijado.

**Acceso SDU:** este submódulo requiere las credenciales/permisos correspondientes
en GitLab. No se ha republicado su código en GitHub. La referencia coincide con
`origin/main` registrado localmente, pero la comprobación remota desde este equipo
no se pudo completar por falta de autenticación. Un clon público de CADTracking
no concede acceso al repositorio de SDU.

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

Con acceso a ambos servicios:

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

Si todavía no se dispone de acceso a SDU, se puede recuperar SAM-6D por separado:

```bash
git submodule update --init --recursive models/SAM-6D
```

No usar `git submodule update --remote` para reproducir la instalación: ese comando
puede avanzar las revisiones. El commit del repositorio principal fija los hashes
exactos; el campo `branch` solo identifica la rama de desarrollo.

Los entornos, pesos, extensiones compiladas y datos preparados se instalan aparte:
consultar [la guía de instalación](../installation/README.md).
