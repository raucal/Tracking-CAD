# Instalación de CAD Tracking

Esta es la guía de instalación y preparación del sistema. El [README principal](../README.md) ofrece el arranque rápido. La [referencia](../docs/REFERENCE.md) explica directorios, comandos y todos los parámetros.

## 1. Si ya utilizas este servidor

El runtime, los pesos y las extensiones están instalados. Desde la raíz de CADTracking:

```bash
bash run.sh settings
bash run.sh sam doctor
bash run.sh ui
```

Abre `http://127.0.0.1:8770`. No hace falta reinstalar ni activar un entorno a mano: el lanzador usa los Python de `config/settings.json`.

La instalación existente utiliza `../.venvs/assembly-runtime` para SAM6D/PEM y `../.venvs/foundationpose` para FoundationPose. El primero depende del Python de Conda `dinobot_env`; el segundo hereda bibliotecas del runtime compartido. Conservar esos entornos y sus rutas mantiene esa instalación funcionando. Una copia de un venv no es un entorno portable: sus ejecutables y archivos `.pth` pueden apuntar a rutas de su máquina de origen.

## 2. Qué hace falta en otra máquina

La referencia comprobada es Linux, Python 3.10, PyTorch 2.5.1 con CUDA 12.4, torchvision 0.20.1 y NumPy 1.26.4. Las extensiones locales también se compilaron con CUDA Toolkit 12.5. Los valores exactos de paquetes están en los constraints de `pipeline/setup/`.

Necesitas:

- Una copia completa de esta integración, incluidos `pipeline/`, `config/` y los modelos con sus adaptaciones locales. Clonar únicamente los repositorios oficiales no proporciona la interfaz ni los adaptadores del proyecto.
- GPU NVIDIA, controlador compatible y memoria suficiente para el segmentador elegido. PEM y FoundationPose usan CUDA; el seguimiento no funciona con una instalación de PyTorch solo CPU.
- Compilador C/C++, CMake, OpenMP, Eigen3, Boost system/program_options y CUDA Toolkit con `nvcc` si se deben compilar las extensiones.
- Blender 3.3.1 para generar plantillas. Configura la ruta a su ejecutable; instalar el paquete Python `bpy` no sustituye ese ejecutable.
- SDK/permisos USB de RealSense si se utiliza cámara. Para analizar archivos guardados no hace falta conectar una cámara.
- Espacio para pesos persistentes, bibliotecas, compilaciones, plantillas y resultados. Los pesos SAM/DINOv2/PEM de esta instalación ocupan aproximadamente 4,9 GiB, además de FoundationPose.

La receta siguiente prepara otra máquina; no se ha creado un entorno nuevo al reorganizar esta documentación. La compatibilidad final del controlador, compilador y extensiones se comprueba con los diagnósticos del apartado 8.

## 3. Crear el runtime base en una instalación nueva

Desde la raíz de tu copia de CADTracking, con Conda disponible:

```bash
conda create -n cad-runtime python=3.10 pip
conda activate cad-runtime
mkdir -p ../.venvs
python -m venv --system-site-packages ../.venvs/assembly-runtime
../.venvs/assembly-runtime/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

El comando de PyTorch usa las ruedas CUDA 12.4 de su [documentación oficial para 2.5.1](https://pytorch.org/get-started/previous-versions/#v251). No sustituirlas por ruedas CPU para esta configuración.

Ajusta en `config/settings.json`:

```json
{
  "paths": {
    "runtime_python": "../../.venvs/assembly-runtime/bin/python",
    "foundationpose_python": "../../.venvs/foundationpose/bin/python"
  }
}
```

El fragmento muestra solo los campos a editar: conserva los demás campos de `settings.json`. Cada ruta se resuelve respecto a ese JSON, no respecto a la terminal. Puedes usar otro nombre o ubicación para los entornos modificando esas rutas.

El paquete de dependencias SAM6D incluye los imports necesarios para datos, interfaz, ISM, PEM, cámara y renderizado. Los constraints limitan sus versiones; no son por sí solos una receta de Conda ni incluyen compiladores, pesos o el ejecutable Blender.

## 4. Configurar rutas, compilador y Blender

Comprueba las rutas efectivas:

```bash
bash run.sh settings
nvidia-smi
nvcc --version
```

`paths.sam6d` debe apuntar a `models/SAM-6D`; `paths.foundationpose`, a `models/FoundationPose`. Los repositorios, checkpoints, extensiones y plantillas se organizan como describe el README principal.

Instala o conserva Blender 3.3.1 y configura `paths.blender` con su ejecutable real. El valor distribuido apunta a una instalación hermana `../.runtime/blender/`. En otra máquina hay que adaptar esa ruta.

Si `nvcc` no está en PATH, define `CUDA_HOME` con el directorio de tu CUDA Toolkit y añade su `bin` a PATH. El instalador detecta el toolkit mediante `nvcc` cuando `CUDA_HOME` no está definido. `TORCH_CUDA_ARCH_LIST` permite fijar la arquitectura al compilar PEM; si no se proporciona, se obtiene de la GPU disponible. `runtime.build_jobs` controla el paralelismo que el lanzador exporta como `MAX_JOBS`; el instalador de FoundationPose mantiene además algunos pasos limitados a dos trabajos.

## 5. Instalar SAM6D y PEM

```bash
bash run.sh install
```

El comando:

1. Instala los paquetes de `pipeline/setup/requirements-sam6d.txt`, respetando `constraints-assembly-runtime.txt`.
2. Descarga o reutiliza SAM ViT-H, DINOv2, FastSAM y PEM en `checkpoints/`.
3. Conserva enlaces relativos desde los repositorios de los modelos hacia esos pesos persistentes.
4. Compila PointNet2 en `pipeline/native/pointnet2/` si no encuentra su extensión.
5. Ejecuta el diagnóstico SAM6D/PEM.

La descarga automática incluye SAM ViT-H. Seleccionar ViT-B o ViT-L requiere disponer también de sus checkpoints compatibles; la interfaz indica cuáles están disponibles.

No se descargan aquí los pesos de FoundationPose, ni se prepara el dataset de ejemplo, ni se renderizan sus plantillas. Esos pasos están separados para no reconstruirlos cada vez que se instala una biblioteca.

## 6. Instalar FoundationPose

Coloca los pesos de inferencia oficiales en:

```text
models/FoundationPose/weights/
├── 2023-10-28-18-33-37/
│   ├── model_best.pth
│   └── config.yml
└── 2024-01-11-20-02-45/
    ├── model_best.pth
    └── config.yml
```

Son el refiner y scorer indicados en la [preparación oficial de FoundationPose](https://github.com/NVlabs/FoundationPose#data-prepare). Esta integración usa esos pesos de inferencia; no necesita descargar el dataset de entrenamiento.

Después:

```bash
bash run.sh install-foundationpose
```

El instalador crea el venv si falta, comparte las bibliotecas del runtime base mediante `assembly_runtime.pth`, instala las dependencias específicas con sus constraints y compila las extensiones de FoundationPose. PyTorch3D usa la referencia `v0.7.9`; nvdiffrast se fija al commit `729261dc64c4241ea36efda84fbf532cc8b425b8`. `mycpp` se compila en `pipeline/native/foundationpose/`.

La instalación existente de PyTorch3D procede de un directorio temporal ya ausente: su versión/binario están auditados, pero ese directorio no constituye una receta reproducible de fuentes. Una recompilación en otra máquina necesita validarse contra la versión de PyTorch y el compilador seleccionados.

## 7. Datos y cámara

### Dataset de ejemplo

Si conservas `data/prepared/` completo, usa directamente sus datos y plantillas. No se regeneran al arrancar la interfaz o una suite.

Para preparar otro dataset, edita `data/dataset.json` y escribe en una carpeta nueva:

```bash
bash run.sh prepare --config data/dataset.json --output data/prepared-new
bash run.sh inspect --manifest data/prepared-new/manifest.json
bash run.sh sam render --manifest data/prepared-new/manifest.json
```

La preparación crea CAD centrados, RGB, profundidad raw y calibración. El renderizado genera 42 vistas por CAD, sus máscaras y coordenadas XYZ métricas. Cambia también `paths.prepared` en settings y `manifest` en el perfil de suite para seleccionar esa carpeta. Usa `--part CLAVE_CAD` con `sam render` si solo quieres preparar las vistas de una pieza.

`--repair-topology` es opcional y necesita `manifold3d`. La auditoría completa de proximidad de plantillas puede requerir dependencias de trimesh adicionales, como `rtree`. La detección y el seguimiento habituales no dependen de ejecutar esa auditoría.

Borrar toda `data/` también elimina `dataset.json`; hay que recuperarlo antes de reconstruir los datos. La interfaz con CAD propios prepara sus referencias en `results/interface/cache/`, pero sigue leyendo el manifest del ejemplo al arrancar.

### Intel RealSense

Instala el SDK y las reglas/permisos USB siguiendo la [guía oficial de librealsense](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md). El paquete Python `pyrealsense2` figura en las dependencias del runtime.

La cámara debe estar conectada al ordenador que ejecuta el servicio, también cuando se usa un navegador remoto. Comprueba desde la interfaz que aparece en la lista y que otro proceso no la mantiene abierta. La adquisición se abre al pulsar conectar; los comandos anteriores no capturan vídeo ni mueven el robot.

## 8. Validar antes de utilizar

```bash
bash run.sh settings
bash run.sh sam doctor
bash run.sh ui --preload none
```

En la web revisa CUDA, pesos, disponibilidad de FoundationPose y los CAD de ejemplo. Cierra esa instancia con Ctrl+C y arranca normalmente con `bash run.sh ui` para precargar modelos.

Para verificar las dependencias con el Python que hayas configurado:

```bash
../.venvs/assembly-runtime/bin/python -m pip check
../.venvs/foundationpose/bin/python -m pip check
bash run.sh test
```

Si cambiaste las rutas de los entornos, adapta los dos primeros comandos a las rutas mostradas por `run.sh settings`. Una matriz válida o un diagnóstico de módulos no certifican precisión de pose; revisa la proyección del CAD sobre tu escena.

## 9. Qué archivos usa realmente la instalación

`installation/` contiene solo esta guía. Los archivos que ejecuta el programa están en `pipeline/setup/`:

| Archivo | Función |
| --- | --- |
| `install_models.sh` | Instalar dependencias/pesos SAM6D y compilar PEM. |
| `install_foundationpose.sh` | Preparar el venv y las extensiones FoundationPose. |
| `requirements-sam6d.txt` | Dependencias directas de datos, interfaz, SAM6D, PEM y renderizado. |
| `requirements-foundationpose.txt` | Dependencias específicas de FoundationPose. |
| `constraints-assembly-runtime.txt` | Versiones efectivas del runtime compartido Linux. |
| `constraints-foundationpose.txt` | Versiones efectivas del venv FoundationPose. |

Se necesitan dos listas de versiones porque son dos entornos con paquetes y algunas versiones diferentes. Conservarlas evita que una instalación cambie bibliotecas sin querer. Modificarlas requiere validar los backends; no se deben combinar los entornos eliminando uno de esos archivos.

Las guías oficiales dentro de `models/` pertenecen a los proyectos originales. El arranque habitual está en el README principal; los detalles operativos y parámetros están en la referencia, y esta guía cubre la instalación.


## Recuperar los repositorios de modelos

SAM-6D y FoundationPose están fijados como submódulos. Antes de instalar, ejecutar
`git submodule sync --recursive` y `git submodule update --init --recursive`.
FoundationPose utiliza la variante de SDU, que requiere acceso a su GitLab.
Consultar [procedencia, revisiones y clonación parcial](../docs/PROVENANCE.md).
