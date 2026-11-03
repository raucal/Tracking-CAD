# CAD Tracking

Detecta una pieza CAD en una imagen o en la Intel RealSense, calcula su pose con PEM/FoundationPose y permite seguirla en directo. Se utiliza desde el navegador.

**Esta página es el arranque rápido.** Los archivos y parámetros se explican en la [referencia](docs/REFERENCE.md). Si este equipo todavía no tiene entornos, pesos o datos preparados, sigue primero la [instalación](installation/README.md).

## Código original, SDU y nuestra integración

- `models/SAM-6D`: submódulo de [nuestro fork](https://github.com/raucal/SAM-6D),
  basado en [SAM-6D oficial](https://github.com/JiehongLin/SAM-6D), con las adaptaciones de CADTracking.
- `models/FoundationPose`: submódulo de la [variante SDU](https://gitlab.sdu.dk/robot_learning/vision/foundation_pose),
  basada en [FoundationPose de NVIDIA](https://github.com/NVlabs/FoundationPose).
- `pipeline/`, `config/` y `run.sh`: nuestra integración, interfaz y suites.

Las revisiones exactas, atribuciones y comandos de clonación están en
[Procedencia y adaptaciones](docs/PROVENANCE.md). El submódulo de SDU requiere acceso
a su GitLab; sus adaptaciones no se han republicado en GitHub. Los pesos y datos se
instalan aparte.

## 1. Abrir el programa

En el servidor actual, abre una terminal y ejecuta:

```bash
cd ~/projects/CADTracking
bash run.sh ui
```

Si guardaste el proyecto en otro lugar, entra en esa carpeta antes de ejecutar `run.sh`.

Abre **http://127.0.0.1:8770** en el navegador. Deja la terminal abierta y espera a que termine la carga de modelos. En esta instalación no necesitas activar Conda manualmente: el lanzador selecciona el Python configurado.

Para detener el programa pulsa **Ctrl+C** en esa terminal.

### Si entras por VS Code o SSH

Ejecuta el comando anterior en la **terminal del servidor**. En VS Code Remote SSH abre **Puertos/Ports**, reenvía **8770** y abre su dirección local en el navegador.

Como alternativa, ejecuta este túnel en una terminal de tu ordenador:

```bash
ssh -N -L 8770:127.0.0.1:8770 usuario@servidor
```

Deja también esa terminal abierta y entra en **http://127.0.0.1:8770**. Esta interfaz web no necesita X11. La RealSense debe estar conectada al servidor que ejecuta el programa.

## 2. Hacer la primera prueba

**Sin cámara:** despliega **Probar con los datos del ensamblaje** y selecciona una captura y un CAD. Si quieres pose, selecciona **RGB-D** y activa la estimación PEM o elige **FoundationPose**. Después pulsa **Ejecutar ejemplo**.

**Con RealSense:**

1. Selecciona **Intel RealSense en directo** y pulsa **Abrir cámara**.
2. Selecciona el archivo CAD e indica sus unidades; normalmente, mm.
3. Elige **FoundationPose** como motor de pose para seguir la pieza.
4. Pulsa **Localizar e iniciar seguimiento rápido** y espera a la primera localización.

El seguimiento usa modelos ya cargados. SAM recupera la pieza al perder el seguimiento y comprueba cada **10 segundos** como respaldo. Puedes cambiar ese intervalo en la web.

**Iniciar detección en directo** repite análisis completos; para seguir una pose con más fluidez utiliza **seguimiento rápido**. Para guardar imágenes o grabar vídeo están los botones de cámara. Los pasos detallados, incluida la caja manual sin SAM, están en la [referencia de la interfaz](docs/REFERENCE.md#interfaz-imágenes-y-cad).

## 3. Ejecutar una suite y ver el informe

Para medir una suite en la misma GPU, detén antes la interfaz con Ctrl+C. Desde la carpeta del proyecto:

```bash
# Primero, una captura con detección y pose PEM
bash run.sh suite run --frame hmi_panel

# Para ejecutar después todas las capturas del perfil
bash run.sh suite run
```

Cada comando crea una ejecución distinta. El perfil actual calcula pose PEM además de detección; FoundationPose se utiliza desde la interfaz. La primera ejecución puede tardar más por la carga de modelos y referencias.

Al terminar aparece la ruta del informe. Para verlo en el navegador:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory results/dashboard
```

Abre **http://127.0.0.1:8766** y pulsa **Matrices y triedros** para revisar las poses. En remoto reenvía también el puerto **8766** desde VS Code o mediante un túnel SSH. Detén este servidor con Ctrl+C cuando termines.

## 4. Comandos principales

Ejecuta estos comandos desde la raíz del proyecto. Donde aparece `ID`, escribe el nombre real de una carpeta de `results/suites/`; puedes consultarlos con `ls results/suites`.

| Quiero… | Comando |
| --- | --- |
| Abrir la interfaz | `bash run.sh ui` |
| Usar otro puerto | `bash run.sh ui --port 8771` |
| Revisar una suite sin ejecutarla | `bash run.sh suite run --dry-run` |
| Ejecutar una sola captura | `bash run.sh suite run --frame hmi_panel` |
| Ejecutar la suite completa | `bash run.sh suite run` |
| Ejecutar solo detección, sin pose | `bash run.sh suite run --no-pose` |
| Usar FastSAM en la suite | `bash run.sh suite run --segmentor fastsam` |
| Regenerar el informe, sin repetir inferencia | `bash run.sh suite report --output results/suites/ID` |
| Añadir poses a una suite guardada | `bash run.sh suite pose --output results/suites/ID` |
| Crear un ZIP del informe | `bash run.sh suite report --output results/suites/ID --zip` |
| Comprobar CUDA, pesos y plantillas SAM6D/PEM | `bash run.sh sam doctor` |
| Ver las rutas configuradas | `bash run.sh settings` |
| Consultar opciones de la interfaz o suite | `bash run.sh ui --help` / `bash run.sh suite run --help` |

Los comandos de instalación, preparación, reconstrucción de plantillas y diagnóstico avanzado están en la [tabla completa de comandos](docs/REFERENCE.md#todos-los-comandos). No necesitas ejecutarlos cada vez que abres la aplicación.

## 5. Qué configurar y dónde se guarda todo

| Archivo | Cuándo editarlo |
| --- | --- |
| `config/interface.json` | Cambiar puerto, cámara, valores iniciales y seguimiento. |
| `config/suite.json` | Elegir capturas, segmentador, umbrales, pose y salidas de la suite. |
| `config/settings.json` | Cambiar las rutas de los Python, Blender, datos o pesos. |

Para empezar puedes usar los valores actuales. Un argumento de terminal prevalece sobre su perfil. Las rutas dentro de los JSON se interpretan respecto a ese archivo; no necesitas copiar rutas personales en los scripts.

- **Imágenes y vídeos guardados:** `captures/`.
- **Análisis, logs y seguimiento de la interfaz:** `results/interface/`.
- **Ejecuciones originales de suites:** `results/suites/`.
- **Informes:** `results/reports/`; `results/dashboard` apunta al último exportado.

**Conserva `data/` y `checkpoints/`.** El dataset, manifest y plantillas del catálogo no se reconstruyen automáticamente al abrir una suite o la interfaz. Consulta [qué se puede reconstruir y cómo](docs/REFERENCE.md#plantillas-manifest-y-reconstrucción).

## 6. Si algo falla

| Qué ocurre | Primer paso |
| --- | --- |
| El puerto 8770 está ocupado | Utiliza la instancia que ya está abierta, detenla desde su terminal o arranca con `--port 8771` y abre ese puerto. |
| La web no abre desde tu ordenador | Comprueba que el servicio sigue en marcha y que has reenviado el puerto correcto. |
| No aparece la RealSense | Comprueba que esté conectada al servidor y que otro programa no la esté utilizando. |
| Faltan modelos, datos o el Python configurado | Ejecuta `bash run.sh settings` y `bash run.sh sam doctor`; sigue la guía de instalación según el error. |
| No aparecen matrices o triedros | Abre el informe actual, comprueba que la suite tenía PEM activado y revisa los errores de pose. Sin detección aceptada puede no haber pose. |
| Un análisis falla | Lee el mensaje de la interfaz y el enlace a `error.log`. |

Para continuar: [referencia de archivos, parámetros y funcionamiento](docs/REFERENCE.md) · [instalación y reconstrucción de datos](installation/README.md).

## Elegir ensamblaje

El registro compartido es `PDDL/Assembly CAD/assemblies.json`. Los CAD originales
están separados en `Assembly parts/printed_assembly/` y
`Assembly parts/printed_assembly_2/`. Consulta la [guía de los ensamblajes](../PDDL/Assembly%20CAD/README.md).

La suite admite `assembly` en `config/suite.json` y `--assembly printed_assembly`
en terminal. La preparación admite `--assembly printed_assembly_2`, que usa un
directorio de salida independiente. La interfaz incluye un selector de ensamblaje
y pieza para usar sus STL originales. La segunda versión aún necesita sus capturas
y plantillas antes de ejecutar una suite; no reutiliza las de la primera.
