# Referencia de CAD Tracking

Para arrancar y utilizar el programa por primera vez sigue el [README principal](../README.md). Aquí se explica el repositorio, sus comandos, parámetros y archivos. Para preparar otro equipo utiliza la [guía de instalación](../installation/README.md).

Todos los comandos se ejecutan desde la raíz de CADTracking; entra en ella antes de copiarlos. No es necesario leer esta referencia completa para abrir la interfaz.

- [Todos los comandos](#todos-los-comandos)
- [Directorios y archivos](#directorios-y-archivos)
- [Configuración completa](#configuración-completa)
- [Plantillas, manifest y reconstrucción](#plantillas-manifest-y-reconstrucción)
- [Suites: comandos y opciones](#suites-comandos-y-opciones)
- [Resultados y logs](#archivos-de-resultados-y-logs)

## Todos los comandos

`run.sh` selecciona el runtime configurado y despacha los siguientes comandos. Cambiar de carpeta o renombrar el checkout no exige editar los scripts; las rutas de los perfiles siguen las reglas indicadas más abajo.

| Comando | Para qué sirve | Cuándo usarlo / efecto |
| --- | --- | --- |
| `bash run.sh ui` | Abre el servicio web de imagen, cámara y pose. | Uso habitual. Precarga por defecto SAM/DINOv2 y FoundationPose; deja la terminal abierta. |
| `bash run.sh suite run` | Ejecuta el conjunto de pruebas de `config/suite.json`. | Crea un run nuevo y su informe; PEM está activado por defecto. |
| `bash run.sh suite report --output results/suites/ID` | Vuelve a exportar un run guardado. | No repite inferencia. `--zip` crea el archivo para compartir. |
| `bash run.sh suite pose --output results/suites/ID` | Añade PEM a detecciones ya guardadas. | Requiere CUDA y el dataset; no repite ISM. |
| `bash run.sh detect --example control_board --part 05_control_board --mode rgbd --pose` | Analiza una captura sin interfaz. | Crea un resultado puntual en `results/runs/`; admite también archivos propios. |
| `bash run.sh scores --run results/suites/ID` | Compara todos los CAD sobre una misma región. | Diagnóstico adicional; no son diez piezas detectadas. Conserva el resultado anterior. |
| `bash run.sh sam doctor` | Comprueba CUDA, dependencias, pesos y plantillas del catálogo. | Primera comprobación si falta algo. No abre cámara ni ejecuta inferencia. |
| `bash run.sh sam render` | Renderiza las 42 vistas de cada CAD del manifest. | Reconstrucción explícita de plantillas; usa Blender y puede tardar. `--part CLAVE` selecciona una pieza. |
| `bash run.sh sam segment --frame control_board` | Ejecuta ISM mediante el adaptador del modelo. | Uso avanzado; la salida predeterminada está bajo `data/prepared/sam6d/results/`. |
| `bash run.sh sam pose --frame control_board` | Ejecuta PEM sobre resultados del adaptador. | Uso avanzado; necesita sus detecciones previas. Para ensayos nuevos se recomienda `suite run`. |
| `bash run.sh prepare --config data/dataset.json --output data/prepared-new` | Convierte CAD y RGB-D fuente y genera un manifest. | Solo al preparar otro dataset o reconstruirlo. No descarga originales ni genera plantillas. |
| `bash run.sh inspect` | Comprueba profundidad y marcos CAD del dataset. | Solo imprime; `--output ARCHIVO` guarda el informe y `--preview` pide una imagen de resumen. |
| `bash run.sh audit --output-dir results/template-audit` | Audita las plantillas renderizadas. | Uso avanzado; puede necesitar `rtree`. La salida solo se guarda si se pide. |
| `bash run.sh settings` | Muestra las rutas efectivas de la configuración global. | Localizar Python, datos, pesos y Blender. |
| `bash run.sh install` | Instala dependencias y pesos SAM6D/PEM y compila PointNet2 si falta. | Instalación o recuperación; requiere un runtime configurado. Modifica ese entorno y puede descargar datos. |
| `bash run.sh install-foundationpose` | Prepara el venv y extensiones de FoundationPose. | Instalación; los pesos FoundationPose se obtienen por separado. |
| `bash run.sh test` | Ejecuta las pruebas automáticas. | Desarrollo/verificación; no utiliza una cámara real. |

Para consultar opciones sin ejecutar modelos:

```bash
bash run.sh --help
bash run.sh ui --help
bash run.sh suite run --help
bash run.sh detect --help
bash run.sh sam --help
bash run.sh prepare --help
```

Los scripts de instalación son acciones directas; utiliza su guía antes de ejecutarlos. La forma de revisar una suite sin lanzarla es `suite run --dry-run`.

## Funcionamiento

1. **SAM o FastSAM** propone máscaras del fotograma.
2. **DINOv2/ISM** compara cada propuesta con las vistas del CAD y calcula sus scores de semántica, apariencia y, con RGB-D, geometría.
3. **PEM o FoundationPose** estima la pose del CAD usando la detección aceptada y la profundidad. FoundationPose permite también una caja manual para prescindir de SAM en la inicialización.
4. **FoundationPose en seguimiento** actualiza la pose sobre el RGB-D más reciente. SAM interviene al perder el seguimiento y periódicamente como respaldo.

En RGB sin profundidad el score usa semántica y apariencia; no hay pose métrica ni score geométrico completo. Para pose usa RGB-D alineado, calibración y unidades CAD correctas.

En RGB-D, el score ISM se calcula como:

```text
ISM = (semántica + apariencia + geometría × visibilidad) / (2 + visibilidad)
```

Es un score de detección, no una probabilidad calibrada ni una medida de precisión de pose.

La interfaz mantiene SAM/DINOv2 y FoundationPose en sus procesos persistentes cuando se utilizan, y el arranque predeterminado los precarga. Las suites reutilizan ISM durante cada ejecución; PEM usa un worker separado para sus estimaciones, por lo que sus cargas y tiempos aparecen aparte. No debe interpretarse una suite completa como una prueba de tracking en tiempo real.

## Directorios y archivos

| Ruta | Contenido y uso |
| --- | --- |
| `run.sh` | Entrada única para interfaz, suites, preparación, instalación y diagnósticos. |
| `config/` | Tres perfiles editables: rutas/runtime, interfaz y suites. |
| `data/dataset.json` | Fuentes del ensamblaje de ejemplo y nombres visibles corregidos. |
| `data/prepared/manifest.json` | Índice generado de piezas, capturas, rutas, unidades y transformaciones. Lo leen la interfaz y las suites. |
| `data/prepared/sam6d/models/` | CAD centrados: PLY para PEM y OBJ para renderizado. Ambos formatos se usan. |
| `data/prepared/sam6d/frames/` | RGB, profundidad uint16 sin pérdida y calibración de los ejemplos. |
| `data/prepared/sam6d/templates/` | 42 vistas por CAD: RGB, máscaras, poses y coordenadas XYZ métricas para ISM/PEM. |
| `checkpoints/` | Pesos persistentes de SAM, FastSAM, DINOv2 y PEM. |
| `models/SAM-6D/` | Fuente SAM-6D y adaptaciones de integración. |
| `models/FoundationPose/` | Fuente FoundationPose y sus pesos en `weights/`. |
| `pipeline/sam6d_ui/` | Servidor, web, validación de entradas, motores, cámara y tracking. |
| `pipeline/suites/` | Ejecución de ensayos, poses e informes. |
| `pipeline/tools/` | Preparación/inspección de datos, descargas, detección y comparación de CAD. |
| `pipeline/workers/` | Worker de estimación PEM. |
| `pipeline/native/` | Extensiones compiladas PEM/FoundationPose; dependen del sistema y de Python/PyTorch/CUDA. |
| `pipeline/setup/` | Scripts de instalación, dependencias y constraints de los dos runtimes; no son guías adicionales. |
| `pipeline/settings.py` | Carga de configuración y resolución de rutas. |
| `pipeline/launcher.py` | Selección de runtime y despacho de comandos. |
| `pipeline/tests/` | Pruebas automáticas. |
| `installation/` | Solo la guía de instalación. |
| `captures/` | Imágenes y grabaciones guardadas desde la RealSense. |
| `results/interface/` | Trabajos, cachés CAD, logs y sesiones de tracking de la interfaz. |
| `results/suites/` | Runs originales de suites, logs, detecciones, poses y manifest de su ejecución. |
| `results/reports/` | Informes exportados de suites. |
| `results/dashboard` | Enlace al último informe exportado. |
| `.cache/` | Embeddings/referencias reutilizables y temporales de compilación locales. |
| `docs/REFERENCE.md` | Esta guía detallada de comandos, archivos y parámetros. |
| `docs/cleanup.md` | Registro histórico de la depuración de archivos; no es otra guía de uso. |

Las plantillas del ejemplo ocupan unos 335 MiB: son datos precomputados reutilizados, no ejecuciones antiguas. **`data/` no se regenera automáticamente al arrancar**. Borrarla elimina también el manifest y la descripción del dataset. Su reconstrucción y renderizado se explican en la guía de instalación.

ASAP mantiene su código, configuración y datos de planificación en su proyecto hermano. CAD Tracking prepara solo entradas de visión.

## Plantillas, manifest y reconstrucción

Las plantillas se crean renderizando el CAD con Blender en **42 orientaciones**. Cada vista produce un RGB, una máscara y coordenadas XYZ en milímetros. ISM compara características visuales con esas vistas; PEM utiliza también la geometría métrica. Los metadatos registran las unidades y el hash del CAD para validar la referencia.

| Elemento | Quién lo crea | Qué ocurre si se borra |
| --- | --- | --- |
| `data/dataset.json` | Configuración de las fuentes, escrita por el proyecto/usuario. | No se regenera; hay que recuperarlo o volver a describir el dataset. |
| `data/prepared/manifest.json` | `prepare`. | La interfaz y la suite no lo reconstruyen al arrancar. |
| `data/prepared/sam6d/models/` y `frames/` | `prepare`, a partir de originales. | Se pueden reconstruir si siguen disponibles las fuentes y la configuración. |
| `data/prepared/sam6d/templates/` | `sam render`. | La suite no las renderiza automáticamente. La interfaz reconoce esos CAD y espera encontrar sus plantillas. |
| `results/interface/cache/cad/` | La interfaz al analizar un CAD nuevo. | Recalcula las referencias que faltan al volver a analizar ese CAD; no es la carpeta del catálogo anterior. |
| Embeddings de referencia | ISM/DINOv2 sobre las plantillas. | Se calculan y reutilizan mediante la caché; no sustituyen a las imágenes/XYZ del renderizado. |

Si solo faltan plantillas y el manifest/modelos siguen disponibles:

```bash
bash run.sh sam render
```

Para reconstruir un dataset en otra carpeta, conservando el existente:

```bash
bash run.sh prepare --config data/dataset.json --output data/prepared-new
bash run.sh sam render --manifest data/prepared-new/manifest.json
```

Después cambia `paths.prepared` en `config/settings.json` y `manifest` en `config/suite.json` para seleccionar el nuevo dataset. Ambos valores se resuelven respecto a sus JSON. `prepare` necesita CAD, capturas y catálogo originales; no puede recuperar fuentes borradas.

La inicialización manual de FoundationPose no necesita renderizar el CAD para SAM, pero el servidor actual sigue leyendo el manifest del ejemplo al arrancar. Borrar el manifest o toda `data/` afecta también a ese arranque.

## Configuración completa

| Archivo | Qué cambiar aquí |
| --- | --- |
| `config/settings.json` | Python de cada runtime, repositorios, Blender, datos, resultados, pesos y cachés. |
| `config/interface.json` | Puerto, precarga, controles iniciales de inferencia, cámara y seguimiento. |
| `config/suite.json` | Dataset del ensayo, filtros, parámetros ISM/PEM y destinos. |
| `data/dataset.json` | CAD/capturas del ejemplo; se utiliza al preparar otro conjunto de datos. |

### Prioridad y rutas

- Las rutas de un JSON son relativas **al archivo que las contiene**. Las de terminal son relativas al directorio actual. También se aceptan rutas absolutas y `~`.
- `--settings ARCHIVO` selecciona la configuración global y prevalece sobre `CAD_TRACKING_SETTINGS`.
- `ui --config ARCHIVO` o `suite run --config ARCHIVO` selecciona un perfil. También existen `CAD_TRACKING_INTERFACE_CONFIG` y `CAD_TRACKING_SUITE_CONFIG`.
- Los perfiles personalizados de interfaz/suite heredan los campos no indicados; las claves desconocidas se rechazan. El bloque `tracking.quality` también admite cambios parciales.
- Los argumentos de terminal, como `--port`, `--frame` o `--confidence`, prevalecen sobre el perfil.
- `settings.json` requiere el bloque completo de rutas/runtime. Sus fragmentos de ejemplo muestran qué editar, no un sustituto de todo el archivo.
- Los controles de la web pueden cambiar parámetros por petición. Configurar el segmentador a precargar y el segmentador inicial de inferencia son dos campos distintos.

Ejemplo de perfil `config/interface.local.json` para otro puerto y FastSAM:

```json
{
  "server": {"port": 8771, "preload": "fastsam"},
  "inference": {"segmentor": "fastsam", "mode": "rgbd", "pose_backend": "foundationpose", "pose": true},
  "tracking": {"redetect_interval_s": 10}
}
```

```bash
bash run.sh ui --config config/interface.local.json
bash run.sh ui --config config/interface.local.json --port 8772
bash run.sh --settings config/settings.json ui
```

### Referencia de parámetros

Los valores de las tablas reflejan los JSON distribuidos al redactar esta guía. El valor efectivo siempre es el del perfil y los argumentos de la ejecución.

#### settings.json

| Campo | Valor actual | Significado |
| --- | --- | --- |
| `paths.sam6d` | `"../models/SAM-6D"` | Repositorio SAM-6D, que contiene el subdirectorio original SAM-6D/. |
| `paths.foundationpose` | `"../models/FoundationPose"` | Repositorio FoundationPose; conserva weights/. |
| `paths.prepared` | `"../data/prepared"` | Manifest, CAD preparados, capturas de ejemplo y plantillas. |
| `paths.results` | `"../results"` | Resultados generales; las suites/interfaz pueden elegir otra salida en su perfil. |
| `paths.captures` | `"../captures"` | Capturas y grabaciones; el perfil de interfaz determina su carpeta efectiva. |
| `paths.checkpoints` | `"../checkpoints"` | Pesos persistentes de SAM, DINOv2 y PEM. |
| `paths.native` | `"../pipeline/native"` | Extensiones compiladas para PEM y FoundationPose. |
| `paths.runtime_python` | `"../../.venvs/assembly-runtime/bin/python"` | Python del runtime compartido SAM6D/PEM/ASAP. |
| `paths.foundationpose_python` | `"../../.venvs/foundationpose/bin/python"` | Python del venv FoundationPose, que comparte el runtime base. |
| `paths.blender` | `"../../.runtime/blender/blender-3.3.1-linux-x64/blender"` | Ejecutable Blender para renderizar plantillas CAD. |
| `paths.file_root` | `"../.."` | Raíz por defecto para explorar archivos; la interfaz puede limitarla en su perfil. |
| `paths.reference_cache` | `"../.cache/reference_features"` | Carpeta de embeddings de las herramientas de comparación. |
| `paths.ui_reference_cache` | `"../.cache/ui_reference_features"` | Carpeta de embeddings reutilizados por la interfaz. |
| `paths.runtime_cache` | `"../../.runtime/cache"` | Cachés del runtime y compilaciones Torch. |
| `paths.matplotlib_cache` | `"../../.runtime/matplotlib"` | Caché/configuración Matplotlib. |
| `paths.dataset` | `"../data/dataset.json"` | Fuentes y nombres visibles del dataset de ejemplo. |
| `runtime.threads` | `16` | Hilos CPU; no es una frecuencia de imagen. |
| `runtime.render_size` | `256` | Opción que el lanzador pasa al adaptador; el helper Blender actual fija 256 px. Ver limitación debajo. |
| `runtime.render_samples` | `16` | Opción que el lanzador pasa al adaptador; el helper Blender actual fija 16 muestras. Ver limitación debajo. |
| `runtime.build_jobs` | `2` | Paralelismo al compilar extensiones nativas. |

El helper `blender_reference.py` utilizado actualmente fija 256×256 píxeles y 16 muestras. Cambiar solo `runtime.render_size` o `runtime.render_samples` no modifica ese renderizado: el adaptador aún no transmite esos valores al helper. Las vistas se generan en 42 orientaciones.

#### interface.json

| Campo | Valor actual | Significado |
| --- | --- | --- |
| `server.port` | `8770` | Puerto HTTP local (solo escucha en 127.0.0.1). |
| `server.preload` | `"sam"` | sam, fastsam o none; modelos que se precargan al arrancar. |
| `server.backbone` | `"vit_h"` | vit_h, vit_l o vit_b; variante de SAM. FastSAM ignora este valor. |
| `server.data_dir` | `"../results/interface"` | Trabajos, logs y seguimiento de la interfaz. |
| `server.capture_dir` | `"../captures"` | Imágenes y grabaciones Intel RealSense. |
| `server.file_root` | `"../.."` | Raíz por defecto para explorar archivos; la interfaz puede limitarla en su perfil. |
| `inference.mode` | `"rgb"` | rgb sin profundidad o rgbd alineado/calibrado; la pose requiere rgbd. |
| `inference.device` | `"cuda"` | auto, cpu o cuda; la pose requiere CUDA. |
| `inference.backbone` | `"vit_h"` | vit_h, vit_l o vit_b; variante de SAM. FastSAM ignora este valor. |
| `inference.cad_units` | `"mm"` | mm, cm o m; unidad física del CAD de entrada. |
| `inference.segmentor` | `"sam"` | sam o fastsam. |
| `inference.pose` | `false` | Activa estimación de pose con el backend elegido; requiere RGB-D y CUDA. |
| `inference.pose_backend` | `"pem"` | pem o foundationpose; FoundationPose exige RGB-D/CUDA y activa pose. |
| `inference.fastsam_conf` | `0.05` | Umbral interno del generador FastSAM (0–1), distinto del score ISM. |
| `inference.fastsam_iou` | `0.9` | Umbral IoU de FastSAM (0–1). |
| `inference.min_score` | `0.3` | Score mínimo ISM conservado (0–1), no mayor que confidence. |
| `inference.confidence` | `0.5` | Umbral de aceptación final ISM (0–1); no una probabilidad calibrada. |
| `inference.stability` | `0.97` | Estabilidad mínima de máscaras SAM (0–1). Se aplica sobre el valor del YAML oficial. |
| `inference.pred_iou` | `0.88` | Umbral predicho de calidad de máscara SAM (0–1). |
| `inference.points_per_batch` | `64` | Prompts SAM por lote (1–128); afecta memoria y rendimiento. |
| `inference.nms` | `0.25` | Umbral IoU de supresión por objeto (0–1). |
| `inference.aggregation` | `"avg_5"` | avg_5, mean, median o max para similitud semántica con vistas CAD. |
| `camera.width` | `640` | Anchura RGB-D: 640 o 1280, con altura compatible. |
| `camera.height` | `480` | Altura RGB-D: 480 o 720 (640×480 / 1280×720). |
| `camera.fps` | `15` | 15 o 30 FPS de adquisición; no garantiza esa tasa de inferencia. |
| `camera.preview_width` | `640` | Anchura máxima de la vista previa JPEG. |
| `camera.preview_height` | `480` | Altura máxima de la vista previa JPEG. |
| `camera.jpeg_quality` | `70` | Calidad JPEG de la vista previa (1–100). No altera el RGB-D guardado. |
| `tracking.redetect_interval_s` | `10.0` | Respaldo SAM periódico; 0 desactiva solo el respaldo. La recuperación automática sigue activa. |
| `tracking.recovery_retry_s` | `1.0` | Pausa entre reintentos de recuperación fallidos; no retrasa el primer intento. |
| `tracking.max_detection_age_s` | `5.0` | Edad máxima del fotograma al recibir una detección SAM; se descartan las anteriores a una pérdida. |
| `tracking.quality_bad_frames` | `3` | Fotogramas geométricamente incoherentes consecutivos para confirmar pérdida. |
| `tracking.quality.enabled` | `true` | Activa controles de profundidad y movimiento; no es una confianza calibrada. |
| `tracking.quality.sample_points` | `1024` | Muestras de superficie CAD calculadas una vez por pieza. |
| `tracking.quality.depth_tolerance_mm` | `20.0` | Diferencia máxima de profundidad para considerar una muestra coherente. |
| `tracking.quality.min_depth_samples` | `20` | Muestras visibles con profundidad para evaluar la pieza. |
| `tracking.quality.min_depth_support` | `0.35` | Fracción mínima de muestras coherentes tras excluir oclusiones parciales. |
| `tracking.quality.min_valid_depth_fraction` | `0.25` | Fracción mínima de profundidad válida sobre la superficie proyectada. |
| `tracking.quality.min_visible_fraction` | `0.1` | Fracción mínima de la superficie frontal proyectada dentro de imagen. |
| `tracking.quality.max_occlusion_fraction` | `0.7` | Oclusión máxima; se toleran oclusiones parciales. |
| `tracking.quality.translation_slack_mm` | `20.0` | Margen de posición antes de considerar el límite de velocidad. |
| `tracking.quality.max_translation_speed_mm_s` | `1000.0` | Velocidad máxima del centro del CAD entre poses aceptadas. |
| `tracking.quality.rotation_slack_deg` | `20.0` | Margen angular para ruido de orientación. |
| `tracking.quality.max_rotation_speed_deg_s` | `360.0` | Velocidad angular máxima entre poses aceptadas. |
| `tracking.quality.max_motion_gap_s` | `0.5` | Con huecos mayores se omite el test de salto; la profundidad se sigue evaluando. |
| `examples.frame` | `"control_board"` | Captura de ejemplo seleccionada inicialmente. |
| `examples.cad` | `"05_control_board"` | Clave CAD seleccionada inicialmente en la interfaz. |

#### suite.json

| Campo | Valor actual | Significado |
| --- | --- | --- |
| `manifest` | `"../data/prepared/manifest.json"` | Manifest de datos preparados (schema_version 1). |
| `output_root` | `"../results/suites"` | Cada suite crea aquí una carpeta UTC por ejecución. |
| `export_root` | `"../results/reports"` | Informes HTML/CSV/ZIP; cada ejecución conserva una carpeta propia. |
| `device` | `"auto"` | auto, cpu o cuda; la pose requiere CUDA. |
| `min_score` | `0.4` | Score mínimo ISM conservado (0–1), no mayor que confidence. |
| `confidence` | `0.5` | Umbral de aceptación final ISM (0–1); no una probabilidad calibrada. |
| `stability_score` | `0.97` | Estabilidad SAM efectiva de la suite. Por defecto 0.97; sobrescribe el 0.85 del YAML. |
| `segmentor` | `"sam"` | sam o fastsam. |
| `backbone` | `"vit_h"` | vit_h, vit_l o vit_b; variante de SAM. FastSAM ignora este valor. |
| `threads` | `16` | Hilos CPU; no es una frecuencia de imagen. |
| `reference_cache` | `"../.cache/reference_features"` | Carpeta de embeddings de las herramientas de comparación. |
| `pose` | `true` | Calcula PEM sobre detecciones aceptadas. --no-pose lo desactiva. |
| `frame` | `[]` | Lista de nombres de captura; vacía significa todas. --frame permite seleccionar desde terminal. |

#### data/dataset.json

| Campo | Significado |
| --- | --- |
| `assembly_id` | Identificador del ensamblaje de ejemplo. |
| `cad_dir` | Carpeta de los CAD fuente. |
| `photos_dir` | Capturas RGB, profundidad y calibración fuente. |
| `catalog` | Catálogo YAML que relaciona piezas, claves CAD y opciones. |
| `photo_aliases` | Alias para seleccionar ejemplos; no son etiquetas usadas para clasificar la imagen. |
| `display_names` | Correcciones de nombres visibles, conservando IDs y geometría del CAD. |


## Interfaz: imágenes y CAD

1. Arranca `bash run.sh ui` y espera a que termine la precarga.
2. Selecciona archivos del servidor o súbelos desde el navegador.
3. Elige el CAD y declara sus unidades: mm, cm o m.
4. Para detección sin pose utiliza RGB. Para pose utiliza RGB-D y aporta profundidad y calibración alineadas con ese RGB.
5. Selecciona el segmentador y, si corresponde, PEM o FoundationPose como motor de pose.
6. Pulsa **Detectar objeto**. El resultado incluye detecciones, score, tiempos, dispositivo real y, cuando se calcula, pose y proyección del CAD.

La lista de ejemplos usa el manifest preparado. El nombre del archivo de una foto no se utiliza como etiqueta real para clasificar la pieza.

### Formatos

| Entrada | Requisito |
| --- | --- |
| RGB | Imagen de color admitida por la interfaz; máximo 4 megapíxeles. |
| CAD | STL, OBJ o PLY; unidades declaradas explícitamente. |
| Profundidad | PNG de 16 bits o NPY `uint16`, sin colorear y con el mismo tamaño que RGB. |
| Calibración | JSON con intrínsecos y escala de profundidad; se admiten el formato BOP/SAM6D y metadatos RealSense compatibles. |

En BOP/SAM6D, `depth_scale` expresa **milímetros por unidad raw**; no debe confundirse con metros por unidad. La preparación convierte los metadatos RealSense cuando corresponde.

### FoundationPose con caja manual

Selecciona FoundationPose y la inicialización **Caja manual · solo FoundationPose**. Con una captura congelada, marca la región de la pieza y estima su pose. Esa región rectangular sirve como máscara aproximada inicial; no es una segmentación precisa del contorno.

En este modo no se ejecuta SAM/DINOv2 para localizar la pieza. El tracking usa los controles de profundidad y pose, pero si pierde el objeto hay que volver a marcar/localizarlo. Para evitar la precarga SAM en una sesión dedicada a este modo, puedes arrancar con `--preload none`; la primera llamada cargará FoundationPose y las siguientes reutilizarán el worker.

## Cámara y seguimiento

Selecciona **Intel RealSense en directo**, elige el dispositivo y conecta la cámara. Las resoluciones admitidas por el control son 640×480 y 1280×720, a 15 o 30 FPS. El SDK/dispositivo puede limitar las combinaciones disponibles.

- **Analizar fotograma actual** toma el RGB-D más reciente en el momento de enviar la petición.
- **Detección en directo** repite análisis completos. Su frecuencia depende de la inferencia; no equivale al tracking rápido.
- **Seguimiento rápido** inicializa FoundationPose y después actualiza la pose continuamente sobre los fotogramas más recientes.
- **Capturar/exportar imagen** guarda una captura de la cámara o permite descargar la original/overlay del resultado.
- **Grabar vídeo** conserva una grabación RGB-D con calibración, marcas de tiempo y vídeo de vista previa. También se pueden descargar resultados del seguimiento de una grabación.
- **Abrir visor compacto** muestra una ventana pequeña de cámara/pose. Cambiar a ese visor no elimina el coste de SAM ni del estimador de pose.

La adquisición conserva el último fotograma y no acumula una cola para la inferencia. Los FPS de cámara y de tracking son medidas distintas. Si un análisis tarda más que la llegada de frames, se descartan frames intermedios para trabajar con el más reciente.

### Seguimiento y recuperación

Con inicialización automática, FoundationPose sigue continuamente y SAM se ejecuta:

1. Al confirmarse una pérdida, sin esperar al temporizador de respaldo.
2. Cada **10 segundos** por defecto para comprobar que sigue sobre la pieza correcta.

Los errores de tracking o las poses numéricamente inválidas activan recuperación. Los controles de profundidad/saltos requieren tres fotogramas incoherentes seguidos por defecto; una pose sospechosa no se publica mientras se confirma. Se comprueba profundidad en la superficie frontal proyectada y movimiento del centro del CAD, tolerando oclusiones parciales.

Solo puede haber una detección SAM pendiente. Una detección anterior a la pérdida o demasiado antigua se descarta; la recuperación usa una captura nueva. Mientras se recupera, el visor muestra el directo de cámara. Si falla un intento, se reintenta con la pausa configurada; no se recargan los modelos en cada comprobación.

El control de intervalo de la web se puede cambiar durante el seguimiento. **`0` desactiva el respaldo periódico, manteniendo la recuperación SAM al perder el seguimiento en modo automático.** El modo manual conserva SAM desactivado.

Los controles son heurísticos, no una certificación de precisión: un desvío puede seguir siendo geométricamente plausible. El respaldo periódico ayuda a detectar esos casos. Ajusta los umbrales para el tamaño del objeto, ruido de profundidad, simetrías y velocidad de movimiento.

## Suites: comandos y opciones

Una suite recorre las capturas seleccionadas del manifest y evalúa los CAD del catálogo. La suite actual usa PEM para la pose; elegir FoundationPose en la interfaz no cambia el backend de las suites.

```bash
# Revisar el comando efectivo sin ejecutar modelos
bash run.sh suite run --config config/suite.json --dry-run

# Ejecutar todas las capturas con el perfil y PEM activado
bash run.sh suite run --config config/suite.json

# Una captura; repetir --frame para seleccionar varias
bash run.sh suite run --frame hmi_panel --frame control_board

# Comparar FastSAM manteniendo los demás parámetros del perfil
bash run.sh suite run --segmentor fastsam

# Solo detección, sin PEM
bash run.sh suite run --no-pose

# Detección en CPU, sin pose
bash run.sh suite run --device cpu --no-pose
```

La salida normal se crea con una marca de tiempo dentro de `results/suites/`; `--output` permite elegir otra carpeta **nueva**. Se rechaza una salida que ya contiene archivos para conservar el resultado anterior.

Para un perfil propio, puedes crear `config/suite.local.json`:

```json
{
  "segmentor": "fastsam",
  "min_score": 0.4,
  "confidence": 0.6,
  "pose": true,
  "frame": ["hmi_panel", "control_board"]
}
```

```bash
bash run.sh suite run --config config/suite.local.json
```

### Argumentos de suite

| Argumento | Efecto |
| --- | --- |
| `run` | Ejecuta una suite nueva. |
| `report` | Exporta de nuevo un run guardado, sin repetir ISM/PEM. Requiere `--output`. |
| `pose` | Calcula PEM para detecciones de un run guardado; no repite ISM. Requiere `--output`. |
| `--config` | Perfil JSON de suite. |
| `--manifest` | Manifest del dataset; prevalece sobre el perfil. |
| `--output` | Carpeta nueva para `run`, o carpeta del run existente para `report`/`pose`. |
| `--export-dir` | Carpeta de informe separada del run. No puede contener archivos ajenos al exportador. |
| `--frame` | Nombre de captura; se puede repetir. Sin filtros utiliza todas. |
| `--device` | `auto`, `cuda` o `cpu`; PEM necesita CUDA. |
| `--segmentor` / `--backbone` | SAM/FastSAM y variante SAM; FastSAM ignora backbone. |
| `--min-score` / `--confidence` | Conservación y aceptación ISM; min-score no puede superar confidence. |
| `--stability-score` | Estabilidad SAM efectiva; sobrescribe el YAML oficial, que puede tener otro valor. |
| `--threads` | Hilos CPU; no fija FPS de inferencia. |
| `--reference-cache` | Caché de embeddings de referencia del CAD. |
| `--pose` / `--no-pose` | Activa o desactiva PEM sobre el perfil. |
| `--overwrite-pose` | Recalcula poses existentes al enriquecer un run; conserva `pose.before.json`. |
| `--dry-run` | Imprime el comando de `run` sin ejecutarlo. |
| `--no-export` | Conserva el run sin generar su informe al terminar. |
| `--zip` | Crea también el ZIP del informe; no se crea automáticamente. |

### Informe y poses de un run existente

```bash
bash run.sh suite report --output results/suites/ID
bash run.sh suite report --output results/suites/ID --zip
bash run.sh suite pose --output results/suites/ID
bash run.sh suite pose --output results/suites/ID --overwrite-pose
```

Cada run conserva `run.json`, sus casos y `dataset_manifest.json` con los bytes del manifest usado. Los hashes evitan mezclar poses e ISM de ejecuciones diferentes. Los recursos compartidos del dataset, como sus plantillas, siguen en `data/prepared/`.

## Interpretación de resultados

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory results/dashboard
```

Abre `http://127.0.0.1:8766/#poses`. En remoto reenvía también el puerto 8766 por SSH/VS Code. Cada exportación completa actualiza el enlace `results/dashboard`.

El informe contiene scores ISM y su desglose, tiempos, imágenes, máscaras, comparación de CAD y poses disponibles. **Matrices y triedros** abre la galería de poses. Las capturas sin detección aceptada no tienen pose; los errores PEM se muestran aparte.

| Archivo de informe | Contenido |
| --- | --- |
| `index.html` | Informe navegable, utilizable sin conexión junto con sus carpetas de recursos. |
| `confianza_sam6d_por_imagen.csv` | Score ISM y desglose por imagen. |
| `poses_pem.csv` | Poses calculadas, traslación y matrices. |
| `resultados_poses.json` | Poses, transformaciones y metadatos. |
| `resultados_completos.json` | Resultado ISM original exportado. |
| `cases/` | RGB, profundidad, máscaras, overlays, auditorías y resultados por caso. |

**R** es la rotación 3×3, **t** la traslación y **T** la transformación homogénea 4×4. Compara `T_camera_from_input_cad_mm` entre backends: transforma el CAD original a la cámara OpenCV (X derecha, Y abajo, Z hacia delante), con traslación en milímetros.

El origen de un CAD puede quedar fuera de su geometría y de la máscara. En la visualización actual PEM representa el triedro del CAD original, mientras FoundationPose representa el del modelo centrado; eso puede desplazar los ejes aunque las matrices exportadas sean comparables. Si se desplaza el propio contorno/nube CAD respecto al objeto, revisa la estimación, unidades, intrínsecos y transformaciones.

El score combinado PEM sirve para ordenar poses y puede incorporar el score ISM; no debe interpretarse como otra confianza de detección ni como exactitud métrica. `estimated` significa que se produjo una estimación, no que se haya validado contra una pose real de referencia.

## Archivos de resultados y logs

| Ruta por defecto | Qué se guarda |
| --- | --- |
| `captures/images/ID/` | Captura RGB-D, calibración y metadatos. |
| `captures/videos/ID/` | Grabación, índices de frames, calibración y vídeo. |
| `results/interface/jobs/sam6d/ID/` | Análisis ISM/PEM de la web. |
| `results/interface/jobs/foundationpose/ID/` | Análisis FoundationPose de la web. |
| `results/interface/cache/cad/` | Referencias renderizadas para CAD propios. |
| `results/interface/logs/` | Logs del worker FoundationPose y del servicio cuando corresponde. |
| `results/interface/live/ID/` | Sesión de seguimiento, comprobaciones/recuperaciones SAM y resumen final. |
| `results/suites/ID/execution.log` | Log de una suite guardada. |
| `results/suites/ID/cases/IMAGEN/pose/CLAVE_CAD/pose.log` | Log PEM de una pieza/captura. |

La terminal muestra el log del servidor HTTP. Un trabajo fallido conserva su `error.log` y la interfaz enlaza ese detalle. Los destinos pueden cambiar mediante los perfiles y argumentos documentados.

## Herramientas y resolución de problemas

```bash
bash run.sh sam doctor
bash run.sh inspect
bash run.sh inspect --output results/input-audit.json
bash run.sh audit --output-dir results/template-audit
bash run.sh test
```

`inspect` comprueba profundidad y marcos CAD; los RGB históricos pueden mostrar diferencias entre decodificadores JPEG si no tienen hashes de bytes. Las nuevas preparaciones registran hashes de fuente y PNG. `audit` revisa XYZ/escala de plantillas y puede necesitar dependencias opcionales de proximidad de mallas. Solo se escriben esos informes auxiliares si se solicita una salida.

Para un análisis puntual desde terminal:

```bash
bash run.sh detect --example control_board --part 05_control_board --mode rgbd --pose
bash run.sh detect --rgb FOTO.png --cad PIEZA.stl --depth PROFUNDIDAD.png --camera CAMARA.json --mode rgbd --pose --cad-units mm
```

`detect` ejecuta ISM y, con `--pose`, PEM; los resultados van a `results/runs/` salvo `--output`. Consulta `bash run.sh detect --help` para cambiar umbrales, dispositivo y segmentador.

`scores --run results/suites/ID` permite el diagnóstico adicional de todos los CAD sobre la misma región. Es una comparación de hipótesis sobre ese recorte, no diez objetos detectados ni probabilidades. Conserva los resultados previos y requiere el manifest exacto del run.

| Problema | Comprobación o solución |
| --- | --- |
| `Address already in use` | Hay un servicio en ese puerto. Utilízalo, detenlo desde su terminal o cambia `--port`. |
| RealSense no aparece/está ocupada | Revisar conexión al servidor, permisos SDK/USB y otro proceso que la esté usando. |
| Faltan checkpoints | Revisar `checkpoints/`, `models/FoundationPose/weights/` y la guía de instalación. |
| No aparecen matrices/triedros | Revisar que el run calculó PEM, abrir el dashboard actual y entrar en `#poses`. Enriquecer con `suite pose` si falta pose. |
| Mask correcta pero CAD mal proyectado | Revisar unidades, profundidad alineada, intrínsecos y T del CAD; segmentar bien no garantiza una pose correcta. |
| Streaming fluido, detección lenta | Cámara e inferencia tienen tasas diferentes. Usar tracking FoundationPose, revisar tiempos y escoger SAM/FastSAM según el caso. |
| Recuperaciones repetidas | Revisar `pose_quality`, profundidad sobre la pieza y límites de movimiento; ajustar umbrales del perfil. |
| Datos/plantillas ausentes tras borrar `data/` | Recuperar la descripción del dataset y seguir preparación/renderizado de la guía de instalación. |
| Extensión `.so` no carga en otra máquina | Comprobar Python/PyTorch/CUDA y recompilar en ese runtime. |

## Procedencia del código

Fuentes oficiales: [SAM-6D de JiehongLin](https://github.com/JiehongLin/SAM-6D) y [FoundationPose de NVlabs](https://github.com/NVlabs/FoundationPose).

La web, RealSense, perfiles, lanzador, suites, informes, control de recuperación y scripts de integración de `pipeline/` son desarrollo del proyecto. En los repositorios de los modelos también hay adaptaciones locales, como `pddl_pipeline.py`, `run_*pddl.py` y la visualización del triedro PEM. No son una copia oficial intacta: conserva esas adaptaciones al trasladar la instalación.

Los repositorios originales mantienen sus licencias, historia Git y documentación propia. Los archivos de `pipeline/setup/` son entradas del instalador; la entrada al proyecto es el [README principal](../README.md), esta referencia concentra los detalles y [la guía de instalación](../installation/README.md) cubre entornos y dependencias.

## Selección de ensamblaje

`config/suite.json`: `assembly` selecciona el ensamblaje (`printed_assembly` por
defecto). `--assembly` lo sobrescribe. Si se selecciona otro ensamblaje y no se
especifica un manifest distinto, se usa
`data/assemblies/<assembly>/prepared/manifest.json`. El manifest debe corresponder
al ensamblaje seleccionado. Las suites nuevas de la segunda versión se guardan
en `results/suites/printed_assembly_2/`; las salidas existentes no se mueven.

`data/dataset.json`: `assembly` y `assembly_registry` seleccionan la geometría e
inventario canónicos al preparar. `prepare --assembly` sobrescribe la selección
sin modificar el JSON. Cada versión utiliza sus propias capturas y modelos.
