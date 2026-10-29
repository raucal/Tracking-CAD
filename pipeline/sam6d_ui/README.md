# Interfaz de visión: SAM-6D, FoundationPose e Intel RealSense

Una única interfaz web permite detectar un CAD en una imagen, estimar su pose 6D,
ver la cámara Intel RealSense conectada al servidor y guardar vídeo RGB-D.

## Origen de esta interfaz y documentación relacionada

Este documento describe **nuestra interfaz e integración**, desarrollada en
`SAM6D/pipeline/sam6d_ui/`: servidor, selección de modelo, explorador de archivos,
cola de trabajos, captura RealSense, exportación y organización de resultados.
Los estimadores y generadores de máscaras proceden de proyectos externos.
SAM-6D aporta ISM/PEM; FoundationPose aporta el registro y el seguimiento;
SAM/FastSAM generan máscaras y DINOv2 aporta descriptores. La unión de estos
componentes en esta aplicación es una aportación local.

La [procedencia y lista de repositorios oficiales](../../README.md#procedencia-modelos-oficiales-y-desarrollo-del-proyecto)
distingue también los cambios de la copia local SAM-6D y la copia institucional
de FoundationPose. El [mapa de documentos y carpetas](../../README.md#estructura-de-la-documentación)
explica qué guía consultar y cómo interpretar configuraciones, manifests, JSON,
JSONL, CSV, informes HTML, vídeos y logs. Los resultados guardados no son documentación
oficial de los modelos ni validaciones automáticas de precisión física.

## Arranque

Desde `/home/raca/projects`:

```bash
bash SAM6D/run.sh ui --port 8770
```

Abrir `http://127.0.0.1:8770`. Si se accede desde otro ordenador, usar un túnel SSH
(`ssh -L 8770:127.0.0.1:8770 raca@disassembly-0`). La cámara USB debe estar conectada
al servidor donde se ejecuta Python: el túnel comparte la interfaz, no el USB.
El servidor escucha únicamente en loopback. Al arrancar precarga SAM ViT-H, DINOv2
y FoundationPose en GPU. El panel de estado avisa cuando están listos. Las peticiones
enviadas durante la carga esperan en la cola. El formulario selecciona FoundationPose
cuando está disponible: SAM detecta y entrega la máscara; FoundationPose estima la pose.

Para precargar la opción más rápida desde el principio:

```bash
bash SAM6D/run.sh ui --port 8770 --preload fastsam
```

`--preload none` omite la precarga y carga cada motor cuando se solicita por primera vez.
En todos los casos, FoundationPose permanece residente tras su primera carga. Cambiar
el segmentador o la variante de SAM carga la nueva selección; no se mantienen todas
las variantes a la vez. Un CAD nuevo sigue necesitando preparar sus vistas y referencias.

## Métodos

| Selección | Entrada | Resultado |
| --- | --- | --- |
| SAM-6D, RGB | RGB y CAD | Máscaras y score de apariencia/similitud |
| SAM-6D, RGB-D | RGB, profundidad alineada, calibración y CAD | Detección con geometría |
| SAM-6D + pose PEM | RGB-D, CAD y CUDA | Detección y pose del modelo PEM original |
| FoundationPose | RGB-D, CAD y CUDA | Máscara inicial automática o manual y pose FoundationPose |

El selector de método elige quién estima la pose. En el modo automático, SAM/FastSAM proporciona
la máscara inicial a FoundationPose; el modo de caja manual se explica al final. Su score de detección no es una probabilidad
de que la pose sea correcta. La interfaz obliga a usar RGB-D y CUDA para FoundationPose.
Se puede elegir un CAD del servidor o subirlo desde el navegador; indicar sus unidades.
La pose se devuelve en milímetros, incluyendo la transformación respecto al CAD original.

Se conserva el flujo de archivos, ejemplos y PEM anterior. Las inferencias se procesan
en una cola con un único trabajador GPU; los fallos quedan visibles en pantalla y en
`error.log` dentro del trabajo. Los detalles del proceso de pose están en `pose.log`.

## Parámetros de la interfaz

Los valores siguientes son los valores iniciales del formulario y de `inputs.py`.
Cambiar un control afecta al siguiente trabajo; los resultados ya guardados no se recalculan.
Los parámetros realmente usados quedan en `inputs.json` y en `result.json`.
Los controles de cámara actúan sobre la adquisición, no sobre un trabajo ya enviado. El intervalo de refresco SAM puede cambiarse durante el seguimiento activo.

### Entradas y selección de modelo

| Control / clave JSON | Valor inicial y opciones | Qué significa y cuándo cambiarlo |
| --- | --- | --- |
| Ubicación de los archivos / `file-source` | Archivos de disassembly-0 (`server`); también `upload` y `realsense` | `server` lee archivos del servidor; `upload` envía los del ordenador del navegador; `realsense` obtiene RGB, profundidad y calibración de la cámara conectada al servidor al enviar la orden. |
| Imagen RGB / `rgb` | Sin seleccionar; JPG o PNG | Imagen de color a analizar. Hasta 4 millones de píxeles y 20 MiB. No se debe redimensionar sin ajustar también profundidad y calibración. |
| Modelo CAD / `cad` | Sin seleccionar; STL, OBJ o PLY | Geometría del objeto buscado. Máximo 50 MiB y 250.000 caras; exportar STEP a malla antes de usarlo. No se usa el nombre de la foto para identificar la pieza. |
| Modo de análisis / `mode` | `rgb`; alternativa `rgbd` | `rgb` usa similitud semántica y apariencia. `rgbd` añade coherencia geométrica con la profundidad. La pose 6D exige RGB-D. |
| Unidades del CAD / `cad_units` | `mm`; también `cm`, `m` | Unidad real de las coordenadas de la malla. Se convierte a milímetros internamente. Una elección incorrecta cambia el tamaño físico del objeto y puede arruinar la pose; no es un ajuste de visualización. |
| Motor de pose 6D / `pose_backend` | El formulario selecciona `foundationpose` si está disponible; la API conserva `pem` como valor por omisión | Elige el estimador de pose. Ambos reciben detecciones de SAM-6D. FoundationPose recibe la máscara SAM/FastSAM, fuerza pose, RGB-D y CUDA y puede seguir grabaciones. Elegir PEM no activa por sí solo la casilla de pose. |
| Generador de máscaras / `segmentor` | `sam`; alternativa `fastsam` | SAM genera máscaras con el modelo SAM seleccionado; FastSAM-x es la alternativa orientada a velocidad. DINOv2 compara las regiones con el CAD en ambos casos. No cambia el estimador de pose elegido. |
| Estimar pose 6D / `pose` | Desactivado (`false`) | Añade orientación y posición tras la detección. Al activarlo se seleccionan RGB-D y CUDA en la interfaz. Con FoundationPose permanece activado y bloqueado. Sin una detección aceptada no se estima pose. |
| Profundidad alineada / `depth` | Sin seleccionar; NPY o PNG de 16 bits | Mapa raw `uint16`, del mismo tamaño que RGB y alineado con él. Máximo 20 MiB; debe contener medidas válidas. Solo es necesario en RGB-D. No sirve una vista de profundidad coloreada. |
| Calibración de cámara / `camera` | Sin seleccionar; JSON | Intrínsecos y escala de la profundidad de esa misma captura. Máximo 1 MiB. Se admite formato SAM-6D/BOP o metadatos RealSense reconocidos por el preparador. En directo se obtiene del SDK automáticamente. |

En **Probar con los datos del ensamblaje**, **Captura** elige un RGB-D preparado
y **CAD a comparar** el modelo que se buscará en él. **Ejecutar ejemplo** usa
esas entradas con la configuración actual. No decide la clase por el nombre de la foto.

### Configuración avanzada

| Control / clave JSON | Valor inicial y rango | Efecto de cambiarlo |
| --- | --- | --- |
| Dispositivo / `device` | `cuda`; también `auto`, `cpu` | CUDA usa la GPU. `auto` selecciona un dispositivo disponible; no garantiza GPU. CPU permite detección, con mayor tiempo de cálculo; la pose requiere que el dispositivo efectivo sea CUDA. FoundationPose bloquea el selector en CUDA. |
| SAM / `backbone` | `vit_h`; también `vit_l`, `vit_b` | Variante del modelo SAM. ViT-H es la de mayor tamaño de estas tres; L y B reducen memoria y coste de cálculo, con resultados que pueden variar. Solo afecta a `segmentor=sam`; se desactiva con FastSAM. Las opciones sin pesos disponibles se deshabilitan. |
| Score mínimo guardado / `min_score` | `0.30`; de 0 a 1 | Descarta detecciones cuyo score final está por debajo de este valor. Bajar el umbral conserva más candidatos de baja puntuación para inspección; no recupera regiones descartadas antes por el generador o el emparejamiento. Debe ser menor o igual que `confidence`. |
| Score para aceptar y dibujar / `confidence` | `0.50`; de 0 a 1 | Determina qué detecciones conservadas se aceptan, se dibujan y pueden iniciar la estimación de pose. Subirlo exige mayor score y puede dejar el trabajo sin detección aceptada. Bajarlo admite más candidatos y puede aumentar falsos positivos. |
| Estabilidad de máscara SAM / `stability` | `0.97`; de 0 a 1 | Filtra máscaras según cuánto cambia su segmentación al variar el umbral interno de binarización. Un valor mayor exige máscaras más estables; uno menor conserva más propuestas. Solo afecta a SAM. No es estabilidad temporal del vídeo. |
| Calidad de máscara SAM · IoU / `pred_iou` | `0.88`; de 0 a 1 | Umbral de calidad de máscara predicha por SAM. Subirlo descarta más máscaras; bajarlo permite propuestas menos fiables. Es una predicción del modelo, no una IoU medida frente a una anotación real. Solo afecta a SAM. |
| Puntos por lote / `points_per_batch` | `64`; entero de 1 a 128 | Cantidad de puntos de consulta que SAM procesa juntos. Subirlo puede mejorar rendimiento si cabe en GPU; bajarlo reduce memoria máxima a costa de más lotes. No cambia por sí mismo la cuadrícula total de puntos muestreados. Solo afecta a SAM. |
| FastSAM · confianza de propuestas / `fastsam_conf` | `0.05`; formulario de 0.01 a 1 | Filtro interno de propuestas de FastSAM. Subirlo reduce propuestas de baja confianza; bajarlo puede recuperar objetos y añadir ruido. No es el score final de coincidencia con el CAD. Solo afecta a FastSAM; la API valida de 0 a 1. |
| FastSAM · IoU de propuestas / `fastsam_iou` | `0.90`; formulario de 0.01 a 1 | Umbral de solapamiento de la supresión de propuestas de FastSAM. Un valor menor elimina más propuestas solapadas; uno mayor permite conservar más. Solo afecta a FastSAM; la API valida de 0 a 1. |
| NMS entre detecciones / `nms` | `0.25`; de 0 a 1 | Umbral IoU de supresión de detecciones solapadas de la misma clase después de la comparación con el CAD. Un valor menor es más restrictivo; uno mayor conserva más solapamientos y puede dejar duplicados. Es un filtro distinto de `fastsam_iou`. |
| Agregación de vistas CAD / `aggregation` | `avg_5`; también `mean`, `median`, `max` | Resume las similitudes semánticas de una región con las 42 vistas del CAD. `avg_5`: media de las cinco mayores; `mean`: media de todas; `median`: mediana; `max`: mayor similitud. `max` puede favorecer una única vista coincidente; las medias consideran varias vistas. Cambiarlo modifica el score y puede requerir revisar los umbrales. |

Ejemplo: con `min_score=0.30` y `confidence=0.50`, una detección de score `0.42`
queda guardada para inspección, pero no se acepta ni inicializa la pose. Una de `0.61`
sí supera ambos filtros. Estos scores no son probabilidades calibradas.

En RGB, el score final es `(semántica + apariencia) / 2`. En RGB-D es
`(semántica + apariencia + geometría × fracción_visible) / (2 + fracción_visible)`.
Por eso un mismo umbral no tiene por qué producir el mismo comportamiento en ambos modos.
Ni estos scores ni la caja proyectada de FoundationPose validan por sí solos la exactitud física.

### Cámara y grabación

| Control | Valor inicial y opciones | Qué controla |
| --- | --- | --- |
| Cámara / `serial` | Se lista lo detectado en el servidor | Selecciona la RealSense por número de serie. Si la API omite el número, solo se elige automáticamente cuando existe una única cámara. No busca cámaras USB del ordenador del navegador. |
| Resolución / `width`, `height` | `640 × 480`; alternativa `1280 × 720` | Resolución solicitada para color y profundidad. Más resolución aumenta detalle, transferencia, almacenamiento y trabajo por fotograma. El dispositivo debe admitir la combinación elegida. |
| Fotogramas por segundo / `fps` | `15`; alternativa `30` | Frecuencia solicitada de adquisición y frecuencia nominal del MP4. Subirla aumenta datos y escritura. El índice JSONL conserva las marcas de tiempo reales aunque se pierdan fotogramas o la escritura no sostenga la frecuencia solicitada. |
| Origen del CAD / `camera-cad-source` | Archivo del servidor (`server`); alternativa `upload` | Con la cámara como entrada, el CAD sigue siendo un archivo: elegirlo del servidor o subirlo desde el navegador. No cambia el origen de los fotogramas. |
| Grabaciones guardadas / `recording_id` | Sin selección | Elige una grabación finalizada con fotogramas. Permite reproducirla, descargarla y seguirla con FoundationPose usando el CAD elegido. No sustituye el fotograma del directo cuando se pulsa «Analizar fotograma actual». |

| Intervalo de refresco SAM / `redetect_interval_s` | `3` s en inicialización automática; `0` en modo manual | Intervalo entre comprobaciones SAM/FastSAM en segundo plano durante el seguimiento rápido. `3` s equivale a aproximadamente 0,33 Hz. `0` desactiva; valores positivos admitidos: 1–3600 s. Se puede cambiar con el seguimiento activo. Se conservan CAD, segmentador, parámetros y umbral del análisis inicial. |

**Abrir cámara** inicia una adquisición compartida. **Cerrar cámara** la detiene y finaliza
una grabación activa. **Guardar captura RGB-D** conserva el par actual; **Exportar imagen PNG**
hace además la descarga del color. **Grabar vídeo** y **Detener grabación** delimitan la sesión.
**Seguir grabación con FoundationPose** segmenta y registra el primer fotograma y sigue los
siguientes; no ejecuta una estimación de pose continua sobre el visor del directo.
Resolución, FPS y dispositivo no se pueden cambiar mientras la cámara está abierta.


### Calibración, unidades y resultados

| Campo | Significado |
| --- | --- |
| `cam_K` | Nueve números, en orden por filas: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`. `fx`, `fy` son focales y `cx`, `cy` el punto principal, en píxeles. Deben corresponder a la resolución de la captura. |
| `depth_scale` | Milímetros por unidad raw en formato BOP. Por ejemplo, raw `500` con escala `1` representa `500 mm`. El SDK RealSense entrega metros por unidad: se multiplica por 1000 al guardar el JSON. |
| `dist_coeffs` | Distorsión residual. La entrada debe estar rectificada: no se aceptan coeficientes residuales distintos de cero. La adquisición RealSense alinea y rectifica antes de guardar cuando corresponde. |
| `R` | Matriz de rotación 3×3 del modelo respecto a la cámara; no son ángulos Euler. |
| `t_mm` | Traslación en milímetros del modelo usado internamente por el estimador. El modelo puede estar centrado respecto al CAD original. |
| `T_camera_from_input_cad_mm` | Transformación homogénea 4×4 desde las coordenadas del CAD original, expresadas en milímetros, hasta la cámara. Usar este campo si se necesita respetar el origen original de la pieza. |
| `pose_score` | Puntuación del estimador de pose cuando se exporta; no debe confundirse con el score de detección SAM-6D ni interpretarse como probabilidad calibrada. |
| `timestamp` / `device_timestamp_ms` | Instante del servidor en segundos Unix e instante del dispositivo en milisegundos, respectivamente. Son referencias distintas; no se deben restar directamente entre sí. |

### Opciones del servidor

Se pasan después de `ui`, por ejemplo:

```bash
bash SAM6D/run.sh ui --port 8771 --data-dir /home/raca/projects/SAM6D/results/interface_prueba
```

| Opción | Valor por defecto | Función |
| --- | --- | --- |
| `--port` | `8770` | Puerto HTTP local. El servidor escucha en `127.0.0.1`. Si ya está ocupado, abrir la instancia existente o elegir otro puerto y ajustar el túnel SSH. |
| `--data-dir` | `/home/raca/projects/SAM6D/results/interface` | Raíz de trabajos de detección y caché de CAD. |
| `--capture-dir` | `/home/raca/projects/SAM6D/captures` | Raíz de medios de cámara: imágenes en `images/` y grabaciones en `videos/`. Es independiente de `--data-dir`. |
| `--file-root` | `/home/raca/projects` | Límite del explorador de archivos del servidor. Solo permite leer rutas resueltas dentro de esta carpeta; no cambia la carpeta de resultados. |
| `--preload` | `sam` | Precarga SAM ViT-H, DINOv2 y FoundationPose. `fastsam` precarga FastSAM en lugar de SAM; `none` desactiva la precarga inicial. No abre la cámara. |
| `--backbone` | `vit_h` | Variante SAM de la precarga: `vit_h`, `vit_l` o `vit_b`. Se ignora si `--preload fastsam`. Cambiarla después desde el formulario carga la nueva variante. |
| `--help` | Sin valor | Muestra la ayuda de arranque y termina. |

| Variable de entorno | Valor del lanzador | Función |
| --- | --- | --- |
| `FOUNDATIONPOSE_PYTHON` | `.venvs/foundationpose/bin/python` bajo `/home/raca/projects` | Intérprete aislado del trabajador FoundationPose. |
| `FOUNDATIONPOSE_ROOT` | `/home/raca/projects/FoundationPose` | Repositorio original y sus pesos. |
| `FOUNDATIONPOSE_NATIVE` | `SAM6D/pipeline/native/foundationpose` bajo `/home/raca/projects` | Extensión `mycpp` compatible con el Python del trabajador. |
| `TORCH_EXTENSIONS_DIR` | `/home/raca/projects/.runtime/cache/torch_extensions` | Caché de compilación de extensiones CUDA/PyTorch. |
| `OMP_NUM_THREADS` | `16`, salvo valor ya definido | Límite de hilos OpenMP del proceso lanzado. FoundationPose fija cuatro hilos PyTorch en su trabajador. |
| `MKL_NUM_THREADS` | `16`, salvo valor ya definido | Límite de hilos de operaciones MKL. No es el número de trabajadores GPU. |

Las cuatro primeras variables admiten rutas alternativas antes del comando de arranque.
El lanzador prepara además `LD_LIBRARY_PATH`, `PATH`, `MPLCONFIGDIR` y `XDG_CACHE_HOME`
para las bibliotecas y cachés del entorno; no es necesario configurarlas para el uso normal.

### Valores internos que no aparecen como controles

- **42 vistas por CAD**: referencias de detección; no es un parámetro del formulario.
- **FoundationPose: 5 iteraciones de registro y 2 por seguimiento**: valores de
  `foundationpose_worker.py`, no campos configurables desde la interfaz actual.
- **Cola: hasta cuatro trabajos pendientes o en ejecución**, con un único trabajador GPU.
- **Grabación: diez minutos como máximo**, al menos 256 MiB libres al empezar y
  parada por debajo de 192 MiB. Estos límites están en `camera/service.py`.
- **Fotograma caducado: más de dos segundos** desde su recepción: se rechaza para
  evitar analizar una imagen vieja después de una interrupción de la cámara.


## Detección automática con la imagen en directo

1. En **Ubicación de los archivos**, seleccionar **Intel RealSense en directo**.
2. Seleccionar el CAD, sus unidades, SAM o FastSAM y el modo RGB/RGB-D.
3. Pulsar **Abrir cámara** y comprobar el visor.
4. Pulsar **Iniciar detección en directo**. Se analiza el fotograma actual y, al
   terminar, se solicita otro par RGB-D actual. El resultado anterior permanece visible
   mientras se calcula el siguiente. No se analiza una foto cargada previamente.
5. **Detener detección** impide nuevos análisis; uno ya enviado termina normalmente.
   También se detiene al cerrar/perder la cámara, cambiar a archivos o recibir un error.

Se ejecuta un único trabajo de esta pestaña cada vez, con una pausa interna de 250 ms
entre su finalización y el siguiente envío. Los 15/30 FPS de adquisición no son FPS
de inferencia: la frecuencia de resultados depende de SAM/FastSAM, CAD, modo y GPU.
La primera inferencia puede tardar más por la carga de modelos y referencias.

El botón **Analizar fotograma actual** conserva el análisis manual de una sola imagen.
La grabación y el visor siguen disponibles durante el análisis automático. Si se activa
pose PEM o FoundationPose, también se estima en cada trabajo; esto no usa `track_one`
ni convierte el visor en un seguimiento de pose por cada fotograma de vídeo.

El CAD y los ajustes se leen para cada envío; los cambios afectan a los ciclos siguientes.
Cada ciclo conserva sus entradas y resultados bajo `results/interface/jobs/` y puede
consumir espacio en disco. El servidor detiene nuevos ciclos si quedan menos de
256 MiB libres. La detección requiere la RealSense conectada al servidor;
no la abre en otra aplicación simultáneamente.

## RealSense en directo

1. Seleccionar **Intel RealSense** como origen y elegir el CAD.
2. Elegir cámara, resolución y frecuencia; pulsar **Abrir cámara**.
3. El visor muestra el directo. **Analizar fotograma actual** toma una copia nueva
   y sincronizada de RGB, profundidad y calibración en el momento de enviar la orden.
4. **Guardar captura RGB-D** guarda una instantánea descargable. No congela el directo
   ni hace que la siguiente inferencia reutilice una captura antigua.
5. Cerrar la cámara al terminar para liberarla para otras aplicaciones.

Una sola adquisición alimenta visor, capturas y grabación. La profundidad uint16 se
alinea con color; se rectifican ambos canales cuando corresponde y se guarda la matriz
intrínseca correspondiente. `depth_scale` sigue la convención BOP: milímetros por
unidad de profundidad raw. Los fotogramas obsoletos, sin profundidad o sin cámara
producen un error explícito; no se sustituyen por imágenes de ejemplo.

## Grabaciones y seguimiento

**Iniciar grabación** guarda un MP4 de color y, además, los pares RGB/profundidad
sin pérdida, calibración e índice JSONL con sus marcas de tiempo. El MP4 permite
previsualizar el vídeo; los PNG RGB-D son la entrada de seguimiento de FoundationPose.
**Detener grabación** finaliza los archivos y añade la sesión al selector de grabaciones.

Con FoundationPose seleccionado, elegir una grabación y pulsar **Seguir grabación**.
El primer fotograma se segmenta con SAM/FastSAM y se registra con FoundationPose;
los siguientes se procesan con `track_one`. Se generan `pose_tracking.jsonl` y
`pose_tracking.mp4`, además de la pose y su proyección inicial. El seguimiento se
procesa sobre la grabación, no se presenta como inferencia continua en tiempo real.

La grabación requiere al menos 256 MiB libres al comenzar y se detiene con menos de
192 MiB disponibles o tras diez minutos. Cerrar la cámara también finaliza una grabación
activa. Revisar el estado mostrado si la adquisición o escritura fallan.

## Carpetas

```text
SAM6D/pipeline/sam6d_ui/
  server.py                 HTTP, validación y cola de trabajos
  engine.py                 Segmentación y selección del estimador
  inputs.py                 Validación y normalización de entradas
  camera/
    realsense.py            SDK, alineación y calibración
    service.py              Adquisición compartida, capturas y vídeos
    tracking.py             Seguimiento y calendario de refrescos
    redetection.py          Nueva máscara SAM en segundo plano sobre RGB-D en RAM
  backends/
    foundationpose.py       Disponibilidad y acceso al trabajador persistente
    foundationpose_worker.py  Registro y seguimiento con el modelo original
    resident.py             Protocolo y ciclo de vida del proceso GPU
    resident_worker.py      Modelos, registro y seguimiento en el mismo proceso
    geometry.py             Proyección de caja CAD y coherencia con la detección SAM
  static/                   Interfaz, estilos y controles de cámara
SAM6D/results/interface/
  jobs/sam6d/<id>/           Entradas, detecciones, pose PEM y logs
  jobs/foundationpose/<id>/  Entradas, pose, seguimiento y logs
  cache/cad/                Referencias reutilizables de CAD nuevos
  logs/foundationpose-worker.log  Salida del proceso de pose persistente
  live/<id>/session.json    FPS, estado y errores de una sesión de seguimiento
  validation/resident-20261029/  Evidencias de las pruebas de rendimiento
SAM6D/captures/
  images/<id>/              rgb.png, depth.png, camera.json, capture.json
  videos/<id>/
    video.mp4
    camera.json
    recording.json
    frames.jsonl
    frames/000000/           rgb.png y depth.png por fotograma
```

Los trabajos antiguos de `pipeline/prepared/sam6d_ui/jobs/` siguen accesibles por
su identificador. Los datos preparados y las suites conservan sus ubicaciones propias.
`--data-dir` cambia la salida de inferencias; `--capture-dir` cambia la de capturas
y vídeos. Las capturas anteriores de `results/interface/camera/captures/` se han
trasladado a `captures/images/`, conservando enlaces de compatibilidad en la ruta
anterior. Lo mismo se aplica a `camera/recordings/` y `captures/videos/`.
Las copias de entrada dentro de cada trabajo se mantienen para trazabilidad.

```bash
bash SAM6D/run.sh ui --capture-dir /ruta/con/espacio/captures
```

La detección continua comprueba el espacio disponible tanto en la carpeta de
resultados como en la de capturas antes de enviar otro análisis.

## Entorno FoundationPose

SAM-6D conserva `.venvs/assembly-runtime`. FoundationPose utiliza el entorno adicional
`.venvs/foundationpose`, comparte las dependencias existentes y añade sus propias
bibliotecas. Este adaptador no escribe en el código ni en los pesos de la copia instalada en
`FoundationPose/`. Esa copia procede del repositorio institucional de SDU; véase
la distinción con el repositorio oficial de NVIDIA en la guía de procedencia.
La extensión nativa para Python 3.10 se guarda en
`SAM6D/pipeline/native/foundationpose/`; la caché CUDA está en `.runtime/cache/torch_extensions/`.

Para reinstalar las dependencias adicionales:

```bash
bash SAM6D/pipeline/install_foundationpose.sh
```

Requiere el entorno SAM-6D, el repositorio y pesos de FoundationPose, CUDA, compilador,
CMake y espacio suficiente. Véase `requirements-foundationpose.txt`. Puede cambiarse
la ubicación mediante `FOUNDATIONPOSE_PYTHON`, `FOUNDATIONPOSE_ROOT` y
`FOUNDATIONPOSE_NATIVE`. El selector muestra el motivo cuando FoundationPose no está
disponible, antes de aceptar un trabajo que lo necesite.

Los pesos de SAM-6D de esta máquina están enlazados a `/dev/shm/pddl-assembly-runtime/`.
Esa ubicación se vacía al reiniciar. Para recuperarlos:

```bash
# Descarga directa de los pesos usados por la interfaz:
.venvs/assembly-runtime/bin/python SAM6D/pipeline/download_runtime_assets.py --cache /dev/shm/pddl-assembly-runtime/checkpoints
.venvs/assembly-runtime/bin/python SAM6D/pipeline/download_pose_assets.py --cache /dev/shm/pddl-assembly-runtime/checkpoints
```

Ejecutar con las variables de bibliotecas del entorno si se invoca Python directamente.
Para conservarlos entre reinicios, emplear almacenamiento persistente y actualizar sus
enlaces después de comprobar el espacio disponible.

## Verificación

Las pruebas de cámara usan una fuente simulada explícita y comprueban capturas,
unidades de profundidad, grabación, cierre, errores y rutas HTTP. No abren el robot.
FoundationPose se ha probado con inferencia CUDA real sobre RGB-D grabado, tanto en
registro inicial como en seguimiento. La validación con una RealSense física requiere
que el servidor detecte el dispositivo.


## Exportar imágenes y descargar vídeos

En «Intel RealSense en directo», abrir la cámara y pulsar **Exportar imagen PNG**
para descargar el fotograma actual. Se conserva también el par RGB-D y la calibración
en la carpeta de capturas. **Guardar captura RGB-D** permite descargar cada componente.
**Grabar vídeo** inicia la grabación y **Detener grabación** la finaliza; seleccionar
una grabación guardada muestra su reproductor y el enlace **Descargar vídeo MP4**.
En los resultados también se puede exportar la imagen original, la detección y la proyección de pose.

FoundationPose necesita una máscara inicial, obtenida automáticamente con SAM/FastSAM
en esta interfaz. No es necesario dibujar una bounding box manualmente. Una caja 2D
por sí sola incluye fondo y no equivale a la máscara de segmentación. La caja 3D
proyectada en el resultado permite inspeccionar la pose estimada; no garantiza su exactitud.


## Seguimiento rápido de FoundationPose

Seleccionar RealSense, el CAD y FoundationPose, abrir la cámara y pulsar
**Localizar e iniciar seguimiento rápido**. SAM6D y FoundationPose calculan una pose
inicial; después un trabajador persistente usa `track_one(iteration=2)` sobre el
RGB-D más reciente. Los modelos permanecen cargados en GPU. Entre refrescos no se repite SAM6D
ni se inicia otro proceso por fotograma. Si el intervalo SAM es mayor que cero, SAM comprueba periódicamente en segundo plano; la pose solo se vuelve a registrar cuando necesita recuperación. El visor muestra la pose sobre el color
actual y los FPS y latencia medidos. No se garantiza la frecuencia de adquisición
de15/30FPS: depende de resolución, objeto, GPU y otros procesos.

La escena debe permanecer estable durante la primera localización. Si se pierde
el objeto y el refresco SAM está activo, se espera una nueva detección para recuperar la pose. Si el refresco está desactivado o cambia el CAD, usar **Detener seguimiento** y volver a localizar. No se estima una nueva confianza de detección
en cada imagen; el score del análisis inicial no acredita el seguimiento posterior.

Se toma siempre el último par disponible, sin acumular una cola de vídeo. Los
fotogramas intermedios se intercambian en memoria temporal y no se guardan como
trabajos individuales. Para conservar vídeo o capturas siguen disponibles los
controles de cámara. Los logs de sesión se guardan en `results/interface/live/`.
Se detienen los análisis nuevos mientras este trabajador está activo para evitar
competir con otra inferencia de la misma interfaz. Al cerrar la cámara o el servidor
se detiene el seguimiento. Los modelos siguen en GPU al detenerlo o cerrar la cámara;
solo se descarga el proceso al cerrar el servidor. DINOBOT y FoundationPose original
conservan su código.


## FoundationPose por separado, con caja manual

1. Arrancar la interfaz normalmente: `bash SAM6D/run.sh ui --port 8770`.
2. Seleccionar **Intel RealSense en directo**, abrir la cámara y elegir el CAD y sus unidades.
3. Elegir **FoundationPose** y, en **Inicialización de FoundationPose**, seleccionar
   **Caja manual · solo FoundationPose**.
4. Pulsar **Capturar fotograma para seleccionar**. Se guarda un par RGB-D alineado
   con calibración en `captures/images/` y se muestra una imagen congelada.
5. Arrastrar sobre esa imagen para dibujar una caja ajustada al objeto. Las coordenadas
   se conservan en píxeles de la resolución original; no se recorta ni cambia la matriz K.
6. Pulsar **Estimar pose desde mi caja**, o **Localizar e iniciar seguimiento rápido**
   para usar esa inicialización y continuar siguiendo imágenes actuales.

Este camino ejecuta directamente el trabajador de FoundationPose con el CAD subido.
Omite SAM/FastSAM, DINOv2, comparación del CAD y renderizado de las42vistas.
El resultado no contiene score de detección: la selección la ha hecho el usuario.

FoundationPose necesita una máscara inicial; aquí se usa el rectángulo como una
**máscara aproximada**. Una caja incluye fondo y no equivale a una segmentación precisa.
Ajustarla al objeto y evitar otras piezas en su interior. Una pose que se dibuja bien
no acredita exactitud física. Si se mueve el objeto después de capturar, volver a
capturar y seleccionar antes de iniciar. Para el seguimiento la captura inicial
debe tener menos de120segundos. La primera carga de FoundationPose también tarda.

Se conservan `mask_000.png`, `manual_selection.json`, `pose.json`, la proyección y
`pose.log` dentro del trabajo de `results/interface/jobs/foundationpose/`.
El modo manual actual está preparado para la RealSense del servidor; la entrada
automática de archivos y ejemplos mantiene su funcionamiento anterior.


## Modelos residentes, rendimiento y visor compacto

El proceso de FoundationPose carga sus dos redes y el contexto CUDA una vez. El registro
inicial y `track_one` reutilizan ese mismo proceso y estimador del CAD; iniciar seguimiento
no vuelve a cargar pesos ni otra copia en GPU. Cada registro reinicia la pose del estimador,
para impedir que una inicialización fallida reutilice una pose anterior. Las referencias
DINOv2 del CAD actual también quedan en GPU; se invalidan si cambian los archivos o el modelo.

**Abrir visor compacto** abre `/viewer` en una ventana pequeña. Muestra una sola imagen:
pose seguida, último resultado o cámara original. Al abrirlo con ese botón se pausan las
previsualizaciones de cámara y seguimiento del panel principal; se restauran al cerrarlo.
Los controles y el bucle de detección permanecen en la página principal, que debe seguir
abierta para gestionar los análisis continuos. Abrir `/viewer` directamente permite ver el
servidor, pero no pausa automáticamente otras pestañas ni inicia detección por sí solo.

Las consultas de resultados pasan de 1000 a 200 ms. Esto reduce la espera de presentación;
no acelera las redes. La visualización solicita un JPEG actual cada vez y espera a recibirlo y decodificarlo antes de pedir otro. Una conexión SSH lenta limita los FPS visibles, pero no acumula una secuencia de imágenes antiguas.

Prueba del 29/10/2026, RTX 4090 y RGB-D guardado de 640×480, mismo CAD y modelos cargados:

| Operación | Tiempo observado |
| --- | --- |
| SAM ViT-H + DINOv2, detección completa | 1,28 s |
| FastSAM + DINOv2, detección completa | 0,34 s |
| Registro FoundationPose y exportación | 0,86–0,89 s |
| Trabajo completo SAM + FoundationPose | 2,26 s |
| Trabajo completo FastSAM + FoundationPose | 1,31 s |
| Seguimiento, solo cálculo sobre imágenes repetidas | 27,4 FPS |
| Consulta de estado del visor, navegador local | ≈6 ms de media en la primera prueba |

La precarga SAM/DINOv2/FoundationPose duró unos 8,5 s en esa prueba. El primer registro
del CAD puede tardar más por inicialización de kernels y referencias. Las cifras no incluyen
red remota ni validan seguimiento de un objeto moviéndose físicamente. No son garantías
de FPS: se han separado carga, detección, registro y seguimiento. Los JSON de evidencia
están en `results/interface/validation/resident-20261029/`.

`pose.log` identifica el PID residente, su log global y el tiempo de esa petición. `pose.json`
incluye `models_reused` y `worker_pid`; `result.json` conserva los tiempos por etapa.
Los errores de carga se muestran en el panel inicial, los errores de análisis en su resultado
y `error.log`. En una desconexión del trabajador, la petición falla y la siguiente puede
reiniciarlo; entonces habrá una nueva carga de modelos.


## Comprobación periódica de SAM en segundo plano

Seleccionar inicialización automática, el CAD y SAM/FastSAM. En **Intervalo de refresco
SAM** indicar `3` para comprobar la pieza cada tres segundos (aproximadamente 0,33 Hz).
**Localizar e iniciar seguimiento rápido** calcula la pose inicial y después mantiene
FoundationPose siguiendo imágenes actuales mientras SAM trabaja en otro hilo con los
modelos ya cargados. Modificar el intervalo no reinicia la sesión. Poner `0` desactiva
las comprobaciones. En **Caja manual · solo FoundationPose** no se ejecuta SAM.

Se admite un único trabajo SAM activo y no hay cola de verificaciones. Si tarda más
que el intervalo, se saltan los instantes vencidos. Cambiar el intervalo o detener la
sesión invalida el resultado pendiente. Al detenerse se espera a que SAM libere el motor
compartido; si aún está acabando una inferencia, la sesión permanece ocupada para
impedir que otro análisis modifique los mismos modelos.

### Cuándo se conserva o recupera la pose

Cada comprobación usa una copia del RGB-D que FoundationPose acaba de seguir. Se guarda
la caja del CAD proyectada por esa pose y se compara con la caja de la máscara SAM
obtenida **sobre esa misma imagen**; no se compara una detección antigua con la pose
actual de un objeto que ha podido moverse.

| Comprobación interna | Valor / comportamiento |
| --- | --- |
| Solapamiento de cajas, IoU | Al menos `0,25` para considerarlas coherentes. |
| Distancia entre centros | Como máximo `0,5` veces la diagonal de la caja detectada por SAM. |
| Discrepancias necesarias | Dos comprobaciones consecutivas incoherentes antes de recuperar la pose. |
| Detecciones ausentes | Una conserva el seguimiento con aviso; dos retiran la pose y esperan otra detección. |
| Edad máxima del RGB-D | Dos segundos al consumir el resultado; los resultados más antiguos se descartan. |
| Error de `track_one` | Retira la pose y solicita una detección nueva. Un resultado capturado antes del fallo no puede recuperarla. |
| Resultado coherente | Conserva la pose seguida y no llama a `register`. |
| Proyección no comprobable | Registra `unverified`; no anuncia que la pose esté verificada. |

La proyección utiliza las ocho esquinas de la caja 3D del CAD. La comparación es una
heurística de coherencia en imagen: no garantiza exactitud 6D, no valida por sí sola
la orientación o profundidad y puede no detectar errores que mantengan una caja
similar. El score SAM identifica la pieza; no es una probabilidad de pose correcta.
Con varias copias del mismo CAD se elige la detección aceptada de mayor score y podría
cambiarse de ejemplar.

Cuando se confirma una discrepancia, o falla el seguimiento, una máscara reciente
permite volver a registrar FoundationPose una vez. **La recuperación sí puede pausar
brevemente la pose** mientras se realiza ese registro; durante esa recuperación se
muestra la cámara original. Una comprobación SAM normal mantiene el vídeo anotado
sin cambiar de visor. Cambiar la calibración o desconectar la cámara detiene la sesión.

SAM y FoundationPose comparten GPU: pueden competir por cómputo y memoria aunque no
se bloqueen en el mismo bucle. SAM ViT-H puede reducir temporalmente los FPS más que
FastSAM. No se garantiza una frecuencia concreta ni detección infalible de pérdida.

### Archivos y estado

Los espacios de trabajo de SAM y seguimiento están separados en `/dev/shm`. SAM no
sobrescribe el RGB-D que FoundationPose está procesando. La detección en segundo plano
omite la exportación de máscaras PNG, figuras y resultados JSON; conserva únicamente
la máscara elegida temporalmente. No crea capturas ni trabajos permanentes cada tres
segundos. Los pesos, descriptores de referencia y el estimador FoundationPose se reutilizan.

`results/interface/live/<sesión>/redetections.jsonl` guarda decisiones como `consistent`,
`disagreement`, `not_detected`, `discarded`, `unverified` y `relocalized`. `session.json`
incluye `sam_running`, `sam_checks`, `sam_discarded`, `sam_misses`, `sam_disagreements`,
`last_sam_agreement`, último score y error. `redetect_attempts` cuenta trabajos SAM;
`redetections` cuenta recuperaciones de pose, no todas las comprobaciones.

API: `POST /api/tracking/start` acepta `job_id` y `redetect_interval_s` (por omisión 0
para clientes existentes). `POST /api/tracking/config` recibe
`{"redetect_interval_s": 3}`. El estado se consulta en `/api/tracking/status`. Un registro
de recuperación ya empezado termina de forma atómica; detener la sesión impide publicar
su resultado después. La interfaz distingue **comprobaciones** y **recuperaciones**.

### Validación del modo asíncrono

Prueba del 29/10/2026, RTX 4090 y RGB-D guardado 640×480 simulando cámara a 15 FPS,
intervalo 3 s y modelos cargados:

| Segmentador | Duración de comprobaciones | Imágenes seguidas mientras SAM trabajaba | Mayor separación entre imágenes observada |
| --- | --- | --- | --- |
| FastSAM | 0,52–0,53 s | 5 por comprobación | 0,18 s |
| SAM ViT-H | 1,67–1,78 s | 22–23 por comprobación | 0,15 s |

Las cuatro comprobaciones fueron coherentes y provocaron cero reinicios de pose,
usando el mismo trabajador FoundationPose. Esto verifica concurrencia sobre capturas
guardadas; no valida seguimiento de un objeto moviéndose físicamente ni la red del
usuario. Las pruebas unitarias cubren pérdida, discrepancias, cambio de intervalo,
cancelación, resultados antiguos y correspondencia entre imagen y pose. La prueba de
navegador comprueba que el vídeo anotado avanza mientras el estado muestra SAM en
segundo plano. Evidencias: `results/interface/validation/async-sam-20261029/`.

## Vídeo con prioridad a la imagen más reciente

La sesión `edcd0338d28d414a869c52357fa3fe2e` terminó con 544 imágenes seguidas,
14,68 FPS y 46 ms de latencia de procesamiento. La cámara estaba configurada a
640×480 y 15 FPS; el refresco SAM estaba a 0. En una consulta durante la sesión,
el servidor indicaba 15,17 FPS y 38 ms. Esto mide captura y procesamiento en el
servidor; no acredita que el navegador estuviera mostrando esa imagen sin retraso.

El envío MJPEG anterior podía acumular imágenes en TCP/SSH o en el navegador si
el receptor consumía más despacio. En una prueba con un consumidor lento se
reprodujo un retraso de 21 imágenes, frente a 0 imágenes pendientes al solicitar
una instantánea actual. Esta prueba reproduce el mecanismo; no mide la conexión
remota del usuario.

La interfaz y `/viewer` usan ahora `static/video.js`: una petición JPEG y una
decodificación por visor, sin peticiones de vídeo simultáneas ni precarga de una
secuencia. La siguiente petición se hace tras terminar la anterior, hasta un
máximo de 30 consultas por segundo. Si se ha avanzado durante la espera, se
saltan imágenes intermedias. Una pestaña oculta deja de solicitar vídeo. El visor
compacto sigue pausando las previsualizaciones del panel principal.

Las previsualizaciones se limitan a 640 píxeles de ancho y calidad JPEG 70. La
inferencia, profundidad, calibración, capturas PNG y vídeos guardados conservan
sus datos originales. El SDK usa una cola de capacidad 1 cuando el sensor admite
esa opción, y la lectura toma el par sincronizado más reciente antes de alinearlo.
Los antiguos endpoints MJPEG siguen disponibles para clientes existentes, pero
la interfaz actual no los utiliza.

**Vídeo recibido** muestra FPS medidos en el navegador y antigüedad estimada:
antigüedad del fotograma al atender la petición más tiempo de recepción y
decodificación observado en el navegador. Es una estimación conservadora y no
requiere que los relojes de ambos equipos coincidan; no mide la exposición del
sensor ni el instante físico de refresco de la pantalla. Los FPS de seguimiento
siguen describiendo únicamente el servidor. Si el enlace es lento, habrá menos
imágenes visibles, pero cada petición nueva toma la imagen actual.

API: `GET /api/camera/frame.jpg` y `/api/tracking/frame.jpg` devuelven un JPEG con
`X-Frame-Sequence` y `X-Frame-Age-Ms`. `?after=<secuencia>` devuelve 204 si no hay
imagen nueva; también devuelve 204 si no hay una imagen vigente o el seguimiento
está refrescando/recuperando. Las respuestas tienen longitud definida, no usan
caché y permiten reutilizar la conexión HTTP. Los accesos de vídeo correctos no
se imprimen por fotograma para evitar llenar los logs.

Con red simulada de 80 ms de latencia y 64.000 bytes/s, la prueba de navegador
mantuvo una sola petición de vídeo y una antigüedad estimada de unos 542 ms,
saltando imágenes para no reproducir una cola. Las condiciones reales pueden
diferir. Las evidencias están en `results/interface/validation/latest-video-20261029/`.

Para aplicar estos cambios hay que reiniciar el servidor y recargar con Ctrl+F5.
La cámara configurada a 15 FPS genera una imagen cada 67 ms; seleccionar 30 FPS
puede aumentar la fluidez si cámara, cálculo y conexión lo permiten. La GPU acelera
SAM y FoundationPose; captura USB, alineación, JPEG, red y navegador tienen sus
propios límites.
