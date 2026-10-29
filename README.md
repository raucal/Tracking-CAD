# SAM6D: reconocer una pieza y estimar su pose a partir de un CAD

Este proyecto permite seleccionar una imagen y un CAD, localizar la pieza en la
imagen y obtener su máscara. Si también se dispone de profundidad y calibración,
puede estimar la **pose 6D**: la orientación y la posición de la pieza respecto a
la cámara.

El procesamiento se ejecuta en **`disassembly-0`**, con su **RTX 4090** cuando se
selecciona CUDA. El navegador muestra los controles y los resultados. Aunque se
abra desde otro ordenador, los modelos siguen ejecutándose en el servidor.

## Procedencia: modelos oficiales y desarrollo del proyecto

`SAM6D/` es la carpeta de nuestra integración. El repositorio publicado por los
autores de SAM-6D está dentro de `SAM6D/SAM-6D/`. La semejanza de nombres no implica
que toda esta carpeta forme parte del software oficial.

### Repositorios oficiales de referencia

| Componente externo | Repositorio oficial | Qué aprovechamos |
| --- | --- | --- |
| SAM-6D | [JiehongLin/SAM-6D](https://github.com/JiehongLin/SAM-6D) | Redes de segmentación de instancias ISM y estimación de pose PEM, referencias CAD y modelos preentrenados. |
| FoundationPose | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | Registro de pose 6D y seguimiento de objetos con sus modelos preentrenados. |
| Segment Anything, SAM | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Modelo y generador de máscaras utilizado por SAM-6D. |
| FastSAM | [CASIA-LMC-Lab/FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) | Alternativa para proponer máscaras. El enlace antiguo `CASIA-IVA-Lab/FastSAM`, presente en documentación original, redirige a este repositorio. |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Descriptores para comparar regiones de la imagen con las vistas CAD dentro de SAM-6D. |
| RealSense SDK | [realsenseai/librealsense](https://github.com/realsenseai/librealsense) | Acceso a cámara, profundidad e intrínsecos mediante `pyrealsense2`. |

Estos enlaces identifican los proyectos de origen. La interfaz común, la organización
local de resultados y los scripts del ensamblaje se documentan aquí como trabajo de
integración. Los modelos, sus pesos y sus métodos conservan la autoría de los proyectos
correspondientes; la aplicación usa inferencia con modelos preentrenados.

### Qué hemos añadido o adaptado

| Parte del proyecto | Ubicación | Aportación local |
| --- | --- | --- |
| Interfaz web común | `pipeline/sam6d_ui/static/`, `server.py` | Formularios, explorador del servidor, cola de trabajos, estados de ejecución, errores y descargas para SAM-6D/PEM y FoundationPose. |
| Orquestación de inferencia | `pipeline/sam6d_ui/engine.py`, `inputs.py` | Validación de RGB-D y CAD, unidades, selección del estimador, modelos residentes, reutilización de referencias y exportación de resultados. |
| Análisis solo RGB | `pipeline/sam6d_ui/engine.py` | Adaptación local que compara similitud y apariencia sin profundidad. No debe confundirse con la evaluación geométrica RGB-D de SAM-6D. |
| Integración FoundationPose | `pipeline/sam6d_ui/backends/` | Trabajador aislado, máscara inicial de SAM/FastSAM, conversión de unidades y del origen CAD, proyección y seguimiento de grabaciones. El registro y `track_one` los ejecuta FoundationPose. |
| Captura RealSense | `pipeline/sam6d_ui/camera/` | Adquisición compartida, visor web, alineación/rectificación, capturas RGB-D, exportación PNG y grabación MP4 con profundidad, calibración y marcas de tiempo. |
| Datos del ensamblaje | `pipeline/prepare.py`, `config.json`, `prepared/` | Preparación de nuestros CAD, catálogo y fotografías para los formatos consumidos por los modelos. |
| Lanzadores y entornos | `run.sh`, `pipeline/run_runtime.sh`, instaladores y requisitos | Arranque reproducible con dependencias y extensiones nativas compatibles con este equipo. |
| Evaluación y documentos generados | `pipeline/sam6d_suite.py`, `sam6d_suite_report.py`, `score_all_cads.py` | Suites, mediciones, comparación de CAD, auditoría de etiquetas, CSV, informe `index.html` y ZIP. |
| Pruebas | `pipeline/tests/` | Comprobaciones de entradas, casos de detección, informes, rutas HTTP y cámara simulada. |
| Adaptadores dentro de la copia SAM-6D | `SAM-6D/pddl_pipeline.py`, `README_PDDL.md`, `run_inference_pddl.py`, `run_suite_pddl.py`, `segmentor_runtime.py`, `render_pddl_templates.py` | Puntos de entrada y utilidades locales de preparación, renderizado e inferencia. Los cuatro últimos scripts están dentro de los subdirectorios correspondientes de `SAM-6D/SAM-6D/`. |

La **selección FastSAM ya existe en el SAM-6D oficial**. Nuestra aportación es su
adaptación al entorno instalado y a esta interfaz, no la creación de FastSAM ni su
primera incorporación a SAM-6D. DINOv2 también forma parte del descriptor de SAM-6D;
no se debe atribuir esta comparación a un clasificador independiente de DINOPDDL.

### Estado de las copias locales

- **SAM-6D:** el remoto `origin` de `SAM6D/SAM-6D/` apunta al repositorio oficial
  de JiehongLin. La copia contiene además los adaptadores anteriores y modificaciones
  en archivos del modelo: tratamiento de cero o una máscara, dimensiones de tensores,
  compatibilidad de FastSAM con Ultralytics y coordenadas de imagen, rutas de compilación
  PointNet2 y lectura numérica de un umbral PEM. Por tanto, no es una copia intacta del
  upstream. `git -C /home/raca/projects/SAM6D/SAM-6D diff` muestra los cambios sobre
  archivos versionados; `git status --short` muestra también los añadidos.
- **FoundationPose:** la carpeta vecina `/home/raca/projects/FoundationPose/` tiene
  configurado `origin` como `git@gitlab.sdu.dk:robot_learning/vision/foundation_pose.git`,
  una copia institucional de SDU. El repositorio oficial público de referencia es
  **NVlabs/FoundationPose**, enlazado arriba. Esta integración importa el código de la
  copia instalada y conserva allí sus pesos; su adaptador y su extensión para Python
  están separados. No se ha certificado que esa copia institucional sea idéntica,
  archivo por archivo, a una revisión del repositorio oficial. Su `README.md` local es
  una plantilla de GitLab y no sustituye la documentación oficial de NVIDIA.

## Estructura de la documentación

| Documento | Origen y finalidad | Cuándo consultarlo |
| --- | --- | --- |
| [README.md](README.md) | Guía general de nuestra integración: componentes, procedencia, estructura y puntos de entrada. | Empezar aquí para entender el proyecto y saber qué es propio. |
| [USO.md](USO.md) | Recetas locales de arranque, capturas, comandos y suites. | Ejecutar tareas concretas en este equipo. |
| [pipeline/README.md](pipeline/README.md) | Mapa de herramientas de integración, preparación y ubicación de datos. | Encontrar el script que implementa una tarea. |
| [pipeline/sam6d_ui/README.md](pipeline/sam6d_ui/README.md) | Manual de la interfaz, todos sus parámetros, RealSense, FoundationPose, calibración y resultados. | Configurar o utilizar la aplicación web. |
| [SAM-6D/README.md](SAM-6D/README.md) | Documentación del repositorio de los autores de SAM-6D. | Consultar el método, instalación y referencias del proyecto de origen. |
| [SAM-6D/README_PDDL.md](SAM-6D/README_PDDL.md) | Documento añadido por el proyecto para los adaptadores PDDL del modelo. Incluye instrucciones de etapas anteriores. | Entender esos adaptadores; para el arranque actual, priorizar esta guía y el manual de interfaz. |
| [results/dashboard/index.html](results/dashboard/index.html) | Informe generado de una suite guardada, con imágenes, scores y tiempos. | Revisar esa ejecución; no es la interfaz interactiva ni documentación del modelo oficial. |
| [maintenance/](maintenance/) | Copias previas, verificaciones y notas de reorganización. | Consultar historia o recuperar versiones anteriores; no usar sus README como instrucciones actuales. |

Los documentos de uso actuales están en los tres primeros niveles de documentación
indicados arriba. Los HTML, CSV y JSON de resultados describen ejecuciones particulares.
No se deben interpretar las medidas de una suite antigua como resultados de una nueva
captura ni como validación automática de la exactitud física de una pose.

### Estructura de carpetas

```text
/home/raca/projects/
├── SAM6D/                           Integración de este proyecto
│   ├── README.md                    Visión general y procedencia
│   ├── USO.md                       Recetas de ejecución
│   ├── run.sh                       Punto de entrada
│   ├── SAM-6D/                      Copia del repositorio SAM-6D, con cambios locales
│   │   ├── README.md                Documento del proyecto de origen
│   │   ├── README_PDDL.md           Documento de nuestros adaptadores
│   │   ├── pddl_pipeline.py         Adaptador local de comandos
│   │   └── SAM-6D/                  Directorio de código dentro del repositorio
│   │       ├── Instance_Segmentation_Model/  ISM y adaptadores de inferencia
│   │       ├── Pose_Estimation_Model/        PEM
│   │       └── Render/              Renderizado y adaptador de plantillas
│   ├── pipeline/                    Herramientas locales de integración
│   │   ├── README.md                Mapa técnico
│   │   ├── config.json              Rutas y configuración de preparación
│   │   ├── sam6d_ui/                Servidor, motores, cámara e interfaz web
│   │   │   └── README.md            Manual completo y parámetros
│   │   ├── prepared/                Datos derivados, manifests y suites históricas
│   │   ├── native/                  Extensiones compiladas de PointNet2 y FoundationPose
│   │   └── tests/                   Pruebas de integración
│   ├── results/                     Resultados y documentos generados
│   │   ├── dashboard/               Informe estático de la suite guardada
│   │   ├── interface/
│   │   │   ├── jobs/sam6d/          Detecciones y poses PEM, por trabajo
│   │   │   ├── jobs/foundationpose/ Poses y seguimiento, por trabajo
│   │   │   ├── cache/cad/          Referencias reutilizables de CAD nuevos
│   │   │   ├── logs/               Logs del servidor cuando se redirigen aquí
│   │   │   └── validation/         Evidencias de verificaciones guardadas
│   │   └── verification/           Evidencias de pruebas anteriores
│   ├── captures/                    Capturas de cámara, separadas de inferencias
│   │   ├── images/                 Pares RGB-D, calibración y metadatos por captura
│   │   └── videos/                 Vídeos y pares RGB-D por grabación
│   └── maintenance/                Historial y copias de archivos
├── FoundationPose/                  Copia institucional; modelo externo a la interfaz
├── PDDL/                            Planificación y datos originales del ensamblaje
│   ├── Assembly CAD/Assembly parts/ CAD originales
│   └── DINOPDDL/realsense_photos/   Capturas originales
├── ASAP/                            Planificación/simulación y sus herramientas
├── .venvs/                          Entornos Python de ejecución
└── .runtime/                        Bibliotecas auxiliares y cachés
```

`SAM6D/SAM-6D/SAM-6D/` tiene dos niveles con el mismo nombre porque uno es la raíz
del repositorio descargado y otro su carpeta de código. `pipeline/prepared/` contiene
entradas derivadas y ejecuciones históricas; `results/interface/jobs/` contiene los
nuevos trabajos de la web. Las suites se separan de las ejecuciones normales en
`pipeline/prepared/sam6d/suites/`. Se conservan los trabajos web antiguos en
`pipeline/prepared/sam6d_ui/jobs/` para poder consultarlos.

### Qué significa cada tipo de documento o archivo

| Tipo / ejemplos | Contenido y relación con el resto |
| --- | --- |
| Markdown: `README.md`, `USO.md` | Explicaciones e instrucciones escritas para las personas. No son resultados de inferencia. |
| Configuración: `pipeline/config.json`, catálogo YAML de PDDL | Indica fuentes de datos y catálogo. Editar una configuración no recalcula resultados ya guardados. |
| Manifests: `prepared/manifest.json`, `manifest_at_execution.json` | El primero describe las rutas actuales; el segundo conserva el manifest histórico para trazabilidad de ejecuciones. |
| Entradas por trabajo: `inputs.json`, `input_rgb.png`, `input_cad.*`, `depth.png`, `camera.json` | Copia y descripción de lo enviado, opciones efectivas y calibración del análisis. |
| Resultados JSON: `result.json`, `detections.json`, `pose.json` | Resumen del trabajo, detecciones y pose. Las máscaras se guardan como PNG y/o RLE; sus identificadores corresponden al trabajo. |
| JSONL: `frames.jsonl`, `pose_tracking.jsonl` | Una fila JSON por fotograma: índice y tiempos de adquisición o transformación de pose. Permite relacionar cada pose con su fotograma. |
| CSV | Tablas exportadas de scores y tiempos para análisis en hojas de cálculo. No son parámetros del modelo. |
| HTML: `results/dashboard/index.html` | Presentación estática generada de resultados de suite. La interfaz web, en cambio, se sirve desde `pipeline/sam6d_ui/static/`. |
| PNG/JPG y NPY | Imágenes de entrada, máscaras, proyecciones y mapas raw. La profundidad uint16 contiene medidas; una vista coloreada de profundidad es solo una visualización. |
| MP4: `video.mp4`, `pose_tracking.mp4` | Color grabado o visualización del seguimiento. La profundidad se conserva aparte en los PNG por fotograma. |
| Logs: `error.log`, `pose.log`, `execution.log` | Diagnóstico de fallos, salida del estimador y registro de suite, respectivamente. La salida del servidor aparece en su terminal o en el archivo al que se redirija. |
| ZIP | Paquete de informe con archivos asociados para traslado y revisión. No incluye los pesos de los modelos. |


## Qué hace cada componente

El flujo completo es:

```text
CAD → 42 vistas de referencia
                    ↓
Imagen → SAM o FastSAM → máscaras candidatas → comparación con el CAD → detección
                                                                         ↓
                                            profundidad + cámara → PEM o FoundationPose → pose 6D
```

| Componente | Para qué se utiliza |
| --- | --- |
| **SAM ViT-H** | Genera regiones candidatas y sus máscaras. |
| **FastSAM-x** | Alternativa a SAM para generar las máscaras. Se elige uno de los dos. |
| **DINOv2 ViT-L/14** | Compara el aspecto de cada región con las vistas del CAD; forma parte de SAM-6D y se usa con ambos segmentadores. |
| **SAM-6D ISM** | Puntúa las coincidencias y conserva detecciones. Con RGB-D también evalúa coherencia geométrica. |
| **SAM-6D PEM** | A partir de las detecciones y del RGB-D, estima rotación y traslación. |
| **FoundationPose** | Estimador alternativo de pose y seguimiento; recibe la máscara inicial de SAM/FastSAM desde nuestra integración. |
| **Blender** | Prepara las vistas, máscaras y coordenadas XYZ del CAD. No necesita abrir una ventana. |

SAM ViT-H, FastSAM-x, DINOv2 y PEM están preparados en el entorno del proyecto.
La primera ejecución carga los modelos; un CAD nuevo necesita además preparar
sus referencias, que se guardan para reutilizarlas.

## 1. Abrir la interfaz para una imagen y un CAD

Ejecutar en la terminal de **disassembly-0**:

```bash
cd /home/raca/projects
bash SAM6D/run.sh ui --port 8770
```

Abrir **http://127.0.0.1:8770/** en un navegador ejecutado en ese Linux. Mantener
la terminal abierta; **Ctrl+C** detiene el servidor.

Si el navegador está en otro ordenador, reenviar el puerto **8770** desde la
pestaña **Puertos** de VS Code Remote y abrir esa misma dirección allí.

### Seleccionar archivos del servidor

1. Dejar **«Archivos de disassembly-0»** en **Ubicación de los archivos**.
2. Pulsar **«Explorar servidor»** junto a **Imagen RGB** y **Modelo CAD**.
3. Usar los accesos **«Fotos y profundidad»** y **«CAD del ensamblaje»**.
4. Elegir el segmentador **SAM ViT-H** o **FastSAM-x** y las unidades del CAD.
5. Pulsar **«Detectar objeto»**.

El explorador muestra archivos de `/home/raca/projects` en el servidor. También
se pueden pegar rutas de Linux en los campos. La opción **«Subir archivos desde
el navegador»** sirve para enviar archivos del ordenador donde está el navegador.

### Elegir detección 2D o pose 6D

| Modo | Archivos necesarios | Resultado |
| --- | --- | --- |
| **Solo RGB** | Imagen + CAD | Detección y máscara mediante similitud y apariencia. Es una adaptación 2D que omite la evaluación geométrica. |
| **RGB-D** | Imagen + CAD + profundidad alineada + calibración | Detección y máscara con el score geométrico de SAM-6D. |
| **RGB-D + «Estimar pose 6D con PEM»** | Los cuatro archivos anteriores y CUDA | Añade rotación, traslación y una proyección del CAD sobre la imagen. |

Al elegir **RGB-D** aparecen los campos **Profundidad alineada** y
**Calibración de cámara**. Por ejemplo, para una captura del control board:

| Campo | Archivo de `PDDL/DINOPDDL/realsense_photos/` |
| --- | --- |
| Imagen RGB | `control_board.jpg` |
| Profundidad | `control_board_depth.npy` |
| Calibración | `control_board_depth.json` |

Seleccionar siempre los archivos de la **misma captura**. La profundidad debe
ser raw `uint16`, estar alineada al color y tener su misma resolución. Una imagen
de profundidad coloreada no sustituye al mapa raw. Los CAD admitidos son
**STL, OBJ y PLY**; los STEP deben exportarse a malla.

### Ejecutar directamente con Python

La interfaz también se puede arrancar sin el lanzador Bash:

```bash
cd /home/raca/projects
LD_LIBRARY_PATH=/home/raca/anaconda3/envs/dinobot_env/lib \
.venvs/assembly-runtime/bin/python SAM6D/pipeline/sam6d_gui.py --port 8770
```

`run.sh` prepara ese entorno y ejecuta Python. No requiere activar conda y evita
usar por accidente las dependencias de otro entorno, como `dinopddl`.
En ambos casos se abre una **interfaz web**; el programa no crea una ventana de
escritorio independiente.

### Referencia de todos los parámetros

La [guía de parámetros de la interfaz](pipeline/sam6d_ui/README.md#parámetros-de-la-interfaz)
explica cada campo, sus valores por defecto, rangos y el efecto de cambiarlo:
entradas, modelos, umbrales SAM/FastSAM, agregación, cámara, grabación, calibración,
resultados, opciones de arranque y variables de entorno.

## 2. Ver la suite que ya se ejecutó con todas las imágenes

La suite guardada compara **22 capturas RGB-D con los 10 CAD**. Se ejecutó con
**SAM ViT-H en GPU** y evalúa detección/segmentación; **no incluye pose PEM**.

Desde `/home/raca/projects`:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory /home/raca/projects/SAM6D/results/dashboard
```

Abrir la dirección completa:

**http://127.0.0.1:8766/index.html**

Si el navegador está en otro ordenador, reenviar también el puerto **8766**.
El informe funciona en cualquier sistema con navegador. También se abre directamente
desde **http://127.0.0.1:8766/** gracias al nombre neutral `index.html`.

El informe permite revisar imágenes, segmentaciones, scores por CAD y tiempos.
La pieza elegida aparece en verde y las alternativas con score mayor de 0,40,
en naranja. Ese color facilita la comparación visual; no cambia los resultados
originales ni sustituye al umbral de aceptación.

## 3. Ejecutar desde terminal

Los comandos siguientes se ejecutan desde `/home/raca/projects`.

```bash
# Una captura y un CAD conocidos: SAM ViT-H + pose 6D.
bash SAM6D/run.sh detect --example control_board --part 05_control_board --segmentor sam --pose

# La misma prueba utilizando FastSAM-x + pose 6D.
bash SAM6D/run.sh detect --example control_board --part 05_control_board --segmentor fastsam --pose
```

Sin `--pose`, estos ejemplos ejecutan detección RGB-D. Para detección sin
profundidad, añadir `--mode rgb` y omitir `--pose`.
Los resultados se guardan en `SAM6D/results/runs/<fecha>/`.

Para una **nueva suite** de todas las capturas frente a todos los CAD:

```bash
bash SAM6D/run.sh suite run --device cuda --segmentor sam --export-dir SAM6D/results/suite_sam
bash SAM6D/run.sh suite run --device cuda --segmentor fastsam --export-dir SAM6D/results/suite_fastsam
```

La suite calcula detecciones y tiempos; no ejecuta PEM. Los registros de cada
nueva ejecución se guardan por separado. `--export-dir` indica dónde generar su
informe y ZIP; usar otro destino para conservar una exportación anterior.

La [guía completa USO.md](USO.md) explica cómo utilizar archivos propios,
segmentar todo el catálogo, calcular pose sobre máscaras guardadas, regenerar
plantillas y configurar los umbrales.

## 4. Interpretar los resultados

- **Máscara**: píxeles que el modelo asigna a una región candidata.
- **Score ISM**: medida de coincidencia con el CAD. No es una probabilidad ni una precisión medida. Los scores RGB y RGB-D emplean fórmulas diferentes.
- **Umbrales iniciales de la interfaz**: se conservan candidatos desde **0,30** y se aceptan/dibujan desde **0,50**. Se pueden cambiar en Configuración avanzada.
- **Sin detección conservada**: ninguna detección de esa clase sobrevivió a la asignación y al filtrado. No significa que el score del CAD sea cero ni que se haya omitido su comparación.
- **Pose 6D**: rotación `R` y traslación `t_mm` respecto a la cámara. No es una pose respecto a la base del robot; para eso hace falta calibración extrínseca.
- **Tiempos**: se separan la inferencia, la carga de modelos, la preparación de referencias y PEM. La primera ejecución puede tardar más por esas etapas adicionales.

En la interfaz se pueden descargar resultados JSON, máscaras y visualizaciones.
Cuando se estima pose, aparecen `pose.json` y `pose_overlay.png`. La proyección
permite revisar el ajuste; todavía no se ha medido el error frente a una pose
real conocida. Comparar una imagen con un único CAD tampoco demuestra que sea
la única clase posible.

## 5. Dónde están los datos y el código

| Ruta | Contenido |
| --- | --- |
| [CAD originales](<../PDDL/Assembly CAD/Assembly parts/>) | Piezas originales del ensamblaje. |
| [Capturas originales](../PDDL/DINOPDDL/realsense_photos/) | Fotografías, profundidad y metadatos de cámara. |
| [SAM-6D/](SAM-6D/) | Repositorio del modelo y adaptaciones locales. |
| [pipeline/](pipeline/) | Interfaz, preparación de datos, evaluación y lanzadores. |
| [pipeline/prepared/](pipeline/prepared/) | Manifest, plantillas y datos derivados para los modelos. |
| [results/dashboard/](results/dashboard/) | Informe de la suite original de 22 capturas. |
| [results/verification/](results/verification/) | Pruebas realizadas de SAM/FastSAM con PEM. |
| `results/runs/` | Ejecuciones nuevas realizadas con `detect`. |
| `results/interface/jobs/{sam6d,foundationpose}/` | Nuevos trabajos, resultados y logs de la interfaz. |
| `captures/` | Capturas RGB-D en `images/` y grabaciones en `videos/`. |
| [pipeline/prepared/sam6d_ui/jobs/](pipeline/prepared/sam6d_ui/jobs/) | Trabajos antiguos de la interfaz, conservados y accesibles. |
| [maintenance/](maintenance/) | Auditorías de organización y deduplicación. |

Los datos originales se mantienen en **PDDL**. Las mallas convertidas, las vistas
CAD y las máscaras son datos derivados necesarios. La planificación del
ensamblaje y el generador de catálogo pertenecen a [ASAP](../ASAP/LAYOUT.md).

## 6. Problemas frecuentes

### `OSError: [Errno 98] Address already in use`

El puerto solicitado ya está ocupado. Si el servicio correcto está iniciado,
abrir su URL sin arrancarlo otra vez. Para otra instancia de la interfaz:

```bash
bash SAM6D/run.sh ui --port 8771
```

Abrir entonces `http://127.0.0.1:8771/`. Para el informe estático se puede cambiar
`8766` por otro puerto libre, tanto en el comando como en la URL. La interfaz
interactiva y el informe estático son servicios distintos.

### No encuentro la calibración o selecciono archivos del PC

Elegir **RGB-D** para mostrar el campo de calibración. Seleccionar
**«Archivos de disassembly-0»** para explorar el disco del servidor; el selector
de subida del navegador muestra los archivos del ordenador del navegador.

### Faltan pesos después de reiniciar Linux

Los paquetes y la extensión CUDA están instalados en disco. Por el espacio
limitado, los checkpoints se guardan actualmente en
`/dev/shm/pddl-assembly-runtime/checkpoints`: permanecen al cerrar SSH, pero
desaparecen al reiniciar el equipo. Para restaurarlos y comprobar la instalación:

```bash
bash SAM6D/run.sh install
bash SAM6D/run.sh sam doctor
```

Con el entorno preparado y acceso a la GPU, `doctor` debe mostrar
`cuda_available`, `segmentation_ready`, `fastsam_ready` y `pose_ready` a `true`.

Más detalles y fuentes oficiales: [USO.md](USO.md).


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
`SAM6D/captures/`. Las suites conservan sus carpetas.

Consulta [la guía de la interfaz](pipeline/sam6d_ui/README.md) para arquitectura,
instalación, unidades, cámaras, seguimiento y estructura de archivos.


### FoundationPose con selección manual

La interfaz ofrece **Inicialización de FoundationPose → Caja manual · solo FoundationPose**.
Permite congelar un par RGB-D de RealSense, dibujar una caja y estimar la pose sin
ejecutar SAM/FastSAM, DINOv2 ni renderizar vistas del CAD. La caja se usa como máscara
rectangular aproximada, por lo que debe incluir poco fondo. También puede inicializar
el seguimiento rápido. Véanse los pasos en el
[manual de interfaz](pipeline/sam6d_ui/README.md#foundationpose-por-separado-con-caja-manual).
