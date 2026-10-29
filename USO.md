# Ejecutar SAM-6D en este equipo

Todos los comandos siguientes parten de `/home/raca/projects`. No hace falta
activar conda: `run.sh` usa `.venvs/assembly-runtime` y configura las bibliotecas.
Los CAD originales siguen en `PDDL/Assembly CAD/Assembly parts/`; las capturas
originales, en `PDDL/DINOPDDL/realsense_photos/`.

## Componentes instalados

| Componente | Función | Estado |
| --- | --- | --- |
| SAM ViT-H | Propone máscaras | Instalado, checkpoint oficial |
| FastSAM-x | Generador alternativo de máscaras | Instalado, Ultralytics 8.0.135 |
| DINOv2 ViT-L/14 | Compara las regiones con las vistas CAD | Instalado; se usa con ambos segmentadores |
| SAM-6D PEM base | Estima rotación y traslación tras segmentar | Instalado, checkpoint oficial y PointNet2 compilado para RTX 4090 |
| Blender 3.3.1 | Genera 42 vistas RGB, máscaras y XYZ del CAD | Instalado, sin ventana gráfica |

SAM y FastSAM son alternativas para la primera etapa. Ambos pueden continuar
con el mismo estimador PEM. ViT-L y ViT-B no se han instalado.

## 1. Interfaz sencilla

```bash
cd /home/raca/projects
bash SAM6D/run.sh ui --port 8770
```

En VS Code Remote, abrir **Puertos**, reenviar **8770** y abrir
**http://localhost:8770/** desde Windows. Para SSH directo:

```bash
ssh -N -L 8770:127.0.0.1:8770 raca@disassembly-0.sandbox.tek.sdu.dk
```

1. Dejar **Archivos de disassembly-0**, la opción predeterminada. Pulsar
   **Explorar servidor** para imagen y CAD. El explorador tiene accesos directos
   a `PDDL/DINOPDDL/realsense_photos/` y `PDDL/Assembly CAD/Assembly parts/`.
   También se pueden pegar rutas absolutas de Linux. No se suben archivos del PC.
   Se admite JPG/PNG y CAD STL/OBJ/PLY; un STEP debe exportarse a malla.
   Para cargar archivos del ordenador del navegador, cambiar a **Subir archivos
   desde el navegador**.
2. Indicar unidades del CAD: mm, cm o m.
3. Elegir **SAM ViT-H** o **FastSAM-x**.
4. Para detección 2D basta **Solo RGB**. Es una adaptación que omite geometría.
5. Para SAM-6D con geometría, elegir **RGB-D** y añadir profundidad raw uint16
   alineada y calibración JSON. Se admite `.npy` o PNG de 16 bits, sin colorear.
6. Para obtener rotación y posición, marcar **Estimar pose 6D con PEM**. Esto
   selecciona RGB-D y CUDA; sigue siendo necesario cargar profundidad y cámara.
7. Pulsar **Detectar objeto**. Aparecen máscaras, scores, tiempos por etapa,
   dispositivo real y, si se estima pose, proyección del CAD y transformación.

El explorador permite navegar dentro de `/home/raca/projects` y muestra solo
los formatos correspondientes a cada entrada. Para habilitar otra carpeta del
servidor, iniciar con `--file-root /ruta/a/la/carpeta`. No se muestran archivos
ocultos ni se siguen enlaces que salgan de la raíz elegida.

La sección **Probar con los datos del ensamblaje** permite elegir una de las 22
capturas y un CAD sin subir los archivos. El nombre de la fotografía no decide
la predicción. Las máscaras se generan automáticamente, sin clics manuales.

Las detecciones aceptadas se muestran desde 0,50. Se conservan candidatos desde
0,30; los scores inferiores descartados no se presentan como cero. Los scores
son medidas de coincidencia, no probabilidades ni precisión medida.

## 2. Probar las cadenas completas por terminal

SAM ViT-H + PEM sobre el control board:

```bash
bash SAM6D/run.sh detect --example control_board --part 05_control_board --segmentor sam --pose
```

FastSAM-x + PEM sobre el mismo objeto:

```bash
bash SAM6D/run.sh detect --example control_board --part 05_control_board --segmentor fastsam --pose
```

Sin `--pose` se ejecuta únicamente la detección/segmentación. Un ejemplo del
ensamblaje usa RGB-D por defecto. Para probar la adaptación sin profundidad:

```bash
bash SAM6D/run.sh detect --example control_board --part 05_control_board --segmentor fastsam --mode rgb
```

Las salidas se crean en `SAM6D/results/runs/<fecha>/`. `--output /ruta/nueva`
permite elegir otra carpeta; se rechaza una carpeta existente.

## 3. Tus propios archivos

Detección 2D sin profundidad:

```bash
bash SAM6D/run.sh detect --rgb /ruta/foto.png --cad /ruta/pieza.stl --cad-units mm --segmentor sam
```

Detección RGB-D y pose 6D:

```bash
bash SAM6D/run.sh detect \
  --rgb /ruta/foto.png \
  --depth /ruta/profundidad.npy \
  --camera /ruta/camara.json \
  --cad /ruta/pieza.stl \
  --cad-units mm --segmentor fastsam --pose
```

RGB y profundidad deben tener la misma resolución y estar alineados. El JSON
puede ser el formato RealSense de nuestras capturas o BOP/SAM-6D:

```json
{"cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1], "depth_scale": 1.0}
```

Aquí `fx`, `fy`, `cx`, `cy` se sustituyen por números de la calibración real.
`depth_scale` es **milímetros por unidad raw**, no metros. No se inventa una
calibración ni se estima pose desde una fotografía RGB aislada.

Para un CAD nuevo se generan 42 vistas y XYZ métricos en Blender; después se
reutilizan. La primera ejecución también incluye la carga de los modelos.
Renderizar referencias usa CPU; SAM/FastSAM, DINO y PEM usan CUDA al seleccionarla.

## 4. Interpretar y descargar la pose

- `result.json`: configuración, decisión, scores y tiempos de detección y pose.
- `detections.json` y `mask_*.png`: máscaras y detecciones conservadas.
- `overlay.png`: segmentación aceptada.
- `pose.json`: rotaciones, traslaciones y transformaciones de pose.
- `pose_overlay.png`: captura original junto al CAD proyectado con la pose estimada.
- `pose.log` y, cuando corresponde, `render.log`: detalle de esas etapas.

`R` es una matriz 3×3; `t_mm` es una traslación de tres componentes en mm.
`T_camera_from_model_mm` transforma el CAD centrado usado por el modelo a la
cámara. `T_camera_from_input_cad_mm` transforma el CAD de entrada conservando su
origen, con sus coordenadas expresadas en mm. Para unidades cm/m se convierte
primero a mm. Se usa la convención de cámara OpenCV: X derecha, Y abajo, Z delante.
La pose no está expresada respecto a la base del robot: eso requiere calibración
extrínseca. La proyección sirve para revisar la estimación; no tenemos una pose
real etiquetada con la que medir su error.

## 5. Comparar todas las piezas con todas las imágenes

Cada comando siguiente crea una suite nueva; no sobrescribe los resultados guardados.
Se comparan los diez CAD con todas las propuestas válidas de cada captura.

```bash
bash SAM6D/run.sh suite run --device cuda --segmentor sam --export-dir SAM6D/results/suite_sam
bash SAM6D/run.sh suite run --device cuda --segmentor fastsam --export-dir SAM6D/results/suite_fastsam
```

La suite evalúa **detección**, con tiempos y scores por imagen; no ejecuta PEM.
El refinamiento ISM se hace sobre el CAD asignado a cada región. Para inspeccionar
hipótesis de todos los CAD sobre una misma región primaria hay un diagnóstico
adicional, que no equivale a diez detecciones:

```bash
bash SAM6D/run.sh scores --run SAM6D/pipeline/prepared/sam6d/suites/gpu_all_22 --device cuda
```

Para regenerar el informe histórico sin volver a inferir:

```bash
bash SAM6D/run.sh suite report \
  --output SAM6D/pipeline/prepared/sam6d/suites/gpu_all_22 \
  --export-dir SAM6D/results/dashboard
```

El exportador crea un informe y un ZIP portable bajo petición. Para abrir el
informe existente por SSH sin X11:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory SAM6D/results/dashboard
```

Reenviar 8765 y abrir `http://localhost:8765/index.html`.

## 6. Comandos de catálogo y plantillas

Para ejecutar la segmentación conjunta del catálogo y después estimar pose para
las clases detectadas:

```bash
bash SAM6D/run.sh sam segment --frame control_board --segmentor fastsam --min-score 0.5 --output SAM6D/results/catalog_test
bash SAM6D/run.sh sam pose --frame control_board --min-score 0.5 --output SAM6D/results/catalog_test
```

`--part 05_control_board` limita ambos comandos a esa pieza; omitirlo compara el
catálogo completo. `--frame` se puede repetir; omitirlo selecciona las 22 capturas.
Se debe usar la misma selección de piezas, capturas y salida para las dos etapas.
PEM utiliza las máscaras guardadas por `segment`.

Regenerar las 42 plantillas métricas de una pieza conocida, en su ubicación de
referencia, mediante Blender sin ventana:

```bash
bash SAM6D/run.sh sam render --part 05_control_board
```

La interfaz y `detect` generan las referencias de CAD nuevos automáticamente.
No se necesita BlenderProc en este entorno.

## 7. Configuración

| Opción de interfaz | Inicial | Significado |
| --- | --- | --- |
| Segmentador | SAM ViT-H | SAM o FastSAM-x |
| Modo | RGB | RGB, RGB-D; PEM exige RGB-D |
| Dispositivo | CUDA | `cuda`, `auto` o `cpu`; PEM exige CUDA |
| Score mínimo | 0,30 | Candidatos conservados tras ISM |
| Score aceptado | 0,50 | Detecciones dibujadas y enviadas a PEM |
| Estabilidad SAM | 0,97 | Filtro de estabilidad de máscaras SAM |
| IoU predicha SAM | 0,88 | Filtro interno de calidad de máscara |
| Puntos SAM por lote | 64 | Memoria y procesamiento de puntos |
| Confianza FastSAM | 0,05 | Filtro de propuestas FastSAM, distinto al score ISM |
| IoU FastSAM | 0,90 | NMS interno de propuestas FastSAM |
| Máximo FastSAM | 200 | Límite de propuestas |
| NMS ISM | 0,25 | Filtra detecciones finales solapadas de la misma clase |
| Agregación CAD | `avg_5` | Media de las cinco mejores vistas; también media, mediana o máximo |

Los parámetros específicos de SAM no se aplican a FastSAM y viceversa.
DINOv2 ViT-L/14 y ancho de análisis 640 se comparten entre ambos.
El umbral semántico previo al refinamiento es 0,20.

## 8. Comprobar y restaurar la instalación

```bash
bash SAM6D/run.sh sam doctor
```

Debe indicar `cuda_available`, `segmentation_ready`, `fastsam_ready` y
`pose_ready` a `true`.

Los paquetes y la extensión compilada persisten en disco. Los checkpoints,
por el poco espacio disponible en disco, están actualmente en
`/dev/shm/pddl-assembly-runtime/checkpoints` y se enlazan desde el repositorio.
Son una única copia compartida y desaparecen al reiniciar el equipo.
Para restaurar SAM ViT-H, DINOv2, FastSAM-x y PEM:

```bash
bash SAM6D/run.sh install
```

Este comando reutiliza los pesos existentes, descarga los que faltan, instala
las dependencias fijadas y compila PointNet2 si no está disponible. La extensión
actual corresponde a Python 3.10, PyTorch 2.5.1+cu124, toolkit CUDA 12.5 y RTX 4090.
No es necesario reinstalar en cada sesión SSH.

## Pruebas realizadas

Se han ejecutado realmente con CUDA las cadenas FastSAM + PEM sobre un CAD
conocido y SAM ViT-H + PEM sobre el mismo objeto exportado a un formato distinto,
para forzar la preparación de un CAD nuevo con XYZ. Las salidas están en
`SAM6D/results/verification/`. Los tiempos incluyen las etapas que figuran en
cada JSON; no son una comparación de velocidad controlada entre modelos.

Fuentes oficiales: [SAM-6D](https://github.com/JiehongLin/SAM-6D),
[instalación PEM](https://github.com/JiehongLin/SAM-6D/blob/main/SAM-6D/Pose_Estimation_Model/dependencies.sh),
[SAM](https://github.com/facebookresearch/segment-anything),
[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).


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
`SAM6D/captures/` (`images/` para capturas y `videos/` para grabaciones). Las suites conservan sus carpetas.

Consulta [la guía de la interfaz](pipeline/sam6d_ui/README.md) para arquitectura,
instalación, unidades, cámaras, seguimiento y estructura de archivos.
