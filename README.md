# CAD Tracking

Detect a CAD part in an image or Intel RealSense stream, estimate its pose with PEM or FoundationPose, and track it through a browser interface.

| Guide | Purpose |
| --- | --- |
| [Installation](docs/INSTALLATION.md) | Set up another machine, install dependencies and weights, and prepare data. |
| [Reference](docs/REFERENCE.md) | Commands, configuration, camera workflows, suites, results, and troubleshooting. |
| [Provenance](docs/PROVENANCE.md) | Model sources, pinned revisions, project adaptations, and attribution. |

## Start the application

On the configured server, run these commands from the project directory:

```bash
bash run.sh ui
```

Open **http://127.0.0.1:8770**, keep the terminal open, and wait for model loading. The launcher selects the configured Python environments; manual environment activation is unnecessary for normal use. Stop the service with **Ctrl+C**.

For a new machine, follow [Installation](docs/INSTALLATION.md) first.

The [dependency list](docs/INSTALLATION.md#dependencies) covers system software and both Python environments. A clone does not include weights or prepared data: use the [asset handoff checklist](docs/INSTALLATION.md#assets-required-after-cloning) before running the application. Complete exported reports can be viewed independently of those assets.

### Remote access

Run the application on the server and forward port **8770** through VS Code Remote SSH, or create a tunnel from your computer:

```bash
ssh -N -L 8770:127.0.0.1:8770 user@server
```

Then open the same local URL. The web interface does not require X11. The RealSense must be connected to the server running the application.

## Try an image or the camera

The interface currently uses Spanish labels.

**Without a camera:** expand **Probar con los datos del ensamblaje**, select a capture and CAD, and click **Ejecutar ejemplo**. For pose estimation, select RGB-D and enable PEM, or choose FoundationPose.

**With RealSense:** select **Intel RealSense en directo**, click **Abrir cámara**, select a CAD and its units (usually mm), choose FoundationPose, and click **Localizar e iniciar seguimiento rápido**.

Fast tracking updates the pose between SAM checks. Automatic initialization enables recovery after tracking loss and a periodic check every 10 seconds by default. **Iniciar detección en directo** instead repeats full analyses. See the [camera and tracking reference](docs/REFERENCE.md#camera-and-tracking) for manual initialization, automatic part identification, recording, and the CAD viewer.

## Run a suite and open its report

Stop the interface before benchmarking on the same GPU. From the project directory:

```bash
# One capture, including PEM pose estimation
bash run.sh suite run --frame hmi_panel

# All captures in the configured profile
bash run.sh suite run
```

Each command creates a separate result directory. The first run can take longer while models and references load. To open the latest exported report:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory results/dashboard
```

Open **http://127.0.0.1:8766** and use **Matrices y triedros** to inspect poses. Forward port **8766** for remote access. See [Suites](docs/REFERENCE.md#suites) for detection-only runs, saved results, ZIP exports, and per-view audits.

## Configuration and saved files

| Location | Purpose |
| --- | --- |
| `config/settings.json` | Python environments, models, Blender, data, weights, and cache paths. |
| `config/interface.json` | Server, inference, camera, and tracking defaults. |
| `config/suite.json` | Suite selection, thresholds, pose estimation, and output paths. |
| `captures/` | Saved camera images and recordings. |
| `results/suites/` | Original suite results. |
| `results/reports/` | Exported reports; `results/dashboard` links to the latest export. |
| `results/interface/`, `results/runs/` | Local interface sessions and individual analyses, ignored by Git. |

**Keep `data/`, `checkpoints/`, and the configured environments.** Prepared data and catalog templates are not rebuilt when the application starts. See [Storage and cleanup](docs/REFERENCE.md#storage-and-cleanup) for what can be removed, and [Troubleshooting](docs/REFERENCE.md#troubleshooting) if startup fails.
