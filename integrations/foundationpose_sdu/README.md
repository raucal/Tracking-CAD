# SDU FoundationPose integration

This directory preserves SDU additions separately from the pinned official FoundationPose submodule. Revisions, inventory, hash validation, attribution, and licensing are documented in [Provenance](../../docs/PROVENANCE.md#sdu-compatibility). Use [Installation](../../docs/INSTALLATION.md) to set up CADTracking; `files/README.md` is a preserved upstream template.

CADTracking uses the official algorithm with SDU's dtype/device compatibility change to `Utils.py` loaded in memory. The cached derivative is regenerated automatically; do not edit the official source. Restart an existing UI process after changing the integration so new workers load it.

## Run a preserved SDU tool

From the project root, using the configured FoundationPose Python:

```bash
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py grasp_config_ui.py
```

Supported scripts are `grasp_config_ui.py`, `run_video_demo.py`, and `run_single_pose_robot_demo.py`. They retain SDU's original options and dependencies. Arguments after the script name, including `--help`, are passed to that script.

## Check imports without running the main block

Place `--check` before the script name:

```bash
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py --check run_video_demo.py
../.venvs/foundationpose/bin/python integrations/foundationpose_sdu/run.py --check --teleop-root ../PDDL/DINOBOT/alignment_and_replay-main run_single_pose_robot_demo.py
```

`--check` loads imports without opening the camera or connecting to the robot. The robot demo requires the external `teleop_setup` package. Point `--teleop-root` to its parent directory or set `SDU_TELEOP_ROOT`; the package is not copied into CADTracking.

**Without `--check`, the original robot demo connects and sends a movement at startup.** For the video demo, supply `--mesh_file` and `--mesh_scale` if its original example data is unavailable.

Recorded checks used the control-panel CAD: a grasp UI in a virtual display, video-model loading, and RealSense RGB-D at 640×480. The robot demo was checked only through `--check`, without connection or motion. These are prior checks, not new hardware tests performed during documentation editing.
