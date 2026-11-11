# Align the robot TCP to a CAD plane

This extension adds a robot panel to the existing CADTracking browser interface. All robot configuration, geometry, RTDE control, UI code, and dedicated tests live here. The existing server, HTML, and browser request builder contain small integration hooks.

The requested target is the **TCP/tool**, **0.20 m perpendicular to the CAD plane**, on its outward-normal side. The tool's positive Z axis points toward the plane; X follows the plane's X direction with the configured yaw. This is a single capture, preview, and linear movement, not continuous visual servoing or replay of a saved demonstration.

## Configure

Edit [settings.json](settings.json). It is the only settings file for this feature. No connection or movement happens when the application imports this module or starts.

| Section | Settings |
| --- | --- |
| `robot` | Controller IP/frequency, expected active TCP offset, camera pose relative to that TCP, camera serial, calibration confirmation, and motion enable. |
| `alignment` | Normal and X axis expressed in the original CAD frame, plane origin, 0.20 m offset, optional in-plane offsets and tool yaw, and per-CAD overrides. |
| `motion` | Speed, acceleration, stop deceleration, workspace box, maximum Cartesian/angular/joint changes, sampled IK checks, expiry, drift, standstill, and endpoint tolerances. |

The initial left-robot IP (`192.168.1.11`), TCP (`[0,0,0.15,0,0,0]`), and camera transform (`[0,0.115,-0.13,0.78,0,0]`) were taken from the existing DINOBOT demo setup. They are **reference settings, not a newly measured or verified hand–eye calibration**. Verify them against the actual tool and camera mount. `tcp_from_camera_m_rad` maps camera coordinates into TCP coordinates and must correspond to the configured active TCP. The module checks the controller's TCP without changing it.

Set `camera_serial` to the calibrated RealSense device's serial. After verifying the calibration and operating limits, set `calibration_confirmed` and `motion_enabled` to `true` to enable the panel's execution button. These start as `false`; preview calculation still works with them disabled. Reconnect after changing settings. Editing settings invalidates existing plans.

A default normal of original CAD **+Y** and plane X of **+X** is appropriate to the intended frontal XZ plane of the supplied printed-assembly CADs, but must be checked for each model. With `plane_origin_cad_mm: null`, the plane origin is the center of the bounding-box face on the normal side. This is a geometrical convention, not automatic detection of a physical planar surface. For another plane, set its point and axes explicitly, or add `alignment.cad_overrides` keyed by the CAD key. Oblique planes require an explicit origin. The normal must face the observing camera.

Example override (replace the values with the intended CAD plane):

```json
{
  "cad_overrides": {
    "uploaded_cad": {
      "plane_normal_cad": [0, 0, -1],
      "plane_x_cad": [1, 0, 0],
      "plane_origin_cad_mm": [0, 0, 0]
    }
  }
}
```

Distances are meters in robot settings, outputs, and control; the original CAD plane point is in millimeters. The selected upload units are honored when converting a fixed CAD; prepared assembly identification uses millimeters. Robot pose angles are rotation vectors in radians, never Euler angles. `tool_yaw_deg` is an explicit angle in degrees around the plane normal. Its initial value is 180° for the supplied assembly convention, avoiding the opposite in-plane heading; verify this choice for your tool. Limits initially allow at most 90° of Cartesian rotation and 1.5 rad change per joint; plans exceeding a limit are rejected rather than automatically changing it.

## Dependencies and startup

Use the configured SAM6D/UI environment after the main [installation](../docs/INSTALLATION.md):

```bash
../.venvs/assembly-runtime/bin/python -m pip install --constraint pipeline/setup/constraints-assembly-runtime.txt -r alignment_pose_cad/requirements.txt
bash run.sh ui
```

The adapter uses `ur-rtde` 1.6.5. The robot controller must be configured for the connection and execution mode supported by that installation. Opening RTDE control may load its control script; the extension does not power on the robot, unlock protective stops, reset safety, change TCP calibration, operate the gripper, or start any motion on connection.

## Workflow

1. In CADTracking select **Live Intel RealSense**, open the calibrated camera, and select the CAD or prepared assembly. Stop fast tracking and continuous detection.
2. In **Align robot to CAD**, click **Connect robot** with the robot stationary.
3. Click **Capture and calculate target**. The service brackets a fresh RGB-D capture with robot telemetry, forces SAM + FoundationPose RGB-D inference, and verifies that the TCP stays still during inference.
4. Review the pose image, target TCP pose in the robot base, CAD plane transform, movement distance, and rotation. Failed detection/pose, wrong-facing normal, movement limits, or rejected IK yield an error rather than a destination.
5. Click **Execute displayed target**. The service rechecks settings, serial binding, plan/capture age, standstill, unchanged TCP/joints, and sampled controller safety/IK checks, then sends one asynchronous `moveL`. Plans are single-use.
6. The service reports completion only after checking actual endpoint position/orientation and standstill. Capture again for visual verification. **Stop robot** cancels planning or sends `stopL` for active motion; shutdown also requests a stop before disconnecting.

A timeout, safety fault, disconnection, or cancellation does not automatically retry or resume a movement. Calculate a new plan after resolving it. A new capture is required after the robot moves.

## Transform convention

`T_A_from_B` maps a homogeneous point in frame B into frame A. The visual estimate uses the **original CAD transform**, not the centered-model overlay axes.

```text
T_base_from_cad = T_base_from_tcp_capture
                @ T_tcp_from_camera
                @ T_camera_from_input_cad

T_base_from_tcp_target = T_base_from_cad
                      @ T_cad_from_plane
                      @ T_plane_from_tcp_target
```

SAM/FoundationPose's CAD translation is converted from millimeters to meters once. The plane target translation is `[in_plane_x, in_plane_y, 0.20]` meters. At zero yaw its rotation in the plane is `diag(1, -1, -1)`, making tool Z point toward the plane. Both rotations are right-handed.

## Outputs

Artifacts are saved alongside the normal vision job in `results/interface/jobs/foundationpose/<job-id>/`, which remains excluded from Git:

- `robot_capture.json`: robot TCP/joints, capture ID/timestamp, serial, and settings hash.
- `robot_alignment_plan.json`: full transform chain, source image, target, sampled path, and expiry.
- `robot_alignment_execution.json`: actual endpoint/error information, created only after an execution request.
- `robot_alignment_error.json`: planning failure, if any.

Regular input images, masks, FoundationPose output, and the pose overlay are retained by the existing vision engine. No original CAD or saved demo is modified.

## Limits and validation

This is a direct linear move, **not collision-aware planning**. The workspace box, step limits, and sampled controller checks cannot establish that the arm/tool path is free of fixtures or obstacles. CAD-frame ambiguity, symmetry, a loose mask, or inaccurate depth/hand–eye calibration can produce a geometrically valid but wrong target. The scene and object must remain fixed between capture and execution.

The web stop is a software request dependent on the connection and controller; it does not replace the robot's physical emergency stop. A hanging or lost RTDE connection can prevent that request from reaching the robot. Do not run another application that commands the same robot concurrently. Hardware acceptance remains necessary before enabling motion.

Tests use simulated telemetry/control and camera/inference; they never connect to the robot:

```bash
PYTHONPATH=pipeline ../.venvs/assembly-runtime/bin/python -B -m unittest discover -s alignment_pose_cad/tests -v
bash run.sh test
```

API behavior follows the installed 1.6.5 bindings and the official [ur_rtde quick start](https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/quick_start.html) and [asynchronous movement example](https://sdurobotics.gitlab.io/ur_rtde/pages/examples/basic_motion/move_async_example.html).
