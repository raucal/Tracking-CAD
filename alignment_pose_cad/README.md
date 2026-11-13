# UR5e alignment: detect pose and execute

The interface has two alignment actions: **Detect pose** and **Execute alignment**.
Connect the robot and open the live RealSense camera first. Detect pose captures
fresh RGB-D and runs SAM-6D + FoundationPose with the CAD source selected in
the vision controls. It checks controller IK and joint limits automatically. Review the overlay
and target, then select Execute alignment and accept the final motion confirmation.
The manual review form, supervisor field and separate path-check button have
been removed following the operator's report that the physical path was checked.
This report does not automatically certify collision clearance or calibration accuracy.

## Configuration and target

The selected mode is `execution_mode: supervised_trial`, with
`preview_only: false`, `motion_enabled: true`, and `calibration_confirmed: false`.
The hand-eye calibration is shared with DINOPDDL:
`../../PDDL/DINOPDDL/config/matrix_calibration.yaml` declares the robot (model and
IP), the calibrated RealSense serial, the TCP active during calibration and the
status, and points to `Calibration.xml`, the unmodified output of the calibration
repository. The matrix is read from that XML (`MovingTransform frame="Camera"`);
to recalibrate, replace the XML. Its status is **provisional**, which this
experimental mode accepts; the normal `validated` mode still requires
`status: validated`. Changing the YAML or the XML invalidates any target. Either `preview_only: true` or `motion_enabled: false`
blocks motion in both modes.

## Select different CADs in one session

The alignment panel follows the existing **CAD source** controls:

- **Server file:** choose an assembly part or browse to an STL/OBJ/PLY.
- **Upload from browser:** upload the CAD and select its units.
- **Identify assembly part:** select a prepared assembly. SAM6D compares its
  parts; FoundationPose registers the accepted winner. The alignment uses that
  winner's CAD frame, mesh and bounds, including when it differs from the first
  catalog entry. Unknown winners or a missing accepted detection produce no target.

Keep the RTDE connection open while changing CADs. Stop continuous detection and
tracking before alignment, then select **Detect pose** (or **Identify CAD and
 detect pose** in automatic mode). Each attempt uses a new RGB-D capture and a
stationary contemporaneous TCP. Review the identified CAD and overlay before
**Execute alignment**. Detection never starts motion or switches the target of
an executing move. This is per-capture automatic identification, not continuous
robot following.

Changing the source, selected file, assembly, upload or units disables execution
and invalidates the previous target. A new target does not inherit rotation
approval or execution consent. Execution is bound to the displayed plan ID and
selection signature. CAD file and catalog hashes are rechecked before motion.

## Align to a recorded skill demo

**Alignment target** selects where the TCP goes. *CAD face standoff* is the
original behaviour (next section). *Skill demo* reproduces the TCP pose, relative
to the object, that DINOBOT recorded for that object's skill, so the skill can be
replayed from the aligned pose wherever the object now is.

Each demo in `PDDL/DINOBOT/alignment_and_replay-main/data_base/demos/<demo_id>/`
stores two measurements of its reference image `btn_rgb.png`: the TCP pose in base
(`trajectory_meta.json` → `reference_start.pose`) and the FoundationPose CAD pose
in the camera (`sam6d_foundationpose_*/cad_pose_camera.json`, produced by
`pipeline/tools/process_saved_demos.py`). With the same eye-in-hand calibration:

```text
T_tcp_cad_demo  = T_tcp_camera @ T_camera_cad_demo
T_base_cad_now  = T_base_tcp_capture @ T_tcp_camera @ T_camera_cad_now
T_base_tcp_goal = T_base_cad_now @ inv(T_tcp_cad_demo)
```

After the move the camera sees the CAD as in the demo image. If the object did not
move, the goal equals the recorded demo TCP; the demo TCP pose is otherwise used
only to report the object displacement since the demo. Replay the skill afterwards
from the current TCP (DINOBOT `replay_demo(..., ask_start_pose=False)`, as the
DINOPDDL bridge does); do **not** move to the recorded start pose first.

- The demo list comes from `alignment.skill_bindings_file`
  (default `../../PDDL/DINOPDDL/config/pddl_to_robot.yaml`: part → skill → `demo_id`).
- *Skill demo of the detected CAD* picks the single demo whose saved pose shows the
  detected CAD (also with automatic assembly identification); a specific demo can be
  selected instead. Several demos of one CAD require an explicit choice.
- Fail-closed checks: catalog CAD only (same mesh hash as the demo pose), detected CAD
  equals the demo CAD, same robot and TCP offset as the recording, and the saved pose
  must come from the current reference image. A demo re-recorded after its pose was
  estimated is rejected until `process_saved_demos.py` is run again for it. Demo files
  are rechecked immediately before motion.
- The usual step, workspace, IK, joint, freshness and confirmation interlocks apply
  unchanged. Changing the target selection invalidates the plan.
- Accuracy depends on both hand-eye calibration and both FoundationPose estimates.
  A symmetric CAD can produce a flipped pose: review the overlay and the reported
  object rotation before executing.

### Execute the skill from the aligned pose

After a completed *Skill demo* alignment, **Execute skill** replays that demo's
recorded trajectory with DINOBOT, starting from the aligned TCP. It is single-use
per alignment and requires a new confirmation. Before handing over, CAD Tracking
checks that the alignment is younger than `motion.max_capture_age_s`, the robot is
stationary at the aligned pose (`position_tolerance_m` / `rotation_tolerance_deg`),
the TCP and settings are unchanged and the demo files still match.

UR RTDE accepts a single control client, and DINOBOT and CAD Tracking both have a
top-level `settings` module, so CAD Tracking **disconnects its RTDE session** and
starts [`skill_runner.py`](skill_runner.py) with the DINOBOT interpreter
(`alignment.skill_python`). The runner uses DINOBOT's own robot wrapper,
`MotionControl` and `replay_demo(..., ask_start_pose=False)`, as the DINOPDDL bridge
does, and rechecks the start pose itself. The DINOBOT robot wrapper **activates and
opens the Robotiq gripper** when it connects. **Stop robot** sends SIGINT to the
runner, which brakes with DINOBOT's `stopJ` (`STOP_J_DECELERATION`).

DINOBOT writes its usual replay trace to `data_base/replays/`; the alignment job
folder keeps `skill_execution_request_*.json`, `skill_runner.log` and
`skill_result_*.json`. After the skill, the robot is disconnected from CAD Tracking:
select **Connect robot** before another alignment. `alignment.skill_speed_factor`
accepts DINOBOT's replay speeds 1, 1.25, 1.5 or 2.

## CAD face center and orientation

The default target is 200 mm along CAD +Y from the center of the +Y face of the
CAD bounding box. This is a geometric convention, not independent confirmation
that +Y is the physical front of every imported CAD. Review the target plane in
the preview; `alignment.cad_overrides` can supply a part-specific point.
The HMI `02_control_panel` retains `[0,30,128.75]` mm and its target
`[0,230,128.75]` mm. The prepared `03_option_cassette` has default face center
`[0,30,134.75]` mm and target `[0,230,134.75]` mm. Changing objects recomputes
the point; the HMI override is never copied to the cassette or an unknown upload.

The selected `plane_frame` mode, CAD +X as plane X and zero yaw implement
**+TCP Y = +CAD Z**, **+TCP Z = −CAD Y**, and **+TCP X = +CAD X**.
All axes follow the measured CAD pose. Legacy orientation modes are retained for
reproducibility. The plan records the chosen CAD, selection mode, mesh/catalog
hashes, face-center coordinates and whether that point came from the bounding
box or a configured override. Source thresholds remain unchanged; a failed
estimate reports the selected CAD, best available SAM6D score and threshold.

## Change alignment speed from the interface

Set **Alignment speed (mm/s)** and select **Apply speed**. The backend converts
to m/s, validates a positive finite value up to the existing module limit of
250 mm/s and persists only `motion.speed_m_s`. Acceleration and all controller
safety settings are unchanged. This ceiling is an application limit, not a claim
that the maximum is suitable for the physical installation.

Applying speed clears the current target. Select **Detect pose** again before
executing. Changes are rejected while alignment is busy, when connected telemetry
reports motion, or if settings have changed since the interface last read them.
A stationary connected robot uses the new speed without reconnecting. The plan
records the speed and acceleration that will be passed to `moveL`.

## Rotation allowance and parameter reference

`motion.max_rotation_deg` is now **180**. Any shortest-arc TCP orientation change
up to 180 degrees is allowed by this application check, independently of tool-Z
direction. The interrupted downward-only change is not enabled. No extra per-plan
rotation-override button is needed; the final execution confirmation remains.
This orientation allowance does not request multiple revolutions of a joint.
IK, joint continuity/delta, controller limits, workspace, freshness and stopping
remain active. A feasible endpoint alone does not validate the physical swept path.

Robot alignment settings belong in `alignment_pose_cad/settings.json`:

| Key | Meaning | Current configuration |
| --- | --- | --- |
| `motion.speed_m_s` | Requested linear TCP speed; UI displays mm/s | 0.25 m/s = 250 mm/s |
| `motion.acceleration_m_s2` | Requested moveL acceleration | 0.01 m/s² = 10 mm/s² |
| `motion.max_rotation_deg` | Maximum shortest-arc orientation change | 180 degrees |
| `motion.max_translation_m` | Maximum capture-to-target translation | 0.30 m |
| `motion.max_joint_delta_rad` | Maximum sampled IK change per joint, from initial and preceding solutions; no 2π wrapping | 2.356194490192345 rad, 135 degrees |
| `motion.workspace_min_m` / `workspace_max_m` | TCP box in robot-base coordinates | [-0.8,-0.8,0.05] to [0.8,0.8,0.9] m |
| `motion.path_samples` | Number of sampled controller IK/limit queries | 21 |
| `motion.max_plan_age_s` | Time allowed after plan calculation | 30 s |
| `motion.max_capture_age_s` | Maximum age of the image used for the target | 120 s |
| `motion.position_tolerance_m` | Endpoint position acceptance tolerance | 0.002 m |
| `motion.rotation_tolerance_deg` | Endpoint orientation acceptance tolerance | 1 degree |
| `motion.move_timeout_s` | Movement timeout | 90 s |
| `motion.inference_timeout_s` | Pose-estimation timeout | 300 s |
| `motion.stop_deceleration_m_s2` | Requested software-stop deceleration | 0.5 m/s² |
| `alignment.offset_m` | Standoff along the outward CAD plane normal | 0.20 m |
| `alignment.skill_bindings_file` | DINOPDDL part → skill → demo registry, for *Skill demo* targets | `../../PDDL/DINOPDDL/config/pddl_to_robot.yaml` |
| `alignment.skill_python` | DINOBOT environment interpreter that runs `skill_runner.py` | `../../../anaconda3/envs/dinobot_env/bin/python` |
| `alignment.skill_speed_factor` | DINOBOT replay speed for **Execute skill** | 1.0 |

The loader accepts positive finite speed up to 0.25 m/s, acceleration up to
0.5 m/s², rotation up to 180 degrees and 2–101 path samples. These are application
bounds, not a suitability assessment of those values for the real cell. Joint-delta
limits are separate from the physical joint limits enforced by the controller.

Requested speed may not be reached on short moves because acceleration is low.
For an ideal symmetric acceleration/braking profile, 0.10 m travel with
0.01 m/s² acceleration peaks near 0.0316 m/s (31.6 mm/s), even if 250 mm/s is
requested. Actual controller execution may differ. Acceleration has not been
increased as part of this change.

Speed can be applied from the UI while stationary. After other JSON edits,
disconnect/reconnect RTDE and calculate a new target to reload robot settings.
Restart the server when Python code changed, then refresh with Ctrl+F5.
Changing calibration/TCP fields requires their corresponding device verification.

`config/interface.json` configures vision UI defaults (camera resolution/FPS,
inference thresholds, preload and port). `config/settings.json` configures paths,
model environments/checkpoints and runtime settings. Neither is the robot motion
configuration. `config/suite.json` belongs to the offline evaluation suite.

## Automatic interlocks retained

- Active TCP must match the configured TCP and the full calibration TCP
  `[0,0,0.150,0,0,0]`, within 0.1 mm and 0.1 degrees. The application never sets TCP.
- The capture camera serial must equal `camera.serial` in `matrix_calibration.yaml`;
  RGB-D must be fresh, rectified and aligned to the color optical frame.
- The robot must remain stationary through capture and inference. TCP or joint
  drift invalidates the target.
- Workspace, displacement, orientation, controller safety state, IK, joint
  continuity and controller joint limits are checked automatically for 21 samples.
  Failed checks prevent publishing an executable target.
- The plan expires after 30 seconds; capture age is limited to 120 seconds.
  Changed settings, YAML or CAD invalidate it. Detect the pose again after expiry.
- Execute requires `confirmed: true` for the current plan ID. The plan is single-use.
  Before dispatch, the backend repeats controller checks and rechecks settings,
  calibration, TCP, freshness and authorization immediately before `moveL`.
- Controller protections, speed/acceleration, stopping and timeout mechanisms
  remain unchanged. The selected speed is shown in mm/s and recorded in every plan.

These checks do not compute obstacle avoidance or certify the continuous swept
volume of the arm, tool, camera and cables. `collision_checked` and
`accuracy_validated` remain `false`. The manual review form is no longer a gate;
physical scene changes still require the operator to reconsider the path.
The final controller TCP error measures command tracking, not independent HMI
alignment accuracy. No completion changes the calibration to `validated`.

## Diagnose joint-path rejection

`motion.max_joint_delta_rad` is separate from the 180-degree TCP orientation
allowance. It is checked twice for each sampled IK solution: against the preceding
solution and against the robot's starting joint configuration. The current
2.356194490192345 rad threshold is 135 degrees per joint. A path can therefore be smooth
between samples and still exceed the allowed excursion from its start.

Errors now distinguish **Adjacent IK step** from **Joint change from starting
posture**, and report the joint name/number, waypoint, measured delta in radians
and degrees, and the configured threshold. No angle is wrapped by 2π to hide
an extra physical turn. Controller joint limits remain a separate check.

`robot_alignment_candidate.json` saves the proposed target before controller
validation; it is not an executable or approved plan. On rejection,
`robot_alignment_error.json` includes `controller_check_details`: initial,
previous and rejected IK joints, both delta arrays, sample index and accepted
samples. Failed explicit rechecks save `controller_rejection_*.json` as well.
Old generic errors cannot identify the rejected joint retrospectively because
those intermediate IK solutions were not recorded. After loading the update,
perform a fresh detection/check without execution to obtain that evidence.

## Frames and provenance

```text
T_base_cad = T_base_tcp_capture @ T_tcp_camera @ T_camera_input_cad
T_base_target = T_base_cad @ T_cad_plane @ T_plane_target
```

`T_tcp_camera` is the SDU `MovingTransform` (camera relative to the calibration
TCP, metres), re-expressed for the active TCP using `tcp_offset_m_rad` from
`matrix_calibration.yaml`. CAD pose translation is converted from mm to m; UR
orientation uses rotation vectors in radians. The original CAD origin is restored
after the estimator's internal centering. The robot's native base is used; no
independent physical frame validation is claimed.

Each result stores capture and plan metadata, calibration/CAD hashes, active TCP,
21 sampled IK solutions in `controller_check_*.json`, explicit execution consent
in `trial_authorization_*.json`, repeated pre-dispatch controller checks and any
execution outcome. No supervisor name or evidence text is required by the API.

## Start and use

```bash
cd /home/raca/projects/CADTracking
bash run.sh ui
```

Open http://127.0.0.1:8770/ and reload with Ctrl+F5 after restarting an older
server. Open **Live Intel RealSense**, stop continuous detection/tracking, and
select **Connect robot → Detect pose → Execute alignment**. The alignment panel
uses the same CAD source as the vision controls and displays the resulting CAD identity.
Starting or restarting the server never connects or moves the robot automatically.

## Historical no-motion result

### Comprobación real realizada el 11 de noviembre de 2026

A las 18:11 CET se capturó el HMI fijo con RealSense `831612072516` y se
consultó el controlador real, sin enviar movimiento. El intento inicial había
sido rechazado por `ur_rtde: Please enable remote control on the robot!`;
la conexión posterior funcionó. No se cambió ese modo desde la aplicación.

El TCP activo coincidió con `[0,0,0.150,0,0,0]`. El controlador aceptó las
21 muestras, con salto máximo entre muestras de 0.02223 rad y desviación máxima
respecto a la solución inicial de 0.38756 rad (límite configurado 1.5 rad).
El objetivo TCP en base fue:

```text
[-0.3488959871, 0.2666617246, 0.3003585279,
  2.2166864048, -2.2168240301, -0.0385856026]  [m, rad rotvec]
```

Desplazamiento desde captura: 0.14419 m; giro: 1.45764 grados.
Informe: [supervised_trial_preflight.json](../results/interface/jobs/foundationpose/c44d1d9f08e64416afdb3deeb6496910/supervised_trial_preflight.json).
El YAML conservó SHA256 `88867ba0bd2d4b96af596cc1a674f78296b57ee46b17be7b8cdd45a54b1e1a6e`
y estado provisional. No hay revisión física completada ni autorización del
operador ni validación de precisión/colisiones. Este objetivo caduca a los 30 s;
es evidencia de comprobación, no una consigna reutilizable para ejecutar.

## Tests without hardware

```bash
cd /home/raca/projects/CADTracking
PYTHONPATH=pipeline:alignment_pose_cad/tests \
../.venvs/assembly-runtime/bin/python -B -m pytest -p no:cacheprovider -q alignment_pose_cad/tests
```

HTTP and browser tests use simulated robot/camera/inference. They check automatic
IK failures, TCP changes, freshness, multiple CADs, automatic identity selection,
mesh/catalog changes, explicit consent, single-use plans,
preview-only blocking and cancellation of the final confirmation. Local HTTP
ports and headless Chrome require host permissions. They do not validate physical
collision clearance or hand-eye accuracy.
