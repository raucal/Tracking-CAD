# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in this software and any modifications.

from estimater import *
from real_sense_interface import RealsenseCamera
from run_video_demo import depth_to_meters, draw_pose_text, make_bbox_lineset, make_K, make_point_cloud, select_roi_mask, transform_point
from scipy.spatial.transform import Rotation
import argparse
import json
import shutil


from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from teleop_setup.robots.ur5e_iq_gripper.robotiq.robotiq_gripper_control import RobotiqGripper
from scipy.spatial.transform import Rotation as R


# Fill these in directly for your UR5e setup.
# Convention:
#   base_T_tcp maps points from the UR5e TCP frame into the robot base frame.
#   tcp_T_cam maps points from the RealSense color camera frame into the TCP frame.
#   cam_T_obj is the FoundationPose output, named ob_in_cam in the original code.
#   base_T_obj = base_T_tcp @ tcp_T_cam @ cam_T_obj
USE_RTDE_TCP_POSE = False
ROBOT_IP = "192.168.1.10"
MANUAL_TCP_POSE_XYZRXRYRZ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Saved grasp setup from grasp_config_ui.py.
# Leave TARGET_GRASP_NAME as None to use TARGET_GRASP_INDEX.
# Set it to a string only when you want a specific named grasp by default.
GRASP_CONFIG_PATH = "grasp_peg_config.json"
TARGET_GRASP_NAME = None
TARGET_GRASP_INDEX = 0


# RealSense color optical frame convention:
#   +X points right in the image, +Y points down in the image, +Z points forward
#   from the camera into the scene.
#
# TCP_T_CAMERA maps points from the RealSense camera frame into the TCP frame.
# In other words, it is the pose of the camera frame expressed in the TCP frame.
# If your calibration gives CAMERA_T_TCP instead, set TCP_T_CAMERA to
# invert_transform(CAMERA_T_TCP).
#
# Fill in both rotation and translation. Translation alone is almost never
# enough, because the RealSense optical axes are usually not aligned with TCP.
#
# If the camera image is upside down, this 180 degree rotation around the
# camera optical Z axis flips camera X/Y while keeping camera Z/depth forward.
CAMERA_IS_UPSIDE_DOWN = True

TCP_T_CAMERA = np.eye(4, dtype=np.float32)
if CAMERA_IS_UPSIDE_DOWN:
  TCP_T_CAMERA[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
TCP_T_CAMERA[:3, 3] = [0.0325, 0.03, -0.17 + 0.03]

def invert_transform(tf):
  inv = np.eye(4, dtype=np.float32)
  inv[:3, :3] = tf[:3, :3].T
  inv[:3, 3] = -inv[:3, :3] @ tf[:3, 3]
  return inv


def transform_from_ur_pose(xyzrxryrz):
  xyzrxryrz = np.asarray(xyzrxryrz, dtype=np.float32)
  tf = np.eye(4, dtype=np.float32)
  tf[:3, :3] = Rotation.from_rotvec(xyzrxryrz[3:6]).as_matrix()
  tf[:3, 3] = xyzrxryrz[:3]
  return tf


def ur_pose_from_transform(tf):
  return np.concatenate([tf[:3, 3], Rotation.from_matrix(tf[:3, :3]).as_rotvec()])


def load_grasp_config(path):
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  if "mesh_file" not in data:
    raise ValueError(f"{path} is missing mesh_file")
  if "mesh_scale" not in data:
    raise ValueError(f"{path} is missing mesh_scale")
  data.setdefault("grasps", [])
  return data


def choose_grasp(base_T_grasps, grasp_name=None, grasp_index=0):
  if not base_T_grasps:
    return None
  if grasp_name:
    matches = [grasp for grasp in base_T_grasps if grasp["name"] == grasp_name]
    if not matches:
      available = ", ".join(grasp["name"] for grasp in base_T_grasps)
      raise ValueError(f'No grasp named "{grasp_name}" in grasp config. Available grasps: {available}')
    return matches[0]
  if grasp_index < 0 or grasp_index >= len(base_T_grasps):
    raise IndexError(f'--grasp_index {grasp_index} outside available grasps 0..{len(base_T_grasps) - 1}')
  return base_T_grasps[grasp_index]


def get_current_base_T_tcp():
  if USE_RTDE_TCP_POSE:
    from rtde_receive import RTDEReceiveInterface
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    tcp_pose = rtde_r.getActualTCPPose()
  else:
    tcp_pose = MANUAL_TCP_POSE_XYZRXRYRZ
    if np.allclose(tcp_pose, 0):
      logging.warning('Using zero MANUAL_TCP_POSE_XYZRXRYRZ. Fill it in before trusting base-frame results.')
  return transform_from_ur_pose(tcp_pose)


def get_tcp_T_cam():
  if np.allclose(TCP_T_CAMERA[:3, :3], np.eye(3), atol=1e-7):
    logging.warning('TCP_T_CAMERA rotation is identity. Fill in the hand-eye calibration rotation if the camera is not aligned with TCP.')
  return TCP_T_CAMERA.copy()


def write_transform(name, tf):
  print(f'\n{name}:')
  for row in tf:
    print('[{: .6f} {: .6f} {: .6f} {: .6f}]'.format(*row))
  print(f'{name} translation xyz [m]: {tf[:3, 3]}')
  print(f'{name} rotation rotvec [rad]: {Rotation.from_matrix(tf[:3, :3]).as_rotvec()}')


def print_frame_diagnostics(base_T_tcp, tcp_T_cam):
  base_T_cam = base_T_tcp @ tcp_T_cam
  R_base_cam = base_T_cam[:3, :3]
  print('\nCamera axes expressed in robot base frame:')
  print(f'camera +X/right   -> base direction {R_base_cam[:, 0]}')
  print(f'camera +Y/down    -> base direction {R_base_cam[:, 1]}')
  print(f'camera +Z/forward -> base direction {R_base_cam[:, 2]}')
  print('If the object moves farther from the camera, base_T_obj translation should change mostly along camera +Z/forward above.')
  return base_T_cam


def make_origin_marker(position, color, radius=0.015):
  marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
  marker.translate(position)
  marker.paint_uniform_color(color)
  return marker


def make_line(start, end, color):
  line = o3d.geometry.LineSet()
  line.points = o3d.utility.Vector3dVector(np.array([start, end], dtype=np.float64))
  line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
  line.colors = o3d.utility.Vector3dVector(np.array([color], dtype=np.float64))
  return line


def label_position(position, z_offset=0.04):
  position = np.asarray(position, dtype=np.float64).copy()
  position[2] += z_offset
  return position


def make_frame(tf, size):
  frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
  frame.transform(tf)
  return frame


def draw_debug_geometries(geometries, labels):
  try:
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer('FoundationPose robot-frame debug scene', 1280, 840)
    for name, geometry in geometries:
      vis.add_geometry(name, geometry)
    for text, position in labels:
      vis.add_3d_label(position, text)
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
  except Exception as e:
    logging.warning(f'Falling back to Open3D draw_geometries without 3D labels: {e}')
    o3d.visualization.draw_geometries(
      [geometry for _, geometry in geometries],
      window_name='FoundationPose robot-frame debug scene',
    )


def show_robot_debug_scene(color, depth, K, bbox, to_origin, base_T_tcp, base_T_cam, cam_T_obj, base_T_obj, base_T_new_tcp=None, base_T_grasps=None, frame_size=0.12):
  base_T_grasps = [] if base_T_grasps is None else base_T_grasps
  pcd_base = make_point_cloud(color, depth, K, stride=3, zfar=2.0)
  pcd_base.transform(base_T_cam)
  base_T_bbox = base_T_obj @ np.linalg.inv(to_origin)

  base_frame = make_frame(np.eye(4), frame_size * 1.4)
  tcp_frame = make_frame(base_T_tcp, frame_size)
  cam_frame = make_frame(base_T_cam, frame_size)
  obj_origin_frame = make_frame(base_T_obj, frame_size * 0.7)
  obj_bbox_frame = make_frame(base_T_bbox, frame_size)
  new_tcp_frame = make_frame(base_T_new_tcp, frame_size) if base_T_new_tcp is not None else None

  bbox_obj = make_bbox_lineset(bbox, base_T_bbox, color=(0.0, 1.0, 0.0))

  base_marker = make_origin_marker(np.zeros(3), [1.0, 1.0, 1.0])
  tcp_marker = make_origin_marker(base_T_tcp[:3, 3], [0.0, 0.4, 1.0])
  cam_marker = make_origin_marker(base_T_cam[:3, 3], [1.0, 0.0, 1.0])
  obj_origin_marker = make_origin_marker(base_T_obj[:3, 3], [1.0, 0.6, 0.0])
  obj_bbox_marker = make_origin_marker(base_T_bbox[:3, 3], [0.0, 1.0, 0.0])
  new_tcp_marker = make_origin_marker(base_T_new_tcp[:3, 3], [1.0, 1.0, 0.0], radius=0.02) if base_T_new_tcp is not None else None

  base_to_tcp = make_line(np.zeros(3), base_T_tcp[:3, 3], [0.0, 0.4, 1.0])
  tcp_to_cam = make_line(base_T_tcp[:3, 3], base_T_cam[:3, 3], [1.0, 0.0, 1.0])
  cam_to_obj = make_line(base_T_cam[:3, 3], base_T_bbox[:3, 3], [0.0, 1.0, 0.0])
  base_to_obj = make_line(np.zeros(3), base_T_bbox[:3, 3], [1.0, 0.0, 0.0])
  base_to_new_tcp = make_line(np.zeros(3), base_T_new_tcp[:3, 3], [1.0, 1.0, 0.0]) if base_T_new_tcp is not None else None
  tcp_to_new_tcp = make_line(base_T_tcp[:3, 3], base_T_new_tcp[:3, 3], [0.0, 1.0, 1.0]) if base_T_new_tcp is not None else None
  obj_to_new_tcp = make_line(base_T_bbox[:3, 3], base_T_new_tcp[:3, 3], [1.0, 0.5, 0.0]) if base_T_new_tcp is not None else None

  print('\nOpening robot debug scene:')
  print('  RGB axes at robot base, TCP, camera, object origin, object bbox frame, and new TCP target.')
  print('  White sphere: robot base origin')
  print('  Blue sphere: TCP origin')
  print('  Magenta sphere/line: camera origin / TCP-to-camera')
  print('  Orange sphere/small frame: OBJ origin, base_T_cam @ cam_T_obj')
  print('  Green box/frame: visual bbox frame, base_T_cam @ cam_T_obj @ inv(to_origin)')
  print('  Yellow sphere/frame: new_tcp_pose target passed to moveL')
  print('  Point cloud is transformed into robot base frame using base_T_cam.')
  print('  Connections: base->TCP blue, TCP->camera magenta, camera->object green, base->object red.')
  print('  Extra target connections: base->new TCP yellow, current TCP->new TCP cyan, object->new TCP orange.')

  geometries = [
    ('RealSense point cloud in robot base frame', pcd_base),
    ('Robot base frame', base_frame),
    ('Robot TCP frame', tcp_frame),
    ('RealSense camera frame', cam_frame),
    ('Object origin frame', obj_origin_frame),
    ('Object bbox frame', obj_bbox_frame),
    ('Object bbox', bbox_obj),
    ('Robot base marker', base_marker),
    ('TCP marker', tcp_marker),
    ('Camera marker', cam_marker),
    ('Object origin marker', obj_origin_marker),
    ('Object bbox marker', obj_bbox_marker),
    ('connection: robot base -> robot TCP', base_to_tcp),
    ('connection: robot TCP -> robot camera', tcp_to_cam),
    ('connection: camera -> object', cam_to_obj),
    ('connection: robot base -> object', base_to_obj),
  ]
  if base_T_new_tcp is not None:
    geometries += [
      ('New TCP target frame', new_tcp_frame),
      ('New TCP target marker', new_tcp_marker),
      ('connection: robot base -> new TCP target', base_to_new_tcp),
      ('connection: current TCP -> new TCP target', tcp_to_new_tcp),
      ('connection: object -> new TCP target', obj_to_new_tcp),
    ]
  for idx, grasp in enumerate(base_T_grasps):
    grasp_tf = grasp["base_T_grasp"]
    grasp_frame = make_frame(grasp_tf, frame_size * 0.8)
    grasp_marker = make_origin_marker(grasp_tf[:3, 3], [1.0, 1.0, 0.0], radius=0.012)
    base_to_grasp = make_line(np.zeros(3), grasp_tf[:3, 3], [1.0, 1.0, 0.0])
    geometries += [
      (f'Configured grasp frame {idx}: {grasp["name"]}', grasp_frame),
      (f'Configured grasp marker {idx}: {grasp["name"]}', grasp_marker),
      (f'connection: robot base -> configured grasp {idx}', base_to_grasp),
    ]
  labels = [
    ('robot base', label_position(np.zeros(3), frame_size * 0.45)),
    ('robot TCP', label_position(base_T_tcp[:3, 3], frame_size * 0.45)),
    ('robot camera', label_position(base_T_cam[:3, 3], frame_size * 0.45)),
    ('object origin', label_position(base_T_obj[:3, 3], frame_size * 0.35)),
    ('object bbox / pose', label_position(base_T_bbox[:3, 3], frame_size * 0.45)),
    ('base -> TCP', (np.zeros(3) + base_T_tcp[:3, 3]) / 2.0),
    ('TCP -> camera', (base_T_tcp[:3, 3] + base_T_cam[:3, 3]) / 2.0),
    ('camera -> object', (base_T_cam[:3, 3] + base_T_bbox[:3, 3]) / 2.0),
    ('base -> object', base_T_bbox[:3, 3] / 2.0),
  ]
  if base_T_new_tcp is not None:
    labels += [
      ('new_tcp_pose target', label_position(base_T_new_tcp[:3, 3], frame_size * 0.65)),
      ('base -> new TCP', base_T_new_tcp[:3, 3] / 2.0),
      ('TCP -> new TCP', (base_T_tcp[:3, 3] + base_T_new_tcp[:3, 3]) / 2.0),
      ('object -> new TCP', (base_T_bbox[:3, 3] + base_T_new_tcp[:3, 3]) / 2.0),
    ]
  for idx, grasp in enumerate(base_T_grasps):
    grasp_tf = grasp["base_T_grasp"]
    labels += [
      (f'grasp {idx}: {grasp["name"]}', label_position(grasp_tf[:3, 3], frame_size * 0.55)),
      (f'base -> grasp {idx}', grasp_tf[:3, 3] / 2.0),
    ]

  draw_debug_geometries(geometries, labels)


def capture_single_frame(camera):
  while True:
    color_bgr, depth_raw = camera.get_aligned_frames()
    if color_bgr is None or depth_raw is None:
      continue
    cv2.imshow('RealSense capture - press c', color_bgr)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      return None, None
    if key == ord('c'):
      cv2.destroyWindow('RealSense capture - press c')
      return color_bgr, depth_raw


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--grasp_config', type=str, default=None, help='JSON saved by grasp_config_ui.py. If omitted, GRASP_CONFIG_PATH from this file is used.')
  parser.add_argument('--mesh_file', type=str, default=None)
  parser.add_argument('--mesh_scale', type=float, default=1.0, help='Scale applied to mesh vertices. Use 0.001 for OBJ files stored in millimeters.')
  parser.add_argument('--device_number', type=int, default=0)
  parser.add_argument('--depth_scale', type=float, default=None)
  parser.add_argument('--roi_depth_margin', type=float, default=0.05, help='Depth band in meters used to refine the ROI mask. Use 0 for raw rectangle.')
  parser.add_argument('--show_robot_scene', action='store_true', default=True)
  parser.add_argument('--no_robot_scene', dest='show_robot_scene', action='store_false')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_single_pose')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  grasp_config_path = args.grasp_config if args.grasp_config is not None else GRASP_CONFIG_PATH
  grasp_config = load_grasp_config(grasp_config_path) if grasp_config_path else None
  if grasp_config is not None:
    args.mesh_file = grasp_config["mesh_file"]
    args.mesh_scale = float(grasp_config["mesh_scale"])
    grasp_names = [grasp.get("name", f"grasp_{idx:02d}") for idx, grasp in enumerate(grasp_config.get("grasps", []))]
    if grasp_names:
      print(f'Loaded grasp config with grasps: {", ".join(grasp_names)}')
    else:
      print('Loaded grasp config, but it does not contain any saved grasps yet.')
  if args.mesh_file is None:
    raise ValueError("Pass --grasp_config, or pass --mesh_file and --mesh_scale.")

  rtde_c = RTDEControlInterface(ROBOT_IP)
  rtde_r = RTDEReceiveInterface(ROBOT_IP)
  gripper = RobotiqGripper(rtde_c)


  HOME = [-0.3161773697897597, -0.17373151021556782, 0.7114076806002352-0.15, 1.8025043626208224, -2.5243109338822083, 0.02892631039643908]
  # curr_tcp_pose = rtde_r.getActualTCPPose()
  # print("homepose: ", curr_tcp_pose)
  rtde_c.moveL(HOME, speed=0.1, acceleration=0.1)
  

  # Used when USE_RTDE_TCP_POSE is False.
  # UR pose format: x, y, z, rx, ry, rz in meters and radians.
  MANUAL_TCP_POSE_XYZRXRYRZ = rtde_r.getActualTCPPose() #np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


  base_T_tcp = get_current_base_T_tcp()
  tcp_T_cam = get_tcp_T_cam()

  mesh = trimesh.load(args.mesh_file)
  if args.mesh_scale != 1.0:
    mesh.apply_scale(args.mesh_scale)

  debug_dir = args.debug_dir
  shutil.rmtree(debug_dir, ignore_errors=True)
  os.makedirs(debug_dir, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  if grasp_config is not None and "source_mesh_to_object_frame" in grasp_config:
    to_origin = np.asarray(grasp_config["source_mesh_to_object_frame"], dtype=np.float32).reshape(4, 4)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=args.debug,
    glctx=glctx,
  )
  logging.info('estimator initialization done')

  camera = RealsenseCamera(device_number=args.device_number)
  K = make_K(camera)
  logging.info(f'K:\n{K}')

  color_bgr, depth_raw = capture_single_frame(camera)
  if color_bgr is None or depth_raw is None:
    cv2.destroyAllWindows()
    raise SystemExit('Capture cancelled.')

  color = color_bgr[..., ::-1].copy()
  depth = depth_to_meters(depth_raw, camera, args.depth_scale)
  mask, roi_depth = select_roi_mask(color, depth=depth, depth_margin=args.roi_depth_margin)

  cam_T_obj = est.register(K=K, rgb=color, depth=depth, ob_mask=mask.astype(bool), iteration=args.est_refine_iter)
  base_T_cam = print_frame_diagnostics(base_T_tcp, tcp_T_cam)
  base_T_obj = base_T_cam @ cam_T_obj
  base_T_bbox = base_T_obj @ np.linalg.inv(to_origin)
  base_T_grasps = []
  if grasp_config is not None:
    for grasp in grasp_config.get("grasps", []):
      object_T_grasp = np.asarray(grasp["object_T_grasp"], dtype=np.float32).reshape(4, 4)
      base_T_grasps.append({
        "name": grasp.get("name", f"grasp_{len(base_T_grasps):02d}"),
        "base_T_grasp": base_T_bbox @ object_T_grasp,
        "object_T_grasp": object_T_grasp,
        "gripper_opening": grasp.get("gripper_opening"),
      })
  if base_T_grasps:
    target_grasp = choose_grasp(base_T_grasps, grasp_name=TARGET_GRASP_NAME, grasp_index=TARGET_GRASP_INDEX)
    target_tf = target_grasp["base_T_grasp"]
  else:
    target_grasp = None
    target_tf = base_T_bbox
  new_tcp_pose = ur_pose_from_transform(target_tf)
  base_T_new_tcp = transform_from_ur_pose(new_tcp_pose)

  obj_center_cam = transform_point(cam_T_obj, est.model_center)
  obj_center_base = transform_point(base_T_obj, est.model_center)

  np.savetxt(f'{debug_dir}/cam_T_obj.txt', cam_T_obj)
  np.savetxt(f'{debug_dir}/base_T_tcp.txt', base_T_tcp)
  np.savetxt(f'{debug_dir}/tcp_T_cam.txt', tcp_T_cam)
  np.savetxt(f'{debug_dir}/base_T_cam.txt', base_T_cam)
  np.savetxt(f'{debug_dir}/base_T_obj.txt', base_T_obj)
  np.savetxt(f'{debug_dir}/base_T_bbox.txt', base_T_bbox)
  np.savetxt(f'{debug_dir}/base_T_new_tcp.txt', base_T_new_tcp)
  np.savetxt(f'{debug_dir}/new_tcp_pose.txt', new_tcp_pose)
  for idx, grasp in enumerate(base_T_grasps):
    safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in grasp["name"])
    np.savetxt(f'{debug_dir}/base_T_grasp_{idx:02d}_{safe_name}.txt', grasp["base_T_grasp"])
  cv2.imwrite(f'{debug_dir}/color_bgr.png', color_bgr)
  cv2.imwrite(f'{debug_dir}/depth_mm.png', np.round(depth * 1000).astype(np.uint16))
  cv2.imwrite(f'{debug_dir}/mask.png', (mask * 255).astype(np.uint8))

  center_pose = cam_T_obj @ np.linalg.inv(to_origin)
  vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
  vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
  vis = draw_pose_text(vis, cam_T_obj, mesh_center_cam=obj_center_cam, roi_depth=roi_depth)
  cv2.imwrite(f'{debug_dir}/pose_overlay_bgr.png', vis[..., ::-1])

  write_transform('base_T_tcp', base_T_tcp)
  write_transform('tcp_T_cam', tcp_T_cam)
  write_transform('base_T_cam', base_T_cam)
  write_transform('cam_T_obj', cam_T_obj)
  write_transform('base_T_obj', base_T_obj)
  write_transform('base_T_bbox', base_T_bbox)
  write_transform('base_T_new_tcp', base_T_new_tcp)
  for idx, grasp in enumerate(base_T_grasps):
    write_transform(f'base_T_grasp_{idx:02d}_{grasp["name"]}', grasp["base_T_grasp"])
  target_name = target_grasp["name"] if target_grasp is not None else "base_T_bbox"
  print(f'\nnew_tcp_pose sent to moveL from {target_name} [x y z rx ry rz]: {new_tcp_pose}')
  print(f'\nObject mesh center in camera xyz [m]: {obj_center_cam}')
  print(f'Object mesh center in robot base xyz [m]: {obj_center_base}')
  if roi_depth is not None:
    print(f'Initial ROI median depth [m]: {roi_depth:.6f}')
  print(f'\nSaved outputs to: {debug_dir}')

  if args.show_robot_scene:
    show_robot_debug_scene(color, depth, K, bbox, to_origin, base_T_tcp, base_T_cam, cam_T_obj, base_T_obj, base_T_new_tcp=base_T_new_tcp, base_T_grasps=base_T_grasps)

  cv2.imshow('Estimated pose overlay', vis[..., ::-1])
  cv2.waitKey(0)
  cv2.destroyAllWindows()


  print("Final base_T_obj Transform: ", base_T_obj)
  print("Final base_T_bbox Transform used for new_tcp_pose: ", base_T_bbox)



  ########### POLICY ##############

  # TODO: Move robot to object
  # Move arm (slowly) to object
  input("(enter) Ready to move robot? ")

  # print("new_tcp_pose: ", new_tcp_pose)


  # final_tcp_pose = new_tcp_pose.copy()
  # final_tcp_pose[2] += 0.025 # half of the object height
  # # final_tcp_pose[2] += 0.1 # above Grasping pose

  # rtde_c.moveL(final_tcp_pose.tolist(), speed=0.1, acceleration=0.1)

  print("Available grasp poses:", ", ".join(g["name"] for g in grasp_config["grasps"]))
  grasp_name = input("grasp_pose: ")
  if not grasp_name:
    grasp_name = target_grasp["name"]
  grasp = next(
    g for g in grasp_config["grasps"]
    if g["name"] == grasp_name
  )

  object_T_grasp = np.asarray(grasp["object_T_grasp"], dtype=np.float32).reshape(4, 4)

  # Convert object-relative grasp into robot base frame
  base_T_grasp = base_T_bbox @ object_T_grasp

  # Convert 4x4 transform into UR moveL format
  grasp_tcp_pose = ur_pose_from_transform(base_T_grasp)

  print(f"{grasp_name} tcp pose: ", grasp_tcp_pose)
  rtde_c.moveL(grasp_tcp_pose.tolist(), speed=0.1, acceleration=0.1)

  # final_tcp_pose[2] -= 0.1 # Grasping pose
  # rtde_c.moveL(final_tcp_pose, speed=0.1, acceleration=0.1)


  input("reset: ")
  # final_tcp_pose[2] += 0.1 # above Grasping pose
  # rtde_c.moveL(final_tcp_pose, speed=0.1, acceleration=0.1)
  rtde_c.moveL(HOME, speed=0.1, acceleration=0.1)
