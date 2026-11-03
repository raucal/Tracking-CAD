# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in this software and any modifications.

from estimater import *
from real_sense_interface import RealsenseCamera
import argparse
import shutil


def make_K(camera):
  return camera.cam_intrinsics.astype(np.float32).copy()


def depth_to_meters(depth_raw, camera, depth_scale=None):
  scale = camera.depth_scale if depth_scale is None else depth_scale
  return depth_raw.astype(np.float32) * float(scale)


def transform_point(tf, point):
  return (tf @ np.append(np.asarray(point, dtype=np.float32), 1.0))[:3]


def pose_lines(pose):
  rows = []
  for row in pose:
    rows.append('[{: .3f} {: .3f} {: .3f} {: .3f}]'.format(*row))
  return rows


def draw_pose_text(img, pose, mesh_center_cam=None, roi_depth=None):
  y = 22
  img = cv_draw_text(img, 'object origin in camera frame', (8, y), color=(255, 255, 255), outline_color=(0, 0, 0))
  y += 22
  for line in pose_lines(pose):
    img = cv_draw_text(img, line, (8, y), color=(255, 255, 255), outline_color=(0, 0, 0))
    y += 20
  t = pose[:3, 3]
  img = cv_draw_text(img, 'origin xyz: [{:.3f}, {:.3f}, {:.3f}] m'.format(*t), (8, y), color=(255, 255, 255), outline_color=(0, 0, 0))
  y += 20
  if mesh_center_cam is not None:
    img = cv_draw_text(img, 'mesh center xyz: [{:.3f}, {:.3f}, {:.3f}] m'.format(*mesh_center_cam), (8, y), color=(255, 255, 255), outline_color=(0, 0, 0))
    y += 20
  if roi_depth is not None:
    img = cv_draw_text(img, 'init ROI median depth: {:.3f} m'.format(roi_depth), (8, y), color=(255, 255, 255), outline_color=(0, 0, 0))
  return img


def make_bbox_lineset(bbox, ob_in_cam, color=(0.0, 1.0, 0.0)):
  min_xyz = bbox.min(axis=0)
  max_xyz = bbox.max(axis=0)
  corners = np.array([
    [min_xyz[0], min_xyz[1], min_xyz[2]],
    [max_xyz[0], min_xyz[1], min_xyz[2]],
    [max_xyz[0], max_xyz[1], min_xyz[2]],
    [min_xyz[0], max_xyz[1], min_xyz[2]],
    [min_xyz[0], min_xyz[1], max_xyz[2]],
    [max_xyz[0], min_xyz[1], max_xyz[2]],
    [max_xyz[0], max_xyz[1], max_xyz[2]],
    [min_xyz[0], max_xyz[1], max_xyz[2]],
  ], dtype=np.float32)
  corners = (ob_in_cam @ to_homo(corners).T).T[:, :3]
  lines = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ], dtype=np.int32)
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
  line_set.lines = o3d.utility.Vector2iVector(lines)
  line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array(color), (len(lines), 1)))
  return line_set


def make_point_cloud(color, depth, K, stride=3, zfar=2.0):
  stride = max(1, int(stride))
  xyz_map = depth2xyzmap(depth, K)
  valid = (xyz_map[..., 2] > 0.001) & (xyz_map[..., 2] < zfar)
  sample_mask = np.zeros(valid.shape, dtype=bool)
  sample_mask[::stride, ::stride] = True
  valid = valid & sample_mask
  return toOpen3dCloud(xyz_map[valid], color[valid])


class PointCloudViewer:
  def __init__(self, axis_scale):
    self.vis = o3d.visualization.Visualizer()
    if not self.vis.create_window(window_name='FoundationPose point cloud', width=960, height=720):
      logging.warning('Could not create Open3D point cloud window')
      self.vis = None
      return

    opt = self.vis.get_render_option()
    opt.background_color = np.array([0.02, 0.02, 0.02])
    opt.point_size = 2.0

    self.axis_scale = axis_scale
    self.pcd = o3d.geometry.PointCloud()
    self.bbox_lines = o3d.geometry.LineSet()
    self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_scale)
    self.object_frame = None
    self.initialized = False

  def add_geometry(self, geometry, reset_bounding_box=False):
    self.vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)

  def update(self, color, depth, K, bbox, center_pose, stride, zfar):
    if self.vis is None:
      return

    view_status = None
    if self.initialized:
      view_status = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

    pcd = make_point_cloud(color, depth, K, stride=stride, zfar=zfar)
    bbox_lines = make_bbox_lineset(bbox, center_pose)

    self.pcd.points = pcd.points
    self.pcd.colors = pcd.colors
    self.bbox_lines.points = bbox_lines.points
    self.bbox_lines.lines = bbox_lines.lines
    self.bbox_lines.colors = bbox_lines.colors

    new_object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_scale)
    new_object_frame.transform(center_pose)

    if not self.initialized:
      self.object_frame = new_object_frame
      self.add_geometry(self.pcd, reset_bounding_box=True)
      self.add_geometry(self.bbox_lines)
      self.add_geometry(self.camera_frame)
      self.add_geometry(self.object_frame)
      self.initialized = True
    else:
      self.vis.update_geometry(self.pcd)
      self.vis.update_geometry(self.bbox_lines)
      if self.object_frame is not None:
        self.vis.remove_geometry(self.object_frame, reset_bounding_box=False)
      self.object_frame = new_object_frame
      self.add_geometry(self.object_frame)

    if view_status is not None:
      self.vis.get_view_control().convert_from_pinhole_camera_parameters(view_status, allow_arbitrary=True)

    self.vis.poll_events()
    self.vis.update_renderer()

  def destroy(self):
    if self.vis is not None:
      self.vis.destroy_window()


def select_roi_mask(color_rgb, depth=None, depth_margin=0.05):
  color_bgr = color_rgb[..., ::-1]
  roi = cv2.selectROI("Select object, press ENTER", color_bgr, showCrosshair=True, fromCenter=False)
  cv2.destroyWindow("Select object, press ENTER")
  x, y, w, h = [int(v) for v in roi]
  mask = np.zeros(color_rgb.shape[:2], dtype=np.uint8)
  if w > 0 and h > 0:
    mask[y:y + h, x:x + w] = 1
  if depth is None or depth_margin <= 0 or not mask.any():
    return mask, None

  roi_depth = depth[mask.astype(bool)]
  roi_depth = roi_depth[roi_depth > 0.001]
  if len(roi_depth) == 0:
    logging.warning('Selected ROI has no valid depth; using rectangular mask')
    return mask, None

  median_depth = float(np.median(roi_depth))
  depth_mask = (depth > median_depth - depth_margin) & (depth < median_depth + depth_margin)
  filtered = mask.astype(bool) & depth_mask
  if filtered.sum() < 20:
    logging.warning('Depth-filtered ROI is too small; using rectangular mask')
    return mask, median_depth
  return filtered.astype(np.uint8), median_depth


def load_mask(mask_file, shape):
  mask = cv2.imread(mask_file, -1)
  if mask is None:
    raise RuntimeError(f"Could not read mask file: {mask_file}")
  if mask.ndim == 3:
    mask = mask[..., 0]
  if mask.shape[:2] != shape:
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
  return (mask > 0).astype(np.uint8)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--mesh_scale', type=float, default=1.0, help='Scale applied to mesh vertices. Use 0.001 for OBJ files stored in millimeters.')
  parser.add_argument('--device_number', type=int, default=0)
  parser.add_argument('--mask_file', type=str, default=None)
  parser.add_argument('--depth_scale', type=float, default=None)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--show_pointcloud', action='store_true', default=True)
  parser.add_argument('--no_pointcloud', dest='show_pointcloud', action='store_false')
  parser.add_argument('--pcd_stride', type=int, default=3)
  parser.add_argument('--pcd_zfar', type=float, default=2.0)
  parser.add_argument('--axis_scale', type=float, default=0.1)
  parser.add_argument('--roi_depth_margin', type=float, default=0.05, help='Depth band in meters used to refine the initial ROI mask. Use 0 to keep the raw rectangle.')
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_live')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  if args.mesh_scale != 1.0:
    mesh.apply_scale(args.mesh_scale)
  debug = args.debug
  debug_dir = args.debug_dir
  shutil.rmtree(debug_dir, ignore_errors=True)
  os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
  os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
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
    debug=debug,
    glctx=glctx,
  )
  logging.info("estimator initialization done")

  camera = RealsenseCamera(device_number=args.device_number)
  K = make_K(camera)
  logging.info(f'K:\n{K}')

  pose = None
  frame_id = 0
  initial_mask = None
  initial_roi_depth = None
  pcd_viewer = PointCloudViewer(args.axis_scale) if args.show_pointcloud else None

  try:
    while True:
      color_bgr, depth_raw = camera.get_aligned_frames()
      if color_bgr is None or depth_raw is None:
        logging.info('No RealSense frame received')
        continue

      color = color_bgr[..., ::-1].copy()
      depth = depth_to_meters(depth_raw, camera, args.depth_scale)

      if pose is None:
        if args.mask_file is not None and initial_mask is None:
          initial_mask = load_mask(args.mask_file, color.shape[:2])
        if initial_mask is None:
          cv2.imshow('FoundationPose live', color_bgr)
          key = cv2.waitKey(1) & 0xFF
          if key == ord('q'):
            break
          if key != ord('r'):
            continue
          initial_mask, initial_roi_depth = select_roi_mask(color, depth=depth, depth_margin=args.roi_depth_margin)

        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=initial_mask.astype(bool), iteration=args.est_refine_iter)
      else:
        pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

      np.savetxt(f'{debug_dir}/ob_in_cam/{frame_id:06d}.txt', pose.reshape(4, 4))

      center_pose = pose @ np.linalg.inv(to_origin)
      mesh_center_cam = transform_point(pose, est.model_center)
      vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=args.axis_scale, K=K, thickness=3, transparency=0, is_input_rgb=True)
      vis = draw_pose_text(vis, pose, mesh_center_cam=mesh_center_cam, roi_depth=initial_roi_depth)
      cv2.imshow('FoundationPose live', vis[..., ::-1])
      if pcd_viewer is not None:
        pcd_viewer.update(color, depth, K, bbox, center_pose, args.pcd_stride, args.pcd_zfar)

      if debug >= 2:
        imageio.imwrite(f'{debug_dir}/track_vis/{frame_id:06d}.png', vis)

      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
        break
      if key == ord('r'):
        pose = None
        initial_mask = None
        initial_roi_depth = None

      frame_id += 1
  finally:
    if pcd_viewer is not None:
      pcd_viewer.destroy()
    cv2.destroyAllWindows()
