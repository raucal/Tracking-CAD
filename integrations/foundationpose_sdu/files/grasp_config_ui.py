import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation


GRIPPER_ASSET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "UR5eGripper", "assets")
DEFAULT_GRIPPER_XML = "UR5eGripper/hande.xml"
HANDE_TCP_Z = 0.135
HANDE_FINGER_Z = 0.099


def make_transform(xyz, rpy_deg):
  tf = np.eye(4, dtype=np.float32)
  tf[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
  tf[:3, 3] = xyz
  return tf


def make_transform_from_rotation(xyz, rotation):
  tf = np.eye(4, dtype=np.float32)
  tf[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
  tf[:3, 3] = xyz
  return tf


def transform_to_xyzrpy(tf):
  rpy = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=True)
  return tf[:3, 3].copy(), rpy


def normalize_rotation(rotation):
  u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float32).reshape(3, 3))
  out = u @ vh
  if np.linalg.det(out) < 0:
    u[:, -1] *= -1
    out = u @ vh
  return out.astype(np.float32)


def matrix_to_list(tf):
  return np.asarray(tf, dtype=float).reshape(4, 4).tolist()


def list_to_matrix(values):
  return np.asarray(values, dtype=np.float32).reshape(4, 4)


def load_mesh_o3d(path, color):
  mesh = o3d.io.read_triangle_mesh(path)
  if mesh.is_empty():
    raise RuntimeError(f"Could not load mesh: {path}")
  mesh.compute_vertex_normals()
  mesh.paint_uniform_color(color)
  return mesh


def transform_mesh(mesh, tf):
  out = o3d.geometry.TriangleMesh(mesh)
  out.transform(tf)
  return out


def trimesh_to_o3d(mesh, color):
  out = o3d.geometry.TriangleMesh()
  out.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
  out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
  out.compute_vertex_normals()
  out.paint_uniform_color(color)
  return out


def load_object_mesh(path, scale):
  mesh = trimesh.load(path, force="mesh")
  if mesh.is_empty:
    raise RuntimeError(f"Could not load CAD mesh: {path}")
  if scale != 1.0:
    mesh.apply_scale(scale)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  mesh.apply_transform(to_origin)
  return mesh, to_origin, extents


def load_hande_parts(opening):
  base_shift = np.eye(4, dtype=np.float32)
  base_shift[:3, 3] = [0.0, 0.0, -HANDE_TCP_Z]

  left_tf = np.eye(4, dtype=np.float32)
  left_tf[:3, 3] = [opening / 2.0, 0.0, HANDE_FINGER_Z - HANDE_TCP_Z]

  right_tf = np.eye(4, dtype=np.float32)
  right_tf[:3, 3] = [-opening / 2.0, 0.0, HANDE_FINGER_Z - HANDE_TCP_Z]

  parts = [
    (load_mesh_o3d(os.path.join(GRIPPER_ASSET_DIR, "coupler.stl"), [0.08, 0.08, 0.08]), base_shift),
    (load_mesh_o3d(os.path.join(GRIPPER_ASSET_DIR, "hande.stl"), [0.12, 0.12, 0.12]), base_shift),
    (load_mesh_o3d(os.path.join(GRIPPER_ASSET_DIR, "finger_2-v3.0.stl"), [0.45, 0.45, 0.45]), left_tf),
    (load_mesh_o3d(os.path.join(GRIPPER_ASSET_DIR, "finger_1-v3.0.stl"), [0.45, 0.45, 0.45]), right_tf),
  ]
  return parts


def make_bbox_lineset(extents, color=(0.0, 1.0, 0.0)):
  half = np.asarray(extents, dtype=np.float32) / 2.0
  min_xyz = -half
  max_xyz = half
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
  lines = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ], dtype=np.int32)
  line_set = o3d.geometry.LineSet()
  line_set.points = o3d.utility.Vector3dVector(corners.astype(np.float64))
  line_set.lines = o3d.utility.Vector2iVector(lines)
  line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (len(lines), 1)))
  return line_set


class GraspConfigApp:
  def __init__(self, args):
    self.args = args
    self.mesh_file = args.mesh_file
    self.mesh_scale = args.mesh_scale
    self.output_file = args.output
    self.grasps = []
    self.object_mesh = None
    self.to_origin = np.eye(4, dtype=np.float32)
    self.extents = np.ones(3, dtype=np.float32) * 0.1
    self.reset_view_requested = True
    self.hande_parts = load_hande_parts(0.035)

    self.root = tk.Tk()
    self.root.title("FoundationPose Grasp Config")
    self.root.protocol("WM_DELETE_WINDOW", self.close)

    self.vis = o3d.visualization.Visualizer()
    if not self.vis.create_window("Grasp authoring view", width=1100, height=800):
      raise RuntimeError("Could not create Open3D visualization window.")
    render_options = self.vis.get_render_option()
    render_options.background_color = np.array([0.04, 0.045, 0.05], dtype=np.float64)
    render_options.mesh_show_back_face = True
    render_options.line_width = 2.0

    self.current_geometries = []
    self.gripper_opening = tk.DoubleVar(value=0.035)
    self.tx = tk.DoubleVar(value=0.0)
    self.ty = tk.DoubleVar(value=0.0)
    self.tz = tk.DoubleVar(value=0.12)
    self.current_rotation = Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix().astype(np.float32)
    self.rotation_slider_values = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
    self.suppress_rotation_callback = False
    self.rx = tk.DoubleVar(value=0.0)
    self.ry = tk.DoubleVar(value=0.0)
    self.rz = tk.DoubleVar(value=0.0)
    self.status = tk.StringVar(value="Load a CAD model to start.")
    self.scene_dirty = True

    self.build_controls()
    if args.config:
      self.load_json(args.config)
    elif self.mesh_file:
      self.load_cad(self.mesh_file, self.mesh_scale)
    self.schedule_update()

  def build_controls(self):
    frame = tk.Frame(self.root, padx=8, pady=8)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Button(frame, text="Load CAD model", command=self.pick_cad).pack(fill=tk.X)
    tk.Button(frame, text="Load JSON config", command=self.pick_json).pack(fill=tk.X)
    self.scale_entry = self.add_entry(frame, "mesh_scale", self.mesh_scale)
    tk.Button(frame, text="Apply mesh scale / reload", command=self.reload_cad).pack(fill=tk.X, pady=(2, 8))
    tk.Button(frame, text="Reset 3D view", command=self.request_view_reset).pack(fill=tk.X, pady=(0, 8))

    self.add_slider(frame, "x [m]", self.tx, -0.5, 0.5, 0.001)
    self.add_slider(frame, "y [m]", self.ty, -0.5, 0.5, 0.001)
    self.add_slider(frame, "z [m]", self.tz, -0.5, 0.5, 0.001)
    self.add_rotation_slider(frame, "local roll delta [deg]", self.rx, "rx")
    self.add_rotation_slider(frame, "local pitch delta [deg]", self.ry, "ry")
    self.add_rotation_slider(frame, "local yaw delta [deg]", self.rz, "rz")
    tk.Button(frame, text="Zero RPY sliders", command=self.zero_rotation_sliders).pack(fill=tk.X, pady=(2, 0))
    self.add_slider(frame, "gripper opening [m]", self.gripper_opening, 0.0, 0.08, 0.001)

    row = tk.Frame(frame)
    row.pack(fill=tk.X, pady=(8, 0))
    tk.Button(row, text="Add grasp", command=self.add_grasp).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(row, text="Update selected", command=self.update_selected).pack(side=tk.LEFT, fill=tk.X, expand=True)
    tk.Button(row, text="Delete selected", command=self.delete_selected).pack(side=tk.LEFT, fill=tk.X, expand=True)

    self.grasp_list = tk.Listbox(frame, height=7)
    self.grasp_list.pack(fill=tk.BOTH, expand=True, pady=(6, 2))
    self.grasp_list.bind("<<ListboxSelect>>", self.on_select_grasp)

    tk.Button(frame, text="Save JSON", command=self.save_json).pack(fill=tk.X, pady=(8, 0))
    tk.Label(frame, textvariable=self.status, anchor="w", justify=tk.LEFT, wraplength=360).pack(fill=tk.X, pady=(8, 0))

  def add_entry(self, parent, label, value):
    row = tk.Frame(parent)
    row.pack(fill=tk.X, pady=2)
    tk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
    entry = tk.Entry(row)
    entry.insert(0, str(value))
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return entry

  def add_slider(self, parent, label, variable, lo, hi, resolution):
    row = tk.Frame(parent)
    row.pack(fill=tk.X, pady=1)
    tk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
    scale = tk.Scale(row, variable=variable, from_=lo, to=hi, resolution=resolution, orient=tk.HORIZONTAL, length=250, command=lambda _value: self.mark_dirty())
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

  def add_rotation_slider(self, parent, label, variable, key):
    row = tk.Frame(parent)
    row.pack(fill=tk.X, pady=1)
    tk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
    scale = tk.Scale(
      row,
      variable=variable,
      from_=-180,
      to=180,
      resolution=1,
      orient=tk.HORIZONTAL,
      length=250,
      command=lambda _value, axis_key=key, var=variable: self.apply_local_rotation_delta(axis_key, var),
    )
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

  def apply_local_rotation_delta(self, key, variable):
    if self.suppress_rotation_callback:
      return
    new_value = float(variable.get())
    old_value = float(self.rotation_slider_values[key])
    delta_deg = new_value - old_value
    if abs(delta_deg) < 1e-9:
      return
    axis = {"rx": "x", "ry": "y", "rz": "z"}[key]
    delta_R = Rotation.from_euler(axis, delta_deg, degrees=True).as_matrix().astype(np.float32)
    self.current_rotation = normalize_rotation(self.current_rotation @ delta_R)
    self.rotation_slider_values[key] = new_value
    self.mark_dirty()

  def zero_rotation_sliders(self):
    self.suppress_rotation_callback = True
    for key, variable in (("rx", self.rx), ("ry", self.ry), ("rz", self.rz)):
      self.rotation_slider_values[key] = 0.0
      variable.set(0.0)
    self.suppress_rotation_callback = False

  def mark_dirty(self):
    self.scene_dirty = True

  def request_view_reset(self):
    self.reset_view_requested = True
    self.mark_dirty()

  def pick_cad(self):
    path = filedialog.askopenfilename(filetypes=[("Mesh files", "*.obj *.stl *.ply"), ("All files", "*.*")])
    if path:
      self.mesh_file = path
      self.reload_cad()

  def pick_json(self):
    path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
    if not path:
      return
    try:
      self.load_json(path)
    except Exception as e:
      messagebox.showerror("Load JSON failed", str(e))

  def load_json(self, path):
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    self.mesh_file = data["mesh_file"]
    self.mesh_scale = float(data["mesh_scale"])
    self.scale_entry.delete(0, tk.END)
    self.scale_entry.insert(0, str(self.mesh_scale))
    self.grasps = data.get("grasps", [])
    self.output_file = path
    self.load_cad(self.mesh_file, self.mesh_scale)
    self.refresh_grasp_list()

  def reload_cad(self):
    try:
      self.mesh_scale = float(self.scale_entry.get())
      if not np.isfinite(self.mesh_scale) or self.mesh_scale <= 0.0:
        raise RuntimeError("mesh_scale must be a positive finite number.")
      self.load_cad(self.mesh_file, self.mesh_scale)
    except Exception as e:
      messagebox.showerror("Load failed", str(e))

  def load_cad(self, path, scale):
    if not path:
      raise RuntimeError("No CAD model selected.")
    self.object_mesh, self.to_origin, self.extents = load_object_mesh(path, scale)
    self.mesh_file = path
    self.mesh_scale = scale
    self.status.set(f"Loaded {path}\nscale={scale}")
    self.reset_view_requested = True
    self.mark_dirty()

  def current_object_T_grasp(self):
    return make_transform_from_rotation(
      [self.tx.get(), self.ty.get(), self.tz.get()],
      self.current_rotation,
    )

  def set_current_transform(self, tf):
    xyz = np.asarray(tf[:3, 3], dtype=np.float32).copy()
    self.tx.set(float(xyz[0]))
    self.ty.set(float(xyz[1]))
    self.tz.set(float(xyz[2]))
    self.current_rotation = normalize_rotation(tf[:3, :3])
    self.zero_rotation_sliders()

  def add_grasp(self):
    name = simpledialog.askstring("Grasp name", "Name this grasp:", initialvalue=f"grasp_{len(self.grasps):02d}")
    if not name:
      return
    self.grasps.append(self.make_grasp_entry(name))
    self.refresh_grasp_list()

  def update_selected(self):
    idx = self.selected_index()
    if idx is None:
      return
    self.grasps[idx] = self.make_grasp_entry(self.grasps[idx]["name"])
    self.refresh_grasp_list()

  def delete_selected(self):
    idx = self.selected_index()
    if idx is None:
      return
    del self.grasps[idx]
    self.refresh_grasp_list()

  def make_grasp_entry(self, name):
    return {
      "name": name,
      "object_T_grasp": matrix_to_list(self.current_object_T_grasp()),
      "gripper_opening": float(self.gripper_opening.get()),
    }

  def selected_index(self):
    selection = self.grasp_list.curselection()
    if not selection:
      return None
    return int(selection[0])

  def on_select_grasp(self, _event):
    idx = self.selected_index()
    if idx is None:
      return
    grasp = self.grasps[idx]
    self.set_current_transform(list_to_matrix(grasp["object_T_grasp"]))
    self.gripper_opening.set(float(grasp.get("gripper_opening", self.gripper_opening.get())))
    self.mark_dirty()

  def refresh_grasp_list(self):
    self.grasp_list.delete(0, tk.END)
    for grasp in self.grasps:
      self.grasp_list.insert(tk.END, grasp["name"])

  def save_json(self):
    if not self.mesh_file:
      messagebox.showerror("Missing CAD", "Load a CAD model before saving.")
      return
    path = filedialog.asksaveasfilename(
      initialfile=os.path.basename(self.output_file),
      defaultextension=".json",
      filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    if not path:
      return
    data = {
      "version": 1,
      "mesh_file": os.path.abspath(self.mesh_file),
      "mesh_scale": float(self.mesh_scale),
      "object_frame": "oriented_bounds_center_frame",
      "source_mesh_to_object_frame": matrix_to_list(self.to_origin),
      "gripper_xml": DEFAULT_GRIPPER_XML,
      "gripper_tcp_z": HANDE_TCP_Z,
      "grasps": self.grasps,
    }
    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)
    self.output_file = path
    self.status.set(f"Saved {path}")

  def refresh_scene(self):
    for geometry in self.current_geometries:
      self.vis.remove_geometry(geometry, reset_bounding_box=False)
    self.current_geometries = []

    if self.object_mesh is not None:
      obj = trimesh_to_o3d(self.object_mesh, [0.7, 0.72, 0.76])
      self.current_geometries.append(obj)
      self.current_geometries.append(make_bbox_lineset(self.extents))

    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    self.current_geometries.append(object_frame)

    object_T_grasp = self.current_object_T_grasp()
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
    grasp_frame.transform(object_T_grasp)
    self.current_geometries.append(grasp_frame)

    self.hande_parts = load_hande_parts(float(self.gripper_opening.get()))
    for part, local_tf in self.hande_parts:
      mesh = transform_mesh(part, object_T_grasp @ local_tf)
      self.current_geometries.append(mesh)

    for grasp in self.grasps:
      tf = list_to_matrix(grasp["object_T_grasp"])
      frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.035)
      frame.transform(tf)
      self.current_geometries.append(frame)

    reset = self.reset_view_requested
    for i, geometry in enumerate(self.current_geometries):
      self.vis.add_geometry(geometry, reset_bounding_box=reset)
    if reset:
      self.vis.reset_view_point(True)
      self.apply_camera_angle()
      self.reset_view_requested = False
    self.scene_dirty = False
    self.vis.poll_events()
    self.vis.update_renderer()

  def apply_camera_angle(self):
    view = self.vis.get_view_control()
    view.set_front([0.7, -0.7, -0.45])
    view.set_up([0.0, 0.0, 1.0])

  def schedule_update(self):
    if self.scene_dirty:
      self.refresh_scene()
    self.vis.poll_events()
    self.vis.update_renderer()
    self.root.after(80, self.schedule_update)

  def close(self):
    self.vis.destroy_window()
    self.root.destroy()

  def run(self):
    self.root.mainloop()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mesh_file", type=str, default=None)
  parser.add_argument("--mesh_scale", type=float, default=1.0)
  parser.add_argument("--config", type=str, default=None)
  parser.add_argument("--output", type=str, default="grasp_config.json")
  args = parser.parse_args()
  GraspConfigApp(args).run()
