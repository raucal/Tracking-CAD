"""Executed by Blender's Python; no external Python packages are required here."""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
spec = json.loads(Path(args.spec).read_text())
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = True
scene.render.threads_mode = "FIXED"
scene.render.threads = 8
scene.render.resolution_x = scene.render.resolution_y = 256
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.view_layers[0].use_pass_position = True
scene.use_nodes = True
nodes = scene.node_tree.nodes
nodes.clear()
render_node = nodes.new('CompositorNodeRLayers')
composite = nodes.new('CompositorNodeComposite')
scene.node_tree.links.new(render_node.outputs['Image'], composite.inputs['Image'])
position_output = nodes.new('CompositorNodeOutputFile')
position_output.base_path = str(output)
position_output.format.file_format = 'OPEN_EXR'
position_output.format.color_depth = '32'
position_output.file_slots[0].path = 'position_'
scene.node_tree.links.new(render_node.outputs['Position'], position_output.inputs[0])
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "Medium High Contrast"
world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.25, 0.25, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1
scene.world = world
geometry = bpy.data.meshes.new("cad_mm")
geometry.from_pydata(spec["vertices"], [], spec["faces"])
geometry.update()
obj = bpy.data.objects.new("CAD", geometry)
bpy.context.collection.objects.link(obj)
obj.scale = (spec["scale"],) * 3
material = bpy.data.materials.new("CAD gray")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1)
obj.data.materials.append(material)
camera_data = bpy.data.cameras.new("Camera")
camera_data.lens = 50
camera_data.sensor_width = 36
camera_data.clip_start = 0.01
camera_data.clip_end = 10
camera = bpy.data.objects.new("Camera", camera_data)
bpy.context.collection.objects.link(camera)
scene.camera = camera
light_data = bpy.data.lights.new("Light", "POINT")
light_data.energy = 1000
light = bpy.data.objects.new("Light", light_data)
bpy.context.collection.objects.link(light)
for i, pose in enumerate(spec["poses"]):
    scene.frame_set(i)
    camera.matrix_world = Matrix(pose)
    light.location = tuple(2.5 * pose[j][3] for j in range(3))
    scene.render.filepath = str(output / f"rgba_{i}.png")
    bpy.ops.render.render(write_still=True)
    print(f"BLENDER_VIEW {i + 1}/42", flush=True)
