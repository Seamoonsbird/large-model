"""生成演示白模场景：球、圆环、倒角立方体、圆柱 + 地面 + 三点布光 + 63mm 相机。"""
import bpy
from mathutils import Vector

OUT = "assets/demo_white_model.blend"


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


def add_area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy, data.size = energy, size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def make_material(name, color, roughness=0.9):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    return m


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    body_mat = make_material("body", (0.9, 0.9, 0.9))
    floor_mat = make_material("floor", (0.55, 0.55, 0.55))

    # 地面
    bpy.ops.mesh.primitive_plane_add(size=24, location=(0, 0, 0))
    floor = bpy.context.object
    floor.data.materials.append(floor_mat)

    # 几何体：球、圆环、倒角立方体、圆柱
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.62, segments=64, location=(-0.95, 0, 0.62))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    sphere.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_torus_add(major_radius=0.42, minor_radius=0.16, location=(0.15, 0, 0.55))
    torus = bpy.context.object
    bpy.ops.object.shade_smooth()
    torus.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cube_add(location=(1.05, 0, 0.45))
    cube = bpy.context.object
    cube.scale = (0.5, 0.5, 0.45)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.modifier_add(type="BEVEL")
    bpy.context.object.modifiers["Bevel"].width = 0.12
    bpy.context.object.modifiers["Bevel"].segments = 4
    bpy.ops.object.modifier_apply(modifier="Bevel")
    cube.data.materials.append(body_mat)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.32, depth=1.1, location=(1.95, 0, 0.55))
    cyl = bpy.context.object
    bpy.ops.object.shade_smooth()
    cyl.data.materials.append(body_mat)

    # 相机 63mm
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 63
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (4.8, -6.0, 2.8)
    look_at(camera, (0, 0, 0.55))
    scene.camera = camera

    # 三点布光
    add_area_light("Key", (3.0, -3.2, 4.2), 300, 3.5, (0, 0, 0.5))
    add_area_light("Fill", (-3.6, -2.8, 2.2), 80, 5.0, (0, 0, 0.6))
    add_area_light("Rim", (0.5, 3.5, 3.2), 220, 2.2, (0, 0, 0.8))

    bpy.ops.wm.save_as_mainfile(filepath=OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
