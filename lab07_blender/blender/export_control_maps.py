"""无头导出 depth / normal / clay 三张图。

用法（Blender 内）:
    blender -b scene.blend -P export_control_maps.py -- --outdir output --name demo --res 512
"""
import os
import sys
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    outdir, name, res, samples = "output", "demo", 512, 64
    i = 0
    while i < len(argv):
        if argv[i] == "--outdir": outdir = argv[i + 1]; i += 2
        elif argv[i] == "--name": name = argv[i + 1]; i += 2
        elif argv[i] == "--res": res = int(argv[i + 1]); i += 2
        elif argv[i] == "--samples": samples = int(argv[i + 1]); i += 2
        else: i += 1
    return outdir, name, res, samples


def scene_depth_range(scene, camera):
    """用相机视锥内可见顶点估算深度范围（2%/98% 百分位），避免远处地面拉爆动态范围。"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    distances = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        for vert in obj_eval.data.vertices:
            world_co = obj_eval.matrix_world @ vert.co
            ndc = world_to_camera_view(scene, camera, world_co)
            if ndc.z <= 0 or not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
                continue
            distances.append((world_co - cam_loc).length)
    if not distances:
        return 0.1, 100.0
    near = float(np.percentile(distances, 2))
    far = float(np.percentile(distances, 98))
    return near, (far if far > near else near + 1.0)


def make_clay_material():
    m = bpy.data.materials.new("clay_override")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.92, 0.92, 0.92, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Metallic"].default_value = 0.0
    return m


def setup_compositor(scene, outdir, name):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")

    # depth: mist 0..1 (near..far) -> invert -> 近白远黑
    invert = tree.nodes.new("CompositorNodeInvert")
    invert.inputs["Fac"].default_value = 1.0

    # normal: n(-1..1) * 0.5 + 0.5
    mul = tree.nodes.new("CompositorNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    add = tree.nodes.new("CompositorNodeMixRGB")
    add.blend_type = "ADD"
    add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)

    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = outdir
    depth_out.file_slots[0].path = name + "_depth_"
    depth_out.format.file_format = "PNG"
    depth_out.format.color_mode = "RGB"

    normal_out = tree.nodes.new("CompositorNodeOutputFile")
    normal_out.base_path = outdir
    normal_out.file_slots[0].path = name + "_normal_"
    normal_out.format.file_format = "PNG"
    normal_out.format.color_mode = "RGB"

    depth_in = invert.inputs["Color"] if "Color" in invert.inputs else invert.inputs["Image"]
    tree.links.new(rl.outputs["Mist"], depth_in)
    tree.links.new(invert.outputs[0], depth_out.inputs[0])
    tree.links.new(rl.outputs["Normal"], mul.inputs[1])
    tree.links.new(mul.outputs[0], add.inputs[1])
    tree.links.new(add.outputs[0], normal_out.inputs[0])


def main():
    outdir, name, res, samples = parse_args()
    os.makedirs(outdir, exist_ok=True)

    scene = bpy.context.scene
    camera = scene.camera

    # 渲染参数
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    # Mist pass 范围：用可见顶点 2%/98% 百分位
    near, far = scene_depth_range(scene, camera)
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)
    scene.world.color = (0.05, 0.05, 0.05)
    scene.render.film_transparent = False

    mist = scene.world.node_tree.nodes.new("ShaderNodeTexCoord")
    # 通过 world 设置 mist 起止
    try:
        scene.world.mist_settings.use_mist = True
        scene.world.mist_settings.start = near
        scene.world.mist_settings.depth = far - near
        scene.world.mist_settings.falloff = "LINEAR"
    except Exception as e:
        print("mist setting warn:", e)

    view_layer = scene.view_layers[0]
    view_layer.use_pass_mist = True
    view_layer.use_pass_normal = True

    # ---- clay 参考图：材质覆盖为纯白，单独渲染一次 ----
    clay_mat = make_clay_material()
    view_layer.material_override = clay_mat
    scene.render.filepath = os.path.join(outdir, name + "_clay.png")
    bpy.ops.render.render(write_still=True)

    # ---- 恢复材质，开合成器导出 depth / normal ----
    view_layer.material_override = None
    setup_compositor(scene, outdir, name)
    bpy.ops.render.render(write_still=True)

    print("done:", outdir, name, res)


if __name__ == "__main__":
    main()
