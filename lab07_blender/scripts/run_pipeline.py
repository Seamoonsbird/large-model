"""一键流水线：Blender 导出控制图 -> sd-cli 生成写实图。

用法:
    python scripts/run_pipeline.py --name test512 --res 512 --steps 4 --cfg 1.2
"""
import argparse
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLEND = ROOT / "assets" / "demo_white_model.blend"
EXPORT_SCRIPT = ROOT / "blender" / "export_control_maps.py"
MODEL_SD = ROOT / "models" / "sd_turbo-q4_0.gguf"
CONTROLNET = ROOT / "models" / "control_depth_sd21-fp16.gguf"
SDCLI = ROOT / "vendor" / "sdcpp-ext" / "sd-cli.exe"
OUT_ROOT = ROOT / "output"

DEFAULT_PROMPT = ("photorealistic render, studio soft lighting, PBR material, "
                  "preserve original 3d geometry, sharp")
DEFAULT_NEG = "deformed geometry, perspective changed"


def run(cmd, cwd=None):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_rgb(path):
    """sd.cpp 的 control image 需要至少 3 通道"""
    from PIL import Image
    with Image.open(path) as img:
        if img.mode != "RGB":
            img.convert("RGB").save(path)


def latest(pattern):
    files = glob.glob(str(pattern))
    return Path(max(files, key=os.path.getmtime)) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test512")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--control-strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--control-image", default=None, help="跳过 Blender，直接用已有深度图")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    args = ap.parse_args()

    OUT_ROOT.mkdir(exist_ok=True)
    blend_exe = os.environ.get("BLENDER_EXE", "blender")

    t0 = time.time()

    # 1) Blender 导出控制图（或直接复用已有深度图）
    if args.control_image:
        depth_path = Path(args.control_image)
    else:
        run([blend_exe, "-b", str(BLEND), "-P", str(EXPORT_SCRIPT), "--",
             "--outdir", str(OUT_ROOT), "--name", args.name, "--res", str(args.res)])
        src = latest(OUT_ROOT / ("%s_depth_*.png" % args.name))
        depth_path = OUT_ROOT / ("%s_depth.png" % args.name)
        shutil.copyfile(src, depth_path)
    ensure_rgb(depth_path)
    t_blender = time.time() - t0

    # 2) sd-cli 生成
    photo_path = OUT_ROOT / ("%s_photoreal.png" % args.name)
    t1 = time.time()
    sd_cmd = [
        str(SDCLI),
        "-m", str(MODEL_SD),
        "-p", args.prompt,
        "-n", args.negative,
        "--control-net", str(CONTROLNET),
        "--control-image", str(depth_path.resolve()),
        "--control-strength", str(args.control_strength),
        "-W", str(args.res), "-H", str(args.res),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg),
        "--sampling-method", "euler_a",
        "-s", str(args.seed),
        "-t", str(args.threads),
        "-o", str(photo_path),
    ]
    run(sd_cmd, cwd=str(SDCLI.parent))
    t_sd = time.time() - t1

    print("\n=== 耗时 ===")
    print(f"Blender 导出: {t_blender:.1f}s")
    print(f"sd-cli 生成 : {t_sd:.1f}s")
    print(f"总耗时      : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
