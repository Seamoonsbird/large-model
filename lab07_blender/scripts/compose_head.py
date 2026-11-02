"""把正面头像照片融合进生成的人物图（拓展项）。

链路：白底软抠图 -> 椭圆限制头框 -> 预乘 alpha 缩放 -> 亮度对齐 -> Poisson 梯度域融合 -> 接触阴影 + 颗粒匹配。

用法:
    python scripts/compose_head.py --base output/headfig_photoreal.png \
        --head C:/path/face.png --depth output/headfig_depth.png \
        --out output/headfig_face.png
"""
import argparse
import cv2
import numpy as np


def head_alpha(rgb, face, ellipse_soft=3.0):
    """白底软抠图，再用椭圆限制成只留头部与少量脖子。"""
    key = np.clip((255.0 - rgb.min(axis=2) - 6.0) / 18.0, 0.0, 1.0)

    left, top, right, chin = face
    h, w = rgb.shape[:2]
    left, right = left * w, right * w
    top, chin = top * h, chin * h
    cx, cy = (left + right) / 2.0, (top + chin) / 2.0 + (chin - top) * 0.03
    rx, ry = (right - left) / 2.0 * 1.05, (chin - top) / 2.0 * 1.06

    emask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.ellipse(emask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
    emask = cv2.GaussianBlur(emask, (0, 0), ellipse_soft).astype(np.float32) / 255.0

    alpha = np.clip(key * emask, 0.0, 1.0)
    hard = (alpha > 0.55).astype(np.uint8)
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.2)


def resize_premul(rgb, alpha, size):
    """先乘 alpha 再缩放，避免白底像素在边缘被插值成亮边。"""
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    p = cv2.resize(rgb.astype(np.float32) * alpha[..., None], size,
                   interpolation=cv2.INTER_AREA)
    safe = np.maximum(a[..., None], 1e-3)
    un = np.where(a[..., None] > 1e-3, p / safe, 0.0)
    return np.clip(un, 0, 255).astype(np.uint8), a, p


def depth_head_box(depth_gray, thresh=0.15, grow=(0.13, 0.14, 0.13, 0.06)):
    """由深度图定位头部框（近处最亮区域的上半部分）。"""
    h, w = depth_gray.shape
    mask = (depth_gray > thresh).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, w, h)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # 只取上半部分作为头部
    y1 = y0 + int((y1 - y0) * 0.55)
    gx0, gy0, gx1, gy1 = grow
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - bw * gx0)); x1 = min(w, int(x1 + bw * gx1))
    y0 = max(0, int(y0 - bh * gy0)); y1 = min(h, int(y1 + bh * gy1))
    return (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face", default="0.16,0.02,0.85,0.71",
                    help="left,top,right,chin (比例)")
    ap.add_argument("--box-grow", default="0.13,0.14,0.13,0.06")
    ap.add_argument("--clone", type=float, default=0.6, help="Poisson 融合占比")
    ap.add_argument("--shadow", type=float, default=0.35)
    ap.add_argument("--grain", type=float, default=1.0)
    args = ap.parse_args()

    face = [float(x) for x in args.face.split(",")]
    grow = [float(x) for x in args.box_grow.split(",")]

    base = cv2.imread(args.base)
    head_rgb = cv2.cvtColor(cv2.imread(args.head), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    h, w = base.shape[:2]
    box = depth_head_box(depth, grow=grow)
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0

    # 抠图 + 预乘 alpha 缩放
    alpha = head_alpha(head_rgb, face)
    face_rgb, a_res, _ = resize_premul(head_rgb, alpha, (bw, bh))

    # 亮度对齐：把头像均值亮度对齐到基图头框区域
    base_crop = base[by0:by1, bx0:bx1].astype(np.float32)
    mask3 = np.stack([a_res] * 3, axis=-1)
    face_float = face_rgb.astype(np.float32)
    base_mean = (base_crop * mask3).sum() / max(mask3.sum(), 1e-3)
    face_mean = (face_float * mask3).sum() / max(mask3.sum(), 1e-3)
    face_float = face_float * (base_mean / max(face_mean, 1e-3))
    face_float = np.clip(face_float, 0, 255)

    # 预乘 alpha 贴入
    out = base.astype(np.float32).copy()
    roi = out[by0:by1, bx0:bx1]
    roi = roi * (1 - mask3) + face_float * mask3
    out[by0:by1, bx0:bx1] = roi

    # Poisson 梯度域融合（NORMAL_CLONE）平滑过渡
    if args.clone > 0:
        face_bgr = cv2.cvtColor(face_float.astype(np.uint8), cv2.COLOR_RGB2BGR)
        base_bgr = out.astype(np.uint8)
        center = (int((bx0 + bx1) / 2), int((by0 + by1) / 2))
        mask_u8 = (a_res * 255).astype(np.uint8)
        try:
            blended = cv2.seamlessClone(face_bgr, base_bgr, mask_u8, center,
                                         cv2.NORMAL_CLONE)
            out = out * (1 - args.clone) + blended.astype(np.float32) * args.clone
        except cv2.error as e:
            print("seamlessClone skip:", e)

    # 接触阴影（下巴与脖子）
    if args.shadow > 0:
        shadow_h = int(bh * 0.15)
        shadow_roi = out[by1:by1 + shadow_h, bx0:bx1]
        grad = np.linspace(args.shadow, 0, shadow_h).reshape(-1, 1, 1)
        shadow_roi *= (1 - grad)
        out[by1:by1 + shadow_h, bx0:bx1] = shadow_roi

    # 颗粒噪声匹配
    if args.grain > 0:
        noise = np.random.normal(0, 2.0 * args.grain, out.shape)
        out = np.clip(out + noise, 0, 255)

    cv2.imwrite(args.out, out.astype(np.uint8))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
