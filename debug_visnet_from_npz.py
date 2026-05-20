import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from matplotlib.colors import hsv_to_rgb

# 让 Python 能找到 src/mazebots 下的模块
sys.path.append("src/mazebots")

import config as cfg
from model import VisNet


# =========================
# 可改参数
# =========================
NPZ_PATH = "data/rec_00.npz"
T_IDX = 0          # 第几个时间步
B_IDX = 0          # 第几个机器人
OUT_DIR = f"debug_vis/t{T_IDX:03d}_b{B_IDX:02d}"


def save_gray(arr: np.ndarray, path: str):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_rgb(arr: np.ndarray, path: str):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def colorize_index_map(idx: np.ndarray, num_classes: int) -> np.ndarray:
    """
    给类别图一个简单伪彩色，便于看。
    不依赖 config 里的具体颜色定义，稳一点。
    """
    idx = idx.astype(np.int64)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        palette[i] = np.array([
            (37 * i) % 256,
            (67 * i) % 256,
            (97 * i) % 256
        ], dtype=np.uint8)
    return palette[idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------
    # 1) 读取 npz
    # -------------------------
    d = np.load(NPZ_PATH)
    img = d["img"]   # (T, N, 5, 48, 96)
    vec = d["vec"]   # (T, N, 126)

    print("img shape:", img.shape)
    print("vec shape:", vec.shape)

    x_all = img[T_IDX, B_IDX]   # (5, 48, 96)
    vec_one = vec[T_IDX, B_IDX]

    # 前4通道是 VisNet 输入：HSV(3) + depth(1)
    x_in = x_all[:4]
    hsv = np.transpose(x_all[:3], (1, 2, 0))   # (48,96,3)
    depth_in = x_all[3]
    seg_gt = x_all[4]

    # 真值位置
    true_loc = vec_one[cfg.BOT_POS_SLICE]
    print("true_loc:", true_loc)

    # -------------------------
    # 2) 保存输入可视化
    # -------------------------
    rgb_in = hsv_to_rgb(np.clip(hsv, 0, 1))
    rgb_in = (rgb_in * 255).astype(np.uint8)

    save_rgb(rgb_in, os.path.join(OUT_DIR, "input_rgb_from_hsv.png"))
    save_gray(depth_in, os.path.join(OUT_DIR, "input_depth.png"))
    save_gray(seg_gt, os.path.join(OUT_DIR, "input_seg_gt_gray.png"))

    # 伪彩色分割图
    seg_gt_idx = seg_gt.astype(np.int64)
    seg_gt_color = colorize_index_map(seg_gt_idx, cfg.N_ENT_CLASSES)
    save_rgb(seg_gt_color, os.path.join(OUT_DIR, "input_seg_gt_color.png"))

    # -------------------------
    # 3) 加载 VisNet
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = VisNet().to(device)
    ckpt_path = os.path.join(cfg.ASSET_DIR, "visnet.pt")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # (1, 4, 48, 96)
    x_tensor = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0).to(device)

    # -------------------------
    # 4) 前向
    # -------------------------
    with torch.no_grad():
        img_out, feat, pred_obj, pred_loc = model(x_tensor)

    pred_obj = pred_obj[0].detach().cpu().numpy()
    pred_loc = pred_loc[0].detach().cpu().numpy()

    print("pred_obj:", pred_obj)
    print("pred_loc:", pred_loc)

    # -------------------------
    # 5) 解析重建图
    # -------------------------
    clr_logits, ent_logits, dep_out = img_out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

    clr_idx = clr_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
    ent_idx = ent_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
    dep_rec = dep_out[0, 0].detach().cpu().numpy()

    # 保存灰度版
    save_gray(clr_idx, os.path.join(OUT_DIR, "recon_clr_gray.png"))
    save_gray(ent_idx, os.path.join(OUT_DIR, "recon_ent_gray.png"))
    save_gray(dep_rec, os.path.join(OUT_DIR, "recon_depth.png"))

    # 保存伪彩色版
    clr_color = colorize_index_map(clr_idx, cfg.N_CLR_CLASSES)
    ent_color = colorize_index_map(ent_idx, cfg.N_ENT_CLASSES)
    save_rgb(clr_color, os.path.join(OUT_DIR, "recon_clr_color.png"))
    save_rgb(ent_color, os.path.join(OUT_DIR, "recon_ent_color.png"))

    # -------------------------
    # 6) 保存数值信息
    # -------------------------
    info = {
        "npz_path": NPZ_PATH,
        "time_index": T_IDX,
        "bot_index": B_IDX,
        "true_loc": true_loc.tolist(),
        "pred_loc": pred_loc.tolist(),
        "pred_obj": pred_obj.tolist(),
    }

    with open(os.path.join(OUT_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()