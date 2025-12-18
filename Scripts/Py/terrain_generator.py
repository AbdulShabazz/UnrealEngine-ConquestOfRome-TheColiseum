# watershed_soft_masks.py
# Requires: pip install numpy pillow
# Input: island_6km_8129_height_16bit.png (16-bit grayscale PNG)

import numpy as np
from PIL import Image

IN_HEIGHT = "island_6km_8129_height_16bit.png"

ANALYSIS_RES = 2049
OUTPUT_RES = 8129

# Biome indices (for reference)
# 0 Tundra, 1 Taiga, 2 Grassland, 3 Badlands,
# 4 DesertPlateau, 5 Dunes, 6 Jungle, 7 Forest, 8 Swamp

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def load_height_norm(path: str) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.float32)
    # Supports 16-bit grayscale PNG (0..65535)
    if arr.max() > 1.5:
        arr = arr / 65535.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def resize_arr(arr: np.ndarray, size: int, resample=Image.BILINEAR) -> np.ndarray:
    im = Image.fromarray((arr * 65535.0).astype(np.uint16))
    im2 = im.resize((size, size), resample=resample)
    return (np.array(im2, dtype=np.float32) / 65535.0).clip(0.0, 1.0)

def flow_accum_d4(h: np.ndarray, iters: int = 18) -> np.ndarray:
    """
    Lightweight D4 accumulation. Good enough to approximate basins/rivers at 2049².
    """
    hp = np.pad(h, 1, mode="edge")
    n = hp[:-2, 1:-1]
    s = hp[2:,  1:-1]
    w = hp[1:-1, :-2]
    e = hp[1:-1, 2:]
    neigh = np.stack([n, s, w, e], axis=0)
    idx = np.argmin(neigh, axis=0).astype(np.uint8)  # 0=N,1=S,2=W,3=E

    acc = np.ones_like(h, dtype=np.float32)
    for _ in range(iters):
        nxt = np.ones_like(acc, dtype=np.float32)
        # Push current acc “downstream”
        nxt += np.roll(acc * (idx == 0), -1, axis=0)  # to N (dy=-1)
        nxt += np.roll(acc * (idx == 1),  1, axis=0)  # to S
        nxt += np.roll(acc * (idx == 2), -1, axis=1)  # to W
        nxt += np.roll(acc * (idx == 3),  1, axis=1)  # to E
        acc = np.maximum(acc, nxt)

    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    return acc

def ridge_map(h: np.ndarray) -> np.ndarray:
    """
    Simple ridge/crest proxy: positive difference vs local minimum of 4-neighbors.
    """
    n = np.roll(h, -1, axis=0)
    s = np.roll(h,  1, axis=0)
    w = np.roll(h, -1, axis=1)
    e = np.roll(h,  1, axis=1)
    local_min = np.minimum(np.minimum(n, s), np.minimum(w, e))
    r = np.clip(h - local_min, 0.0, None)
    return r / (r.max() + 1e-6)

def normalize_weights(ws: np.ndarray) -> np.ndarray:
    """
    ws: (B,H,W) float32
    """
    s = ws.sum(axis=0, keepdims=True) + 1e-8
    return ws / s

def pack_rgba(weights_4: np.ndarray) -> Image.Image:
    """
    weights_4: (4,H,W) in [0,1]
    """
    rgba = np.clip(weights_4 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.transpose(rgba, (1, 2, 0))  # H,W,4
    return Image.fromarray(rgba, mode="RGBA")

def save_aux_gray(arr01: np.ndarray, path: str, out_res: int):
    im = Image.fromarray((np.clip(arr01, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
    if out_res != im.size[0]:
        im = im.resize((out_res, out_res), resample=Image.BILINEAR)
    im.save(path, compress_level=9)

def main():
    # Load 8129² height and downsample to 2049² for watershed analysis
    h_hi = load_height_norm(IN_HEIGHT)
    h = resize_arr(h_hi, ANALYSIS_RES, resample=Image.BILINEAR)

    H = ANALYSIS_RES
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
    nx = (xx - (H - 1) / 2.0) / ((H - 1) / 2.0)   # -1..1
    ny = (yy - (H - 1) / 2.0) / ((H - 1) / 2.0)   # -1..1

    # Proxies (north cold, south tropical; west windward)
    west_bias = np.clip((-nx + 1.0) * 0.5, 0.0, 1.0)
    northness = np.clip((-ny + 1.0) * 0.5, 0.0, 1.0)  # 1 north
    southness = 1.0 - northness

    # Accumulation and wetness
    acc = flow_accum_d4(h, iters=18)
    wet = np.clip(acc * (0.6 + 0.4 * west_bias), 0.0, 1.0)

    # Ridge auxiliary
    ridge = ridge_map(h)

    # Soft membership functions (all in [0,1])
    # Elevation bands
    elev = h
    high = smoothstep(0.60, 0.80, elev)
    mid  = smoothstep(0.25, 0.55, elev) * (1.0 - smoothstep(0.55, 0.70, elev))
    low  = 1.0 - smoothstep(0.25, 0.35, elev)

    # Climate bands
    cold = smoothstep(0.10, 0.30, northness)
    trop = smoothstep(0.10, 0.30, southness)
    wet_hi = smoothstep(0.55, 0.75, wet)
    wet_mid = smoothstep(0.40, 0.60, wet) * (1.0 - smoothstep(0.60, 0.72, wet))
    dry = 1.0 - smoothstep(0.28, 0.38, wet)

    # Incision/canyon proxy: high accumulation in mid elevation (not at highest peaks)
    incision = smoothstep(0.60, 0.82, acc) * mid

    # --- Per-biome raw weights (before normalization) ---
    w = np.zeros((9, H, H), dtype=np.float32)

    # 0 Tundra: cold + high elevation
    w[0] = cold * smoothstep(0.72, 0.85, elev)

    # 1 Taiga: cold + upper-mid elevation + some wetness
    w[1] = cold * smoothstep(0.58, 0.75, elev) * (0.5 + 0.5 * wet_mid)

    # 6 Jungle: tropical + wet + mid/high foothills (avoid very low swamp basins)
    w[6] = trop * wet_hi * smoothstep(0.42, 0.55, elev)

    # 8 Swamp: tropical + very wet + low elevation + low ridge (avoid crests)
    w[8] = trop * smoothstep(0.62, 0.80, wet) * low * (1.0 - smoothstep(0.15, 0.35, ridge))

    # 7 Forest: moderate wetness + mid elevation (broad)
    w[7] = wet_mid * smoothstep(0.35, 0.65, elev)

    # 3 Badlands: incision + dry-ish (stronger in rain-shadow) + not too high
    w[3] = incision * (0.55 + 0.45 * (1.0 - west_bias)) * (0.6 + 0.4 * dry)

    # 4 DesertPlateau: dry + mid elevation + leeward
    w[4] = dry * smoothstep(0.30, 0.55, elev) * (0.55 + 0.45 * (1.0 - west_bias))

    # 5 Dunes: very dry + low elevation + leeward
    w[5] = (1.0 - smoothstep(0.32, 0.42, wet)) * low * (0.60 + 0.40 * (1.0 - west_bias))

    # 2 Grassland: residual temperate/open (mid wetness or mid elev) and everywhere else
    w[2] = (0.35 + 0.65 * (1.0 - np.abs(ny))) * (0.35 + 0.65 * (1.0 - dry)) * (0.5 + 0.5 * mid)

    # Ensure nonzero everywhere (numerical stability)
    w += 1e-4

    # Normalize to probabilities
    w = normalize_weights(w)

    # Save aux maps (upsampled to 8129)
    save_aux_gray(acc, "aux_flow_accum_8129.png", OUTPUT_RES)
    save_aux_gray(wet, "aux_wetness_8129.png", OUTPUT_RES)
    save_aux_gray(ridge, "aux_ridge_8129.png", OUTPUT_RES)

    # Pack RGBA groups at 2049 then upscale to 8129
    def save_pack(group, out_name):
        im = pack_rgba(w[group])
        im = im.resize((OUTPUT_RES, OUTPUT_RES), resample=Image.BILINEAR)
        im.save(out_name, compress_level=9)

    save_pack([0, 1, 2, 3], "watershed_soft_rgba_0_3_8129.png")
    save_pack([4, 5, 6, 7], "watershed_soft_rgba_4_7_8129.png")
    # Swamp only in R
    swamp_rgba = np.zeros((4, H, H), dtype=np.float32)
    swamp_rgba[0] = w[8]
    im = pack_rgba(swamp_rgba).resize((OUTPUT_RES, OUTPUT_RES), resample=Image.BILINEAR)
    im.save("watershed_soft_rgba_8_8129.png", compress_level=9)

    print("Done.")
    print("Outputs:")
    print("  watershed_soft_rgba_0_3_8129.png")
    print("  watershed_soft_rgba_4_7_8129.png")
    print("  watershed_soft_rgba_8_8129.png")
    print("  aux_flow_accum_8129.png")
    print("  aux_wetness_8129.png")
    print("  aux_ridge_8129.png")

if __name__ == "__main__":
    main()
