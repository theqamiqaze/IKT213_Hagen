import cv2
import numpy as np
from pathlib import Path
import time
import argparse

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img

# ---------- light pre-processing ----------
def apply_clahe(gray):
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# ---------- tutorial-style helpers ----------
def verdict_from_good(good_count, threshold):
    return "MATCH" if good_count >= threshold else "NO MATCH"

def ratio_filter(knn_matches, ratio):
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def draw_and_save(img1, img2, kp1, kp2, matches, out_path: Path, title: str, max_lines=50):
    matches = sorted(matches, key=lambda m: m.distance)[:max_lines]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # Readable title (black background)
    Himg, Wimg = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, Wimg / 1600.0)
    thick = max(2, int(Wimg / 900))
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    pad = 10
    cv2.rectangle(vis, (10, 10), (10 + tw + 2*pad, 10 + th + 2*pad), (0,0,0), -1)
    cv2.putText(vis, title, (10 + pad, 10 + th + pad), font, scale, (255,255,255), thick, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

# ---------- pipelines ----------
def sift_flann(img1, img2):
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    t0 = time.perf_counter()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return 0, (time.perf_counter() - t0)*1000, k1, k2, []
    knn = flann.knnMatch(d1, d2, k=2)
    good = ratio_filter(knn, 0.75)
    t1 = time.perf_counter()
    return len(good), (t1 - t0)*1000, k1, k2, good

def orb_bf(img1, img2):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    t0 = time.perf_counter()
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return 0, (time.perf_counter() - t0)*1000, k1, k2, []
    knn = bf.knnMatch(d1, d2, k=2)
    good = ratio_filter(knn, 0.8)
    t1 = time.perf_counter()
    return len(good), (t1 - t0)*1000, k1, k2, good

# ---------- dataset runners ----------
def run_fingerprint_sets(root: Path, vis_dir: Path, max_lines=50,
                         sift_thresh=30, orb_thresh=20):
    """
    Process fingerprint dataset where each immediate subfolder contains a pair:
      root/
        same_1/ (two images)
        different_1/ (two images)
        ...
    Decision is based purely on *good* match counts.
    """
    if not root.exists():
        print(f"[FINGERPRINTS] Folder not found: {root}")
        return

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"[FINGERPRINTS] No subfolders in {root}")
        return

    print(f"\nFINGERPRINTS: {root}  (processing {len(subdirs)} sets)")
    for sd in subdirs:
        imgs = sorted(list_images(sd))
        if len(imgs) < 2:
            print(f"- {sd.name}: needs at least 2 images, found {len(imgs)} — skipping")
            continue

        a, b = imgs[0], imgs[1]
        img1 = apply_clahe(read_gray(a))
        img2 = apply_clahe(read_gray(b))

        # SIFT + FLANN
        s_good_count, s_ms, s_k1, s_k2, s_good = sift_flann(img1, img2)
        s_verdict = verdict_from_good(s_good_count, sift_thresh)

        # ORB + BF
        o_good_count, o_ms, o_k1, o_k2, o_good = orb_bf(img1, img2)
        o_verdict = verdict_from_good(o_good_count, orb_thresh)

        print(f"- {sd.name}: {a.name} vs {b.name}")
        print(f"  SIFT+FLANN -> good:{s_good_count:4d}, time:{s_ms:7.1f} ms, {s_verdict}")
        print(f"  ORB+BF    -> good:{o_good_count:4d}, time:{o_ms:7.1f} ms, {o_verdict}")

        # visuals (lines only; no homography box)
        out_sift = vis_dir / f"FINGERPRINTS_{sd.name}_SIFT_{a.stem}_VS_{b.stem}.jpg"
        draw_and_save(img1, img2, s_k1, s_k2, s_good, out_sift,
                      f"SIFT+FLANN  good={s_good_count}  ({s_verdict})", max_lines=max_lines)

        out_orb  = vis_dir / f"FINGERPRINTS_{sd.name}_ORB_{a.stem}_VS_{b.stem}.jpg"
        draw_and_save(img1, img2, o_k1, o_k2, o_good, out_orb,
                      f"ORB+BF      good={o_good_count}  ({o_verdict})", max_lines=max_lines)

def run_on_folder(folder: Path, label: str, vis_dir: Path, limit_pairs: int = 0,
                  max_lines=50, sift_thresh=80, orb_thresh=25):
    """
    Generic folder runner (compares all pairs).
    """
    files = list_images(folder)
    files.sort()
    if len(files) < 2:
        print(f"[{label}] Need at least two images in {folder}")
        return
    print(f"\n{label}: {folder}")
    count = 0
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a, b = files[i], files[j]

            img1 = apply_clahe(read_gray(a))
            img2 = apply_clahe(read_gray(b))

            s_good_count, s_ms, s_k1, s_k2, s_good = sift_flann(img1, img2)
            o_good_count, o_ms, o_k1, o_k2, o_good = orb_bf(img1, img2)

            s_verdict = verdict_from_good(s_good_count, sift_thresh)
            o_verdict = verdict_from_good(o_good_count, orb_thresh)

            print(f"- Pair: {a.name} vs {b.name}")
            print(f"  SIFT+FLANN -> good: {s_good_count:4d}, time: {s_ms:7.1f} ms, {s_verdict}")
            print(f"  ORB+BF    -> good: {o_good_count:4d}, time: {o_ms:7.1f} ms, {o_verdict}")

            out_sift = vis_dir / f"{label}_SIFT_{a.stem}_VS_{b.stem}.jpg"
            draw_and_save(img1, img2, s_k1, s_k2, s_good, out_sift,
                          f"SIFT+FLANN  good={s_good_count}  ({s_verdict})", max_lines=max_lines)

            out_orb = vis_dir / f"{label}_ORB_{a.stem}_VS_{b.stem}.jpg"
            draw_and_save(img1, img2, o_k1, o_k2, o_good, out_orb,
                          f"ORB+BF      good={o_good_count}  ({o_verdict})", max_lines=max_lines)

            count += 1
            if limit_pairs and count >= limit_pairs:
                return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", required=True, help="e.g. ./dataset_folder")
    ap.add_argument("--school", required=True, help="e.g. ./uia_images")
    ap.add_argument("--limit_pairs", type=int, default=0, help="stop after N pairs per folder")
    ap.add_argument("--vis_dir", default="vis", help="folder to save visualizations")
    ap.add_argument("--max_lines", type=int, default=50, help="max match lines to draw per image")

    # thresholds (tweak if your counts differ)
    ap.add_argument("--sift_thresh_fp", type=int, default=30, help="good-match cutoff for SIFT on fingerprints")
    ap.add_argument("--orb_thresh_fp",  type=int, default=20, help="good-match cutoff for ORB on fingerprints")
    ap.add_argument("--sift_thresh_sc", type=int, default=80, help="good-match cutoff for SIFT on school images")
    ap.add_argument("--orb_thresh_sc",  type=int, default=25, help="good-match cutoff for ORB on school images")
    args = ap.parse_args()

    vis_dir = Path(args.vis_dir); vis_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprints: each subfolder is one pair
    run_fingerprint_sets(Path(args.fingerprints), vis_dir, args.max_lines,
                         sift_thresh=args.sift_thresh_fp, orb_thresh=args.orb_thresh_fp)

    # School: all pairs in the given folder
    run_on_folder(Path(args.school), "SCHOOL", vis_dir, args.limit_pairs, args.max_lines,
                  sift_thresh=args.sift_thresh_sc, orb_thresh=args.orb_thresh_sc)

if __name__ == "__main__":
    main()