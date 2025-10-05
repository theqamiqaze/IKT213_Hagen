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

# ---------- NEW: small helpers ----------
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def inlier_subset(matches, mask):
    if mask is None:
        return matches
    keep = mask.ravel().tolist()
    return [m for m, k in zip(matches, keep) if k]

def verdict_from_inliers(inliers, good, min_inliers=15, min_ratio=0.30):
    if good == 0:
        return "NO MATCH"
    ratio = inliers / float(good)
    return "MATCH" if (inliers >= min_inliers and ratio >= min_ratio) else "NO MATCH"
# ----------------------------------------

def ratio_filter(knn_matches, ratio):
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def find_homography(kp1, kp2, matches, reproj_thresh=3.0):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    return H, mask

def draw_and_save(img1, img2, kp1, kp2, matches, H, out_path: Path, title: str, max_lines=50):
    # Sort matches and keep a few for visualization
    matches = sorted(matches, key=lambda m: m.distance)[:max_lines]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # If we have a homography, draw the projected box of img1 onto img2
    if H is not None:
        h, w = img1.shape[:2]
        box = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        try:
            dst = cv2.perspectiveTransform(box, H)
            # drawMatches puts img2 to the RIGHT of img1; shift polygon accordingly
            dst[:,0,0] += img1.shape[1]
            cv2.polylines(vis, [np.int32(dst)], True, (255,255,255), 3)
        except cv2.error:
            pass
    # Small title overlay
    cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

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

def run_on_folder(folder: Path, label: str, vis_dir: Path, limit_pairs: int = 0, max_lines=50):
    files = list_images(folder)
    files.sort()
    if len(files) < 2:
        print(f"[{label}] Need at least two images in {folder}")
        return
    print(f"\n=== {label}: {folder} ===")
    count = 0
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            a, b = files[i], files[j]
            # --- NEW: CLAHE pre-processing ---
            img1 = apply_clahe(read_gray(a))
            img2 = apply_clahe(read_gray(b))

            sift_good, sift_ms, s_k1, s_k2, s_good = sift_flann(img1, img2)
            orb_good,  orb_ms,  o_k1, o_k2, o_good  = orb_bf(img1, img2)

            # --- Compute homographies and inliers ---
            H_sift, mask_s = find_homography(s_k1, s_k2, s_good)
            H_orb,  mask_o = find_homography(o_k1, o_k2, o_good)

            s_inliers = inlier_subset(s_good, mask_s)
            o_inliers = inlier_subset(o_good, mask_o)

            s_verdict = verdict_from_inliers(len(s_inliers), len(s_good), min_inliers=15, min_ratio=0.30)
            o_verdict = verdict_from_inliers(len(o_inliers), len(o_good),  min_inliers=15, min_ratio=0.30)

            print(f"- Pair: {a.name} vs {b.name}")
            print(f"  SIFT+FLANN -> good: {len(s_good):4d}, inliers: {len(s_inliers):4d}, time: {sift_ms:7.1f} ms, {s_verdict}")
            print(f"  ORB+BF    -> good: {len(o_good):4d}, inliers: {len(o_inliers):4d}, time: {orb_ms:7.1f} ms, {o_verdict}")

            # --- Draw ONLY INLIERS (clean visuals) ---
            out_sift = vis_dir / f"{label}_SIFT_{a.stem}_VS_{b.stem}.jpg"
            draw_and_save(
                img1, img2, s_k1, s_k2, s_inliers, H_sift, out_sift,
                f"SIFT+FLANN  inliers={len(s_inliers)}  ({s_verdict})", max_lines=max_lines
            )

            out_orb = vis_dir / f"{label}_ORB_{a.stem}_VS_{b.stem}.jpg"
            draw_and_save(
                img1, img2, o_k1, o_k2, o_inliers, H_orb, out_orb,
                f"ORB+BF      inliers={len(o_inliers)}  ({o_verdict})", max_lines=max_lines
            )

            count += 1
            if limit_pairs and count >= limit_pairs:
                return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", required=True, help="e.g. ./dataset_folder/data_check")
    ap.add_argument("--school", required=True, help="e.g. ./uia_images")
    ap.add_argument("--limit_pairs", type=int, default=0, help="stop after N pairs per folder")
    ap.add_argument("--vis_dir", default="vis", help="folder to save visualizations")
    ap.add_argument("--max_lines", type=int, default=50, help="max match lines to draw per image")
    args = ap.parse_args()

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    run_on_folder(Path(args.fingerprints), "FINGERPRINTS", vis_dir, args.limit_pairs, args.max_lines)
    run_on_folder(Path(args.school), "SCHOOL", vis_dir, args.limit_pairs, args.max_lines)

if __name__ == "__main__":
    main()