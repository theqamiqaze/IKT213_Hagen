import cv2
import numpy as np
import os

# I originally tried SIFT/FLANN, but the output was bad, and ORB, despite being unstable, worked better.
# Though ORB isn't perfect either.

def harris_corner_detection(reference_image: np.ndarray) -> np.ndarray:
    # Detect and mark corners using Harris corner detector.
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst_dilated = cv2.dilate(dst, None)

    corners_marked = reference_image.copy()
    corners_marked[dst_dilated > 0.01 * dst_dilated.max()] = [0, 0, 255]
    return corners_marked


def align_images(
    image_to_align: np.ndarray,
    reference_image: np.ndarray,
    max_features: int = 1500,
    good_match_percent: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    # Align one image to a reference using ORB features and homography.
    im1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, homography, (width, height))

    matches_image = cv2.drawMatches(
        image_to_align,
        keypoints1,
        reference_image,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return aligned_image, matches_image


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    solutions_dir = os.path.join(script_dir, "solutions")

    reference_path = os.path.join(script_dir, "reference_img.png")
    to_align_path = os.path.join(script_dir, "align_this.jpg")

    ref_img = cv2.imread(reference_path)
    align_img = cv2.imread(to_align_path)

    if ref_img is None or align_img is None:
        raise RuntimeError("Failed to load reference or alignment image")

    # Harris corner detection
    harris_result = harris_corner_detection(ref_img)
    cv2.imwrite(os.path.join(solutions_dir, "harris.png"), harris_result)
    print("Harris corner detection done, output saved to solutions folder.")

    # Feature-based alignment
    aligned, matches = align_images(
        align_img, ref_img, max_features=1500, good_match_percent=0.15
    )
    cv2.imwrite(os.path.join(solutions_dir, "aligned.png"), aligned)
    cv2.imwrite(os.path.join(solutions_dir, "matches.png"), matches)
    print("Feature-based alignment done, outputs saved to solutions folder.")

    print("Assignment 4 processing complete.")


if __name__ == "__main__":
    main()