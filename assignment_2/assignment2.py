import cv2
import numpy as np
import os

# I wasn't sure whether the processed images should go with the code, but they are present as well. For local testing, just delete the images within the solution folder and run the program.

def padding(image: np.ndarray, border_width: int) -> np.ndarray:
    # Add a reflected border around the image.
    return cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT,
    )


def crop(image: np.ndarray, x_0: int, x_1: int, y_0: int, y_1: int) -> np.ndarray:
    # Crop the image by removing pixels from each side.
    h, w = image.shape[:2]
    x_start = max(x_0, 0)
    x_end = max(w - x_1, x_start)
    y_start = max(y_0, 0)
    y_end = max(h - y_1, y_start)
    return image[y_start:y_end, x_start:x_end]


def resize(image: np.ndarray, size: tuple = (200, 200)) -> np.ndarray:
    # Resize the image to the given (width, height).
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def copy(image: np.ndarray, empty_picture_array: np.ndarray) -> np.ndarray:
    # Manually copy pixels from image into empty_picture_array.
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            empty_picture_array[y, x] = image[y, x]
    return empty_picture_array


def grayscale(image: np.ndarray) -> np.ndarray:
    # Convert BGR image to grayscale.
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def hsv(image: np.ndarray) -> np.ndarray:
    # Convert BGR image to HSV colour space.
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hue_shift(image: np.ndarray, hue: int) -> np.ndarray:
    # Shift all colour channels by a constant value, clipped to [0, 255].
    shifted = image.astype(np.int16) + hue
    return np.clip(shifted, 0, 255).astype(np.uint8)


def smoothing(image: np.ndarray) -> np.ndarray:
    # Apply Gaussian blur with kernel size 15x15.
    return cv2.GaussianBlur(image, ksize=(15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)


def rotation(image: np.ndarray, rotation_angle: int) -> np.ndarray:
    # Rotate image by 90 or 180 degrees.
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("rotation_angle must be 90 or 180")


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lena_path = os.path.join(script_dir, "lena.png")
    solutions_dir = os.path.join(script_dir, "solutions")

    if not os.path.exists(lena_path):
        raise FileNotFoundError(f"lena.png not found at {lena_path}")
    lena = cv2.imread(lena_path)
    if lena is None:
        raise RuntimeError("Failed to load lena.png")

    # 1. Padding
    padded = padding(lena, border_width=100)
    cv2.imwrite(os.path.join(solutions_dir, "lena_padded.png"), padded)
    print("Padding operation done, output saved to solutions folder.")

    # 2. Cropping (80 from left/top, 130 from bottom/right)
    cropped = crop(lena, x_0=80, x_1=130, y_0=80, y_1=130)
    cv2.imwrite(os.path.join(solutions_dir, "lena_cropped.png"), cropped)
    print("Cropping operation done, output saved to solutions folder.")

    # 3. Resize to 200x200
    resized = resize(lena, size=(200, 200))
    cv2.imwrite(os.path.join(solutions_dir, "lena_resized.png"), resized)
    print("Resize operation done, output saved to solutions folder.")

    # 4. Manual copy
    height, width = lena.shape[:2]
    empty_picture_array = np.zeros([height, width, 3], dtype=np.uint8)
    copied = copy(lena, empty_picture_array)
    cv2.imwrite(os.path.join(solutions_dir, "lena_copied.png"), copied)
    print("Copy operation done, output saved to solutions folder.")

    # 5. Grayscale
    gray = grayscale(lena)
    cv2.imwrite(os.path.join(solutions_dir, "lena_gray.png"), gray)
    print("Grayscale operation done, output saved to solutions folder.")

    # 6. HSV
    hsv_image = hsv(lena)
    cv2.imwrite(os.path.join(solutions_dir, "lena_hsv.png"), hsv_image)
    print("HSV operation done, output saved to solutions folder.")

    # 7. Colour shift (+50)
    hue_shifted = hue_shift(lena, hue=50)
    cv2.imwrite(os.path.join(solutions_dir, "lena_hueshift.png"), hue_shifted)
    print("Hue shift operation done, output saved to solutions folder.")

    # 8. Smoothing
    smoothed = smoothing(lena)
    cv2.imwrite(os.path.join(solutions_dir, "lena_smoothed.png"), smoothed)
    print("Smoothing operation done, output saved to solutions folder.")

    # 9. Rotations
    rotated_90 = rotation(lena, rotation_angle=90)
    cv2.imwrite(os.path.join(solutions_dir, "lena_rotated90.png"), rotated_90)
    print("90 degree rotation done, output saved to solutions folder.")

    rotated_180 = rotation(lena, rotation_angle=180)
    cv2.imwrite(os.path.join(solutions_dir, "lena_rotated180.png"), rotated_180)
    print("180 degree rotation done, output saved to solutions folder.")

    print("Assignment 2 processing complete.")


if __name__ == "__main__":
    main()