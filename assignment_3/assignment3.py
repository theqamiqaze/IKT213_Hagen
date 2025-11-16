import cv2
import numpy as np
import os

# As with assignment 2, processed images are uploaded to the repo along with the code itself. To do local testing, just delete the images in the folder and then run the program.

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    # Sobel edge detection with Gaussian blur first.
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def canny_edge_detection(image: np.ndarray, threshold_1: int, threshold_2: int) -> np.ndarray:
    # Canny edge detection with Gaussian blur first.
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, threshold_1, threshold_2)


def template_match(image: np.ndarray, template: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    # Template matching with red rectangles on matches.
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    output = image.copy()
    for pt in zip(*locations[::-1]):
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return output


def resize(image: np.ndarray, scale_factor: int, up_or_down: str) -> np.ndarray:
    # Resize using image pyramids (pyrUp / pyrDown).
    resized = image.copy()
    for _ in range(scale_factor):
        if up_or_down == "up":
            resized = cv2.pyrUp(resized)
        elif up_or_down == "down":
            resized = cv2.pyrDown(resized)
        else:
            raise ValueError("up_or_down must be 'up' or 'down'")
    return resized


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    solutions_dir = os.path.join(script_dir, "solutions")

    lambo = cv2.imread(os.path.join(script_dir, "lambo.png"))
    shapes_img = cv2.imread(os.path.join(script_dir, "shapes-1.png"))
    template_img = cv2.imread(os.path.join(script_dir, "shapes_template.jpg"))

    if lambo is None or shapes_img is None or template_img is None:
        raise RuntimeError("One or more required images failed to load.")

    # Sobel
    sobel_edges = sobel_edge_detection(lambo)
    cv2.imwrite(os.path.join(solutions_dir, "lambo_sobel_edges.png"), sobel_edges)
    print("Sobel edge detection done, output saved to solutions folder.")

    # Canny
    canny_edges = canny_edge_detection(lambo, threshold_1=50, threshold_2=50)
    cv2.imwrite(os.path.join(solutions_dir, "lambo_canny_edges.png"), canny_edges)
    print("Canny edge detection done, output saved to solutions folder.")

    # Template matching
    matched = template_match(shapes_img, template_img, threshold=0.9)
    cv2.imwrite(os.path.join(solutions_dir, "shapes_template_matched.png"), matched)
    print("Template matching done, output saved to solutions folder.")

    # Pyramid resizing
    lambo_down = resize(lambo, scale_factor=1, up_or_down='down')
    cv2.imwrite(os.path.join(solutions_dir, "lambo_resized_down.png"), lambo_down)
    print("Downscale (pyramid) done, output saved to solutions folder.")

    lambo_up = resize(lambo, scale_factor=1, up_or_down='up')
    cv2.imwrite(os.path.join(solutions_dir, "lambo_resized_up.png"), lambo_up)
    print("Upscale (pyramid) done, output saved to solutions folder.")

    print("Assignment 3 processing complete.")


if __name__ == "__main__":
    main()