import cv2

cam = cv2.VideoCapture(0)

fps = cam.get(cv2.CAP_PROP_FPS)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

with open("assignment_1/solutions/camera_outputs.txt", "w") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"height: {height}\n")
    f.write(f"width: {width}\n")

cam.release()