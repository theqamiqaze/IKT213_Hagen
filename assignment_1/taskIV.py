import cv2

def print_image_information(image):
    height, width, channels = image.shape
    
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size (total number of values):", image.size)
    print("Data type:", image.dtype)

def main():
    image = cv2.imread("assignment_1/lena.png")
    print_image_information(image)

if __name__ == "__main__":
    main()