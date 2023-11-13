import cv2
import numpy as np

def resize_image(clean_image):
    screen_height, screen_width = 720, 1080  # Adjust the screen_height as needed
    window_height, window_width = clean_image.shape[:2]

    if window_height > screen_height:
        # Calculate the width to maintain the aspect ratio
        aspect_ratio = window_width / window_height
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
        clean_image = cv2.resize(clean_image, (new_width, new_height))
    return clean_image

def process_image(image, lower_green, upper_green):
    green_mask = cv2.inRange(image, lower_green, upper_green)

    return green_mask

def on_slider_change(x):
    # Get current slider values
    lower_green = (cv2.getTrackbarPos('Lower B', 'Trackbars'),
                   cv2.getTrackbarPos('Lower G', 'Trackbars'),
                   cv2.getTrackbarPos('Lower R', 'Trackbars'))
    upper_green = (cv2.getTrackbarPos('Upper B', 'Trackbars'),
                   cv2.getTrackbarPos('Upper G', 'Trackbars'),
                   cv2.getTrackbarPos('Upper R', 'Trackbars'))

 
    # Process the image with the updated values
    # image = cv2.imread('images/IMG_5023.JPG')
    
    clean_image = process_image(image, lower_green, upper_green)
    
    resized_image = resize_image(clean_image=clean_image)
    # Show the processed image
    cv2.imshow('Processed Image', resized_image)

if __name__ == "__main__":
    image = cv2.imread('a.JPEG')

    # Create a window for trackbars
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)

    init_lower_green = np.array([30, 115, 70], dtype=np.uint8)
    init_upper_green = np.array([135, 200, 170], dtype=np.uint8)

    # Create sliders for lower and upper green values
    cv2.createTrackbar('Lower B', 'Trackbars', init_lower_green[0], 255, on_slider_change)
    cv2.createTrackbar('Lower G', 'Trackbars', init_lower_green[1], 255, on_slider_change)
    cv2.createTrackbar('Lower R', 'Trackbars', init_lower_green[2], 255, on_slider_change)

    cv2.createTrackbar('Upper B', 'Trackbars', init_upper_green[0], 255, on_slider_change)
    cv2.createTrackbar('Upper G', 'Trackbars', init_upper_green[1], 255, on_slider_change)
    cv2.createTrackbar('Upper R', 'Trackbars', init_upper_green[2], 255, on_slider_change)

 
    # Display the original image
    image_resize = resize_image(image)
    cv2.imshow('Original Image', image_resize)

    # Process the image with initial values
    clean_image = process_image(image, init_lower_green, init_upper_green)
    resized_image = resize_image(clean_image=clean_image)
    # Display the processed image
    cv2.imshow('Processed Image', resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
