from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import cv2, os, img2pdf, fitz, shutil

def get_image_list(output_dir):
    images = os.listdir(output_dir)
    images.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    image_paths = [os.path.join(output_dir, image) for image in images]
    if not image_paths:
        print("No valid images were processed.")
        return None, None
    return image_paths


def get_max_image_dimensions(folder_path):
    max_width = 0
    max_height = 0

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is not None:
                height, width, _ = image.shape
                max_width = max(max_width, width)
                max_height = max(max_height, height)

    return max_width

def stitch_all(folder_path, image_list):
    images = []
    output_width = get_max_image_dimensions(folder_path=folder_path)
    top_padding = 20
    bottom_padding = 20

    for filename in tqdm(image_list):
        image = cv2.imread(filename)

        if image is not None:
            current_width = image.shape[1]
            border_width = output_width - current_width
            left_border_width = border_width // 2
            right_border_width = border_width - left_border_width

            canvas_height = image.shape[0] + top_padding + bottom_padding
            canvas = np.ones((canvas_height, output_width, 3), np.uint8) * 255

            # Add the image to the canvas with padding
            canvas[top_padding:top_padding + image.shape[0], left_border_width:left_border_width + current_width, :] = image

            images.append(canvas)
    try:
        combined_image = np.vstack(images)
        return combined_image
    except Exception as e:
        return False



def get_document(image, name):
    # Load the input image using cv2
    output_dir = "output_images"  # Directory to save split images

    split_height = 1555  # Height at which to split the image

    if image is None:
        print("Error: Image not found.")
        return

    height, width, channels = image.shape

    # Check if the split height is within the image's height bounds
    if split_height <= 0 or split_height >= height:
        print("Error: Invalid split height.")
        return

    # Determine the number of splits
    num_splits = height // split_height

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the image and save each part
    imgs = []
    start = 0
    split_count = 0

    while start < height:
        end = min(start + split_height, height)
        split_image = image[start:end, :]
        # Save the split image
        output_path = os.path.join(output_dir, f"split_{split_count}.png")
        cv2.imwrite(output_path, split_image)

        # print(f"Saved: {output_path}")
        imgs.append(output_path)
        split_count += 1
        start = end

    
    output_pdf =  f'{name}_notes.pdf'  # Specify the output PDF path

    with open(output_pdf, 'wb') as pdf_output:
        pdf_output.write(img2pdf.convert(imgs))


def process_image(page, counter, output_dir):
    # Read the input image
    image_list = page.get_pixmap(matrix = fitz.Matrix(300/72, 300/72))
    image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)
    image_np = np.array(image)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Define the RGB color range for rectangles
    # lower_color = np.array([40, 135, 94], dtype=np.uint8)
    # upper_color = np.array([80, 220, 110], dtype=np.uint8)
    lower_color = np.array([40,115, 70], dtype=np.uint8)
    upper_color = np.array([110, 185, 150], dtype=np.uint8)

    # Threshold the image to isolate the specified color
    color_mask = cv2.inRange(image, lower_color, upper_color)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Loop through identified contours and crop rectangles
    for idx, contour in (enumerate(contours)):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(contour):
            x, y, w, h = cv2.boundingRect(contour)
            if w>400:
                cropped_rectangle = image[y:y + h, x:x + w]

                # Save the cropped rectangle to the output directory
                cv2.imwrite(os.path.join(output_dir, f'page_{counter+1}_{idx}.jpg'), cropped_rectangle)

            else:
                pass


input_pdf = 'Samiksha Program.pdf'
name = os.path.splitext(os.path.basename(input_pdf))[0]
output_dir = os.path.join('output', name)
os.makedirs(output_dir, exist_ok=True)
pdf_document = fitz.open(input_pdf)
total_pages = pdf_document.page_count
for page_number in tqdm(range(total_pages)):
    page = pdf_document.load_page(page_number)
    processed_image = process_image(page, page_number, output_dir)

pdf_document.close()

image_paths = get_image_list(output_dir)
combined_image = stitch_all(output_dir, image_paths)
get_document(combined_image, name)




# import os

# # Function to process the text file
# def process_text_file(input_file_path):
#     with open(input_file_path, 'r') as input_file:
#         # Read the content of the file
#         lines = input_file.readlines()

#     # Remove empty lines
#     non_empty_lines = [line.strip() for line in lines if line.strip()]

#     # Remove lines starting with 'Pen' and the line below it
#     # Remove lines starting with 'Rectangle'
#     filtered_lines = []
#     skip_next_line = False
#     for line in non_empty_lines:
#         if line.startswith('Pen'):
#             skip_next_line = True
#         elif line.startswith('Rectangle'):
#             continue
#         elif skip_next_line:
#             skip_next_line = False
#         else:
#             filtered_lines.append(line)

#     # Extract page numbers starting with 'Page #'
#     original = [line.split()[1] for line in filtered_lines if line.startswith('Page')]

#     return original

# # Function to process image file names in a folder
# def process_image_folder(folder_path):
#     # Get all file names from the folder
#     file_names = os.listdir(folder_path)

#     # Extract xyz parts from file names
#     extracted_numbers = [name.split('_')[1] for name in file_names if name.startswith('page')]

#     return extracted_numbers

# # Specify the paths
# input_text_file_path = 'count.txt'
# image_folder_path = output_dir

# # Process the text file
# original_numbers = process_text_file(input_text_file_path)
# print(f"Length of orig - {len(original_numbers)}")
# # Process the image folder
# extracted_numbers = process_image_folder(image_folder_path)
# print(f"Length of extracted - {len(extracted_numbers)}")

# # Find numbers not matching in the two lists
# mismatched_numbers = set(original_numbers) - set(extracted_numbers)

# # Print or use the results as needed
# print("Original Numbers:", sorted(original_numbers))
# print("Extracted Numbers:", sorted(extracted_numbers))
# print("Mismatched Numbers:", list(mismatched_numbers))

shutil.rmtree(output_dir)
shutil.rmtree('output_images')