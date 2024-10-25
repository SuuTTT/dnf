import cv2
import os
import pytesseract
import numpy as np

# ======================= Configuration =======================

# Specify the path to the Tesseract executable if it's not in your PATH
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder containing the material icon images
materials_folder = 'material_icons/'

# Path to the main game screenshot
game_screenshot_path = './IMG_2360.PNG'

# Coordinates to crop the area where material icons are located (x, y, width, height)
# Adjust these values based on your game's UI layout
crop_x, crop_y, crop_w, crop_h = 1300, 900, 400, 200  # Example values

# Minimum number of good matches required to consider a detection valid
MIN_MATCH_COUNT = 10

# ORB Parameters
ORB_FEATURES = 1000  # Number of features to detect

# ===============================================================

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# Load the main game screenshot
game_screenshot = cv2.imread(game_screenshot_path)

if game_screenshot is None:
    raise FileNotFoundError(f"Screenshot not found at {game_screenshot_path}")

# Crop the specified area from the screenshot
cropped_area = game_screenshot#[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
cropped_area_gray = cv2.cvtColor(cropped_area, cv2.COLOR_BGR2GRAY)

# Create directories for saving debug images if they don't exist
debug_dir = 'debug'
detected_icons_dir = os.path.join(debug_dir, 'detected_icons')
number_regions_dir = os.path.join(debug_dir, 'number_regions')

os.makedirs(detected_icons_dir, exist_ok=True)
os.makedirs(number_regions_dir, exist_ok=True)

# List to store detected materials
detected_materials = []

# Loop over all material icons in the folder
for material_file in os.listdir(materials_folder):
    if material_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"\nProcessing: {material_file}")
        material_name = os.path.splitext(material_file)[0]  # Remove file extension
        material_icon_path = os.path.join(materials_folder, material_file)
        material_icon_color = cv2.imread(material_icon_path)

        if material_icon_color is None:
            print(f"Warning: Unable to load image {material_icon_path}. Skipping.")
            continue

        # Convert the material icon to grayscale
        material_icon_gray = cv2.cvtColor(material_icon_color, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors for the template icon
        kp1, des1 = orb.detectAndCompute(material_icon_gray, None)
        if des1 is None:
            print(f"Warning: No descriptors found in {material_file}. Skipping.")
            continue

        # Detect keypoints and compute descriptors for the cropped area
        kp2, des2 = orb.detectAndCompute(cropped_area_gray, None)
        if des2 is None:
            print("Warning: No descriptors found in the cropped area. Skipping.")
            continue

        # Initialize Brute-Force matcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        if not matches:
            print(f"No matches found for {material_name}.")
            continue

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter good matches based on a distance threshold
        # This threshold may need adjustment based on your data
        good_matches = [m for m in matches if m.distance < 60]

        print(f"Found {len(good_matches)} good matches for {material_name}.")

        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract the locations of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography to find the position of the icon in the cropped area
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h_icon, w_icon = material_icon_gray.shape
                # Define the corners of the template icon
                pts = np.float32([[0, 0], [0, h_icon], [w_icon, h_icon], [w_icon, 0]]).reshape(-1, 1, 2)
                # Transform the corners to the cropped area
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the detected icon region on a copy of the cropped area for debugging
                detected_area = cropped_area.copy()
                cv2.polylines(detected_area, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
                detected_icon_path = os.path.join(detected_icons_dir, f"{material_name}_icon_detected.png")
                cv2.imwrite(detected_icon_path, detected_area)
                print(f"Saved detected icon area to {detected_icon_path}")

                # Calculate bounding box coordinates
                x_coords = dst[:, 0, 0]
                y_coords = dst[:, 0, 1]
                x_min = int(np.min(x_coords))
                y_min = int(np.min(y_coords))
                x_max = int(np.max(x_coords))
                y_max = int(np.max(y_coords))

                # Define the number region based on the detected icon's position
                # Adjust the offset (e.g., to the right of the icon) as per your game's UI
                number_region_offset_x = 10  # Pixels to the right of the icon
                number_region_x = x_max + number_region_offset_x
                number_region_y = y_min
                number_region_w = w_icon  # Width of the number region
                number_region_h = h_icon  # Height of the number region

                # Ensure the number region is within the bounds of the cropped area
                number_region_x_end = min(number_region_x + number_region_w, cropped_area.shape[1])
                number_region_y_end = min(number_region_y + number_region_h, cropped_area.shape[0])

                # Crop the number region
                number_region = cropped_area[number_region_y:number_region_y_end,
                                            number_region_x:number_region_x_end]

                # Save the number region for debugging
                number_region_path = os.path.join(number_regions_dir, f"{material_name}_number.png")
                cv2.imwrite(number_region_path, number_region)
                print(f"Saved number region to {number_region_path}")

                # Preprocess the number region for better OCR results
                gray_number = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
                _, thresh_number = cv2.threshold(gray_number, 150, 255, cv2.THRESH_BINARY_INV)

                # Optionally, apply dilation or erosion to enhance the number visibility
                kernel = np.ones((2, 2), np.uint8)
                thresh_number = cv2.dilate(thresh_number, kernel, iterations=1)

                # Use pytesseract to read the number from the cropped region
                number_text = pytesseract.image_to_string(thresh_number, config='--psm 6 digits')
                number_text = number_text.strip()

                if number_text == '':
                    number_text = '1'  # Default count if OCR fails

                print(f"Detected Number for {material_name}: {number_text}")

                # Append the detected material and its count to the list
                detected_materials.append((material_name, number_text))
            else:
                print(f"Homography could not be computed for {material_name}.")
        else:
            print(f"Not enough good matches for {material_name} (found {len(good_matches)}).")

# ======================= Output Results =======================

print("\n================ Detected Materials ================")
for material, count in detected_materials:
    print(f"Material: {material}, Count: {count}")
print("=======================================================")

# ======================= Visualization (Optional) =======================

# Create a visual debug image with all detections
visual_debug = cropped_area.copy()

for material, count in detected_materials:
    material_icon_path = os.path.join(materials_folder, f"{material}.png")
    material_icon_color = cv2.imread(material_icon_path)

    if material_icon_color is None:
        continue

    material_icon_gray = cv2.cvtColor(material_icon_color, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(material_icon_gray, None)

    if des1 is None:
        continue

    kp2, des2 = orb.detectAndCompute(cropped_area_gray, None)
    if des2 is None:
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 60]

    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h_icon, w_icon = material_icon_gray.shape
            pts = np.float32([[0, 0], [0, h_icon], [w_icon, h_icon], [w_icon, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw rectangle around the icon
            cv2.polylines(visual_debug, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw rectangle around the number region
            x_coords = dst[:, 0, 0]
            y_coords = dst[:, 0, 1]
            x_min = int(np.min(x_coords))
            y_min = int(np.min(y_coords))
            x_max = int(np.max(x_coords))
            y_max = int(np.max(y_coords))

            number_region_offset_x = 10  # Must match the offset used earlier
            number_region_x = x_max + number_region_offset_x
            number_region_y = y_min
            number_region_w = w_icon
            number_region_h = h_icon

            cv2.rectangle(visual_debug, (number_region_x, number_region_y),
                          (number_region_x + number_region_w, number_region_y + number_region_h),
                          (255, 0, 0), 2)

# Save the visual debug image
visual_debug_path = os.path.join(debug_dir, 'visual_debug.png')
cv2.imwrite(visual_debug_path, visual_debug)
print(f"\nSaved visual debug image to {visual_debug_path}")

# ===============================================================

