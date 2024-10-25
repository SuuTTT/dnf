import cv2
import os
import pytesseract

# Folder containing the material icon images
materials_folder = 'material_icons/'

# Load the main game screenshot
game_screenshot_path = './IMG_2360.PNG'
game_screenshot = cv2.imread(game_screenshot_path)

# Crop the part of the screenshot where the material icons are located
# Update the values (x, y, width, height) according to the position of material icons on your screen
x, y, w, h = 1670, 900, 700, 400   # Example coordinates for the bottom-right corner

cropped_area = game_screenshot[y:y+h, x:x+w]

# Create directories for saving debug images if they don't exist
debug_dir = 'debug'
detected_icons_dir = os.path.join(debug_dir, 'detected_icons')
number_regions_dir = os.path.join(debug_dir, 'number_regions')

os.makedirs(detected_icons_dir, exist_ok=True)
os.makedirs(number_regions_dir, exist_ok=True)

# List to store detected materials
detected_materials = []

# Function to compare the cropped area with material icons
def match_material_icons(cropped_img, template_img, threshold=0.4):
    result = cv2.matchTemplate(cropped_img, template_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val >= threshold, max_loc, max_val

# Loop over all material icons in the folder
for material_file in os.listdir(materials_folder):
    if material_file.endswith('.png'):
        print(f"Processing: {material_file}")
        material_name = os.path.splitext(material_file)[0]  # Remove file extension
        material_icon_path = os.path.join(materials_folder, material_file)
        material_icon = cv2.imread(material_icon_path)

        if material_icon is None:
            print(f"Warning: Unable to load image {material_icon_path}. Skipping.")
            continue

        # Resize the icon if needed to match the size in the game screenshot
        resized_icon = material_icon  # Adjust based on icon size in game if needed

        # Perform template matching
        matched, max_loc, match_score = match_material_icons(cropped_area, resized_icon)
        if matched:
            print(f"Match found for {material_name} with score {match_score:.2f}")
            # Calculate the region where the icon is located
            icon_height, icon_width = resized_icon.shape[:2]
            icon_x, icon_y = max_loc

            # Crop the detected icon area from the cropped area
            detected_icon = cropped_area[icon_y:icon_y + icon_height, icon_x:icon_x + icon_width]

            # Save the detected icon for debugging
            detected_icon_path = os.path.join(detected_icons_dir, f"{material_name}_icon.png")
            cv2.imwrite(detected_icon_path, detected_icon)
            print(f"Saved detected icon to {detected_icon_path}")

            # Calculate the region where the number should be located
            number_region_x = icon_x + icon_width
            number_region_y = icon_y
            number_region_w = icon_width  # Width of the number region (adjust as needed)
            number_region_h = icon_height  # Height of the number region (adjust as needed)

            # Ensure the number region is within the bounds of the cropped area
            number_region_x_end = min(number_region_x + number_region_w, cropped_area.shape[1])
            number_region_y_end = min(number_region_y + number_region_h, cropped_area.shape[0])

            # Crop the number area from the cropped area
            number_region = cropped_area[number_region_y:number_region_y_end,
                                        number_region_x:number_region_x_end]

            # Save the number region for debugging
            number_region_path = os.path.join(number_regions_dir, f"{material_name}_number.png")
            cv2.imwrite(number_region_path, number_region)
            print(f"Saved number region to {number_region_path}")

            # Use pytesseract to read the number from the cropped region
            # Preprocess the number region for better OCR results
            gray_number = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
            _, thresh_number = cv2.threshold(gray_number, 150, 255, cv2.THRESH_BINARY_INV)

            number_text = pytesseract.image_to_string(thresh_number, config='--psm 6 digits')
            #print('Detected Number:', number_text)

            number_text = number_text.strip()
            if number_text == '':
                number_text = '1'
            # Append the detected material and its count to the list
            detected_materials.append((material_name, number_text))
        else:
            print(f"No match found for {material_name}")

# Output the detected materials
print("\nDetected Materials:")
for material, count in detected_materials:
    print(f"Material: {material}, Count: {count}")

# Optionally, save the cropped area with rectangles drawn around detected icons and number regions
# This can help visualize all detections on a single image

# Make a copy of the cropped area to draw rectangles
visual_debug = cropped_area.copy()

for material, count in detected_materials:
    # Find the corresponding icon and number region
    material_icon_path = os.path.join(materials_folder, f"{material}.png")
    material_icon = cv2.imread(material_icon_path)
    if material_icon is None:
        continue
    resized_icon = material_icon
    result = cv2.matchTemplate(cropped_area, resized_icon, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= 0.1:
        icon_height, icon_width = resized_icon.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + icon_width, top_left[1] + icon_height)
        # Draw rectangle around the icon
        cv2.rectangle(visual_debug, top_left, bottom_right, (0, 255, 0), 2)
        # Draw rectangle around the number region
        number_top_left = (top_left[0] + icon_width, top_left[1])
        number_bottom_right = (number_top_left[0] + icon_width, number_top_left[1] + icon_height)
        cv2.rectangle(visual_debug, number_top_left, number_bottom_right, (255, 0, 0), 2)

# Save the visual debug image
visual_debug_path = os.path.join(debug_dir, 'visual_debug.png')
cv2.imwrite(visual_debug_path, visual_debug)
print(f"Saved visual debug image to {visual_debug_path}")

# Note: Ensure Tesseract is installed and pytesseract is configured correctly.
# You might need to specify the tesseract executable path if it's not in your PATH.
# Example:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
