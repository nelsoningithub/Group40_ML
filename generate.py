import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import string

# Device setup (GPU is not used here, only CPU)
device = torch.device("cpu")

# Paths
INPUT_FOLDER = "./web_dataset/no_watermark"  # Input images without watermark
LOGO_FOLDER = "./web_dataset/logo"           # Input logo images
OUTPUT_FOLDER = "./web_dataset/watermarked"  # Output images with watermark
MASK_FOLDER = "./web_dataset/masks"          # Output binary masks for watermarks

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# List of font paths (update paths as needed)
font_paths = [
    "arial.ttf",  # Replace with actual paths to fonts on your system
    "verdana.ttf",
    "times.ttf",
    "comic.ttf"
]

# Function to randomly choose a font and size
def get_random_font():
    font_path = random.choice(font_paths)
    font_size = random.randint(20, 80)  # Random font size
    font = ImageFont.truetype(font_path, font_size)
    return font

# Function to generate random meaningless text
def get_random_text():
    text_length = random.randint(5, 20)  # Random text length
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=text_length))
    return random_text

# Function to randomly select color with 80% probability of gray
def get_random_color():
    if random.random() < 0.8:  # 80% chance of being gray
        gray_value = random.randint(100, 200)  # Range for gray
        return (gray_value, gray_value, gray_value, random.randint(50, 150))  # Add alpha for transparency
    else:  # 20% chance of being a random color
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(50, 150))

# Function to generate a random position for watermark
def get_random_position(width, height, obj_width, obj_height):
    if obj_width >= width or obj_height >= height:
        return (0, 0)
    max_x = width - obj_width
    max_y = height - obj_height
    return (random.randint(0, max_x), random.randint(0, max_y))

# Function to add watermark (text or logo) and generate mask
def add_watermark_and_generate_mask(image_tensor, watermark_type="text"):
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(image_tensor.cpu())
    image = image.convert("RGBA")  # Ensure image is in RGBA mode
    transparent_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))  # Transparent layer
    mask_layer = Image.new('L', image.size, 0)  # Black background for mask

    width, height = image.size

    if watermark_type == "text":
        # Add random text
        draw = ImageDraw.Draw(transparent_layer)
        mask_draw = ImageDraw.Draw(mask_layer)
        font = get_random_font()
        watermark_text = get_random_text()

        # Get text size
        textbbox = draw.textbbox((0, 0), watermark_text, font=font)
        textwidth = textbbox[2] - textbbox[0]
        textheight = textbbox[3] - textbbox[1]
        position = get_random_position(width, height, textwidth, textheight)

        # Draw watermark text and mask
        text_color = get_random_color()
        draw.text(position, watermark_text, font=font, fill=text_color)
        mask_draw.text(position, watermark_text, font=font, fill=255)

    else:  # Add random logo
        logo_path = random.choice(os.listdir(LOGO_FOLDER))
        logo = Image.open(os.path.join(LOGO_FOLDER, logo_path)).convert("RGBA")
        logo_width, logo_height = logo.size

        # Randomly resize logo to smaller size
        scale_factor = random.uniform(0.05, 0.1)  # Resize between 5% to 10% of the image width
        new_logo_width = int(width * scale_factor)
        new_logo_height = int(new_logo_width * logo_height / logo_width)
        logo = logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)

        # Get position
        position = get_random_position(width, height, new_logo_width, new_logo_height)

        # Create a new blank layer to position the logo
        positioned_logo = Image.new('RGBA', image.size, (255, 255, 255, 0))
        positioned_logo.paste(logo, position, logo)

        # Paste the logo on the transparent layer and mask
        transparent_layer = Image.alpha_composite(transparent_layer, positioned_logo)
        mask_layer.paste(logo.split()[3], position)

    # Combine the original image with the transparent layer
    watermarked_image = Image.alpha_composite(image, transparent_layer)

    # Convert back to tensor
    watermarked_image = transforms.ToTensor()(watermarked_image.convert("RGB")).to(device)
    mask_tensor = transforms.ToTensor()(mask_layer).to(device)

    return watermarked_image, mask_tensor



# Process images
def process_images(input_folder, output_folder, mask_folder):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

            # Randomly decide whether to add text or logo
            watermark_type = "logo" if random.random() < 0.7 else "text"

            # Generate watermarked image and mask
            watermarked_image, mask = add_watermark_and_generate_mask(image_tensor.squeeze(), watermark_type)

            # Convert tensors back to PIL Images and save
            output_image = transforms.ToPILImage()(watermarked_image.cpu())
            output_image.save(os.path.join(output_folder, image_file))

            # Save mask
            mask_image = transforms.ToPILImage()(mask.cpu())
            mask_image.save(os.path.join(mask_folder, image_file))

            print(f"Processed {image_file} with {watermark_type} watermark.")

# Main function
if __name__ == "__main__":
    print(f"Processing images from {INPUT_FOLDER}...")
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MASK_FOLDER)
    print("Processing complete. Watermarked images and masks saved.")