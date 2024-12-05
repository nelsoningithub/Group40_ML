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
    font_size = random.randint(20, 30)  # Random font size
    return ImageFont.truetype(font_path, font_size)

# Function to generate random meaningless text
def get_random_text():
    text_length = random.randint(5, 20)  # Random text length
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=text_length))
    return random_text

# Function to generate a random position for watermark
def get_random_position(width, height, obj_width, obj_height):
    max_x = max(0, width - obj_width)
    max_y = max(0, height - obj_height)
    return random.randint(0, max_x), random.randint(0, max_y)

# Function to apply a logo watermark
def apply_logo_watermark(image, mask_layer):
    logo_path = random.choice(os.listdir(LOGO_FOLDER))
    logo = Image.open(os.path.join(LOGO_FOLDER, logo_path)).convert("RGBA")

    # Resize the logo to 1/3 of its original size
    new_logo_width = logo.width // 3
    new_logo_height = logo.height // 3
    logo = logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)

    # Generate random position for the logo
    position = get_random_position(image.width, image.height, new_logo_width, new_logo_height)

    # Create a semi-transparent logo
    logo_alpha = logo.split()[3]  # Extract alpha channel
    transparent_logo = Image.new("RGBA", logo.size, (200, 200, 200, random.randint(100, 150)))
    transparent_logo.putalpha(logo_alpha)

    # Paste the logo onto the image and mask
    image.paste(transparent_logo, position, transparent_logo)
    mask_layer.paste(logo_alpha, position)

# Function to apply text watermark
def apply_text_watermark(image, mask_layer):
    transparent_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(transparent_layer)
    mask_draw = ImageDraw.Draw(mask_layer)

    font = get_random_font()
    watermark_text = get_random_text()

    text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    position = get_random_position(image.width, image.height, text_width, text_height)
    color = (200, 200, 200, random.randint(100, 150))  # Semi-transparent gray
    draw.text(position, watermark_text, font=font, fill=color)
    mask_draw.text(position, watermark_text, font=font, fill=255)

    image.alpha_composite(transparent_layer)

# Function to add watermark (text or logo) and generate mask
def add_watermark_and_generate_mask(image_tensor):
    image = transforms.ToPILImage()(image_tensor.cpu()).convert("RGBA")
    mask_layer = Image.new("L", image.size, 0)

    # Choose watermark type (logo: 60%, text: 40%)
    if random.random() < 0.6:
        apply_logo_watermark(image, mask_layer)
    else:
        apply_text_watermark(image, mask_layer)

    watermarked_image = transforms.ToTensor()(image.convert("RGB")).to(device)
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

            watermarked_image, mask = add_watermark_and_generate_mask(image_tensor.squeeze())

            output_image = transforms.ToPILImage()(watermarked_image.cpu())
            output_image.save(os.path.join(output_folder, image_file))

            mask_image = transforms.ToPILImage()(mask.cpu())
            mask_image.save(os.path.join(mask_folder, image_file))

            print(f"Processed {image_file}")

# Main function
if __name__ == "__main__":
    print(f"Processing images from {INPUT_FOLDER}...")
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, MASK_FOLDER)
    print("Processing complete. Watermarked images and masks saved.")
