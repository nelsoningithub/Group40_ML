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

# Function to generate a random position for watermark
def get_random_position(width, height, obj_width, obj_height):
    if obj_width >= width or obj_height >= height:
        return (0, 0)
    max_x = width - obj_width
    max_y = height - obj_height
    return (random.randint(0, max_x), random.randint(0, max_y))

# Function to add watermark (text or logo) and generate mask
def add_watermark_and_generate_mask(image_tensor, watermark_type="text"):
    image = transforms.ToPILImage()(image_tensor.cpu())
    mask_layer = Image.new('L', image.size, 0)  # Black background for mask

    width, height = image.size

    if watermark_type == "text":
        # Add random text
        draw = ImageDraw.Draw(image)  # Draw text directly on the image
        mask_draw = ImageDraw.Draw(mask_layer)
        font = get_random_font()
        watermark_text = get_random_text()

        # Generate random color for text
        text_color = tuple(random.randint(0, 255) for _ in range(3))

        # Get text size
        textbbox = draw.textbbox((0, 0), watermark_text, font=font)
        textwidth = textbbox[2] - textbbox[0]
        textheight = textbbox[3] - textbbox[1]
        position = get_random_position(width, height, textwidth, textheight)

        # Draw watermark text and mask
        draw.text(position, watermark_text, font=font, fill=text_color)
        mask_draw.text(position, watermark_text, font=font, fill=255)

    else:  # Add random logo
        logo_path = random.choice(os.listdir(LOGO_FOLDER))
        logo = Image.open(os.path.join(LOGO_FOLDER, logo_path)).convert("RGBA")
        logo_width, logo_height = logo.size

        # Randomly resize logo
        scale_factor = random.uniform(0.1, 0.3)  # Resize between 10% to 30% of the image width
        new_logo_width = int(width * scale_factor)
        new_logo_height = int(new_logo_width * logo_height / logo_width)
        logo = logo.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)

        # Generate random color for logo
        logo_color = tuple(random.randint(0, 255) for _ in range(3))
        colored_logo = Image.new("RGBA", logo.size)
        colored_logo.paste(logo, (0, 0), logo)
        pixels = colored_logo.load()

        for y in range(colored_logo.height):
            for x in range(colored_logo.width):
                if pixels[x, y][3] > 0:  # Non-transparent pixels
                    pixels[x, y] = (*logo_color, pixels[x, y][3])

        # Get position
        position = get_random_position(width, height, new_logo_width, new_logo_height)

        # Extract the alpha channel from the resized logo (mask shape)
        logo_mask = colored_logo.split()[3]  # The alpha channel

        # Paste the colored logo and its corresponding alpha mask onto the image and mask layers
        image.paste(colored_logo, position, logo_mask)
        mask_layer.paste(logo_mask, position)

    # Convert back to tensor
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

            # Randomly decide whether to add text or logo
            watermark_type = "logo" if random.random() < 0.7 else "text"

            # Generate watermarked image and mask
            watermarked_image, mask = add_watermark_and_generate_mask(image_tensor.squeeze(), watermark_type)

            # Convert tensors back to PIL Images and save
            # Save watermarked image
            if isinstance(watermarked_image, torch.Tensor):
                output_image = transforms.ToPILImage()(watermarked_image.cpu())
            else:  # If it's already a PIL Image
                output_image = watermarked_image
            
            # Convert RGBA images to RGB if saving as JPEG
            if image_file.lower().endswith(('.jpg', '.jpeg')):
                output_image = output_image.convert("RGB")
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
