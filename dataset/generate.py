import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
import string

# Check if GPU is available
device = torch.device("cpu")

# List of font paths (update this with the paths to the fonts you have)
font_paths = [
    "arial.ttf",  # Example system font, replace with actual paths to other fonts
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

# Function to generate a random color with transparency
def get_random_color_with_transparency():
    # Random light color with adjustable transparency (alpha channel)
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    alpha = random.randint(50, 150)  # Transparency range from 50 to 150
    return (r, g, b, alpha)

# Function to randomly position the watermark
def get_random_position(width, height, textwidth, textheight):
    # Ensure that text fits inside the image
    if textwidth >= width or textheight >= height:
        return (0, 0)
    
    max_x = width - textwidth
    max_y = height - textheight
    return (random.randint(0, max_x), random.randint(0, max_y))

def add_watermark(image_tensor, watermark_text):
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(image_tensor.cpu())
    
    # Create a transparent layer for the watermark
    transparent_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
    
    # Create a drawing context on the transparent layer
    draw = ImageDraw.Draw(transparent_layer)
    
    # Get a random font and size
    font = get_random_font()
    
    # Get the size of the image
    width, height = image.size
    
    # Get the bounding box of the text
    textbbox = draw.textbbox((0, 0), watermark_text, font=font)
    textwidth = textbbox[2] - textbbox[0]
    textheight = textbbox[3] - textbbox[1]
    
    # Adjust font size if the text doesn't fit
    while textwidth >= width or textheight >= height:
        font_size = font.size - 5
        if font_size < 10:
            font_size = 10
        font = ImageFont.truetype(font.path, font_size)
        textbbox = draw.textbbox((0, 0), watermark_text, font=font)
        textwidth = textbbox[2] - textbbox[0]
        textheight = textbbox[3] - textbbox[1]

    # Randomly position the watermark
    position = get_random_position(width, height, textwidth, textheight)
    
    # Draw the watermark with a random transparent color
    color = get_random_color_with_transparency()
    draw.text(position, watermark_text, font=font, fill=color)
    
    # Combine the original image with the transparent watermark layer
    image = image.convert("RGBA")
    watermarked_image = Image.alpha_composite(image, transparent_layer)
    
    # Convert back to tensor
    watermarked_image = transforms.ToTensor()(watermarked_image.convert("RGB")).to(device)
    
    return watermarked_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            
            # Convert image to tensor
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
            
            # Generate random meaningless text for the watermark
            watermark_text = get_random_text()
            
            # Generate watermarked image
            watermarked_image = add_watermark(image_tensor.squeeze(), watermark_text)
            
            # Convert tensor back to PIL Image and save it with the same name
            output_image = transforms.ToPILImage()(watermarked_image.cpu())
            output_image.save(os.path.join(output_folder, image_file))  # Save with the same name
            print(f"Watermarked {image_file} saved with text: {watermark_text}")

# Main function
if __name__ == "__main__":
    input_folder = "no_watermark"
    output_folder = "watermarked"
    
    process_images(input_folder, output_folder)
