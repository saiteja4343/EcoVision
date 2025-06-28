"""
Flux 1.dev Synthetic Food Image Generation
Generates synthetic food images for classes with insufficient training data
"""

import os
import torch
import argparse
from diffusers import FluxPipeline
from PIL import Image
from huggingface_hub import login
import random

# Authenticate with Hugging Face (replace with your token)
login(token="your_huggingface_token_here")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic food images using Flux 1.dev")
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2, 3], required=True, help="GPU number (0-3)")
    parser.add_argument('--output_dir', type=str, default='/home/gen_imgs_flux_all', help="Output directory")
    args = parser.parse_args()

    # Configure GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Flux pipeline
    print("Loading Flux 1.dev model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # Memory optimization
    pipe.enable_model_cpu_offload(device=device)
    if hasattr(pipe, 'vae'):
        pipe.vae.enable_slicing()

    # Class lists for different GPUs (balanced distribution, can be changed acc. to no. of images)
    gpu_class_lists = {
        0: ['Blueberries', 'Garlic', 'Radish'],
        1: ['Raspberries', 'Beans', 'Passion_fruits'],
        2: ['Bell_peppers', 'Brussel_sprouts', 'Grapes'],
        3: ['Galia_melons', 'Asparagus', 'Plums']
    }
    
    # Image counts for each class
    images_needed = {
        'Asparagus': 400, 'Beans': 50, 'Bell_peppers': 150,
        'Brussel_sprouts': 500, 'Galia_melons': 450, 'Grapefruits': 450,
        'Grapes': 260, 'Passion_fruits': 410, 'Peas': 425,
        'Plums': 150, 'Raspberries': 210, 'Blueberries': 20,
        'Garlic': 30, 'Radish': 20
    }

    # Generate images for selected GPU's classes
    class_list = gpu_class_lists.get(args.gpu, [])
    
    for class_name in class_list:
        generate_class_images(pipe, class_name, images_needed.get(class_name, 100), args.output_dir)

def generate_class_images(pipe, class_name, num_images, output_dir):
    """Generate images for a specific food class"""
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Number word mapping
    words = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
        13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
        17: 'seventeen', 18: 'eighteen', 19: 'nineteen'
    }
    
    for i in range(num_images):
        # Randomize number of objects
        if class_name in ['Beans', 'Blueberries', 'Brussel_sprouts', 'Raspberries']:
            num = random.randint(10, 19)
        else:
            num = random.randint(2, 6)
        
        # Generate prompt
        prompt = create_prompt(class_name, words[num])
        
        # Generate image
        print(f"Generating {class_name} image {i+1}/{num_images} with {num} objects...")
        result = pipe(
            prompt=prompt,
            num_inference_steps=35,
            guidance_scale=9.0,
        ).images[0]
        
        # Save image
        output_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")
        result.save(output_path)
    
    print(f"Completed generating {num_images} images for {class_name}")

def create_prompt(class_name, num_word):
    """Create class-specific prompts for realistic food generation"""
    prompts = {
        "Asparagus": f"A natural photograph of {num_word} Asparagus spears placed at a noticeable distance from each other in random positions, sharp and clear background, natural lighting, unique and varied background setting, high resolution, photorealistic detail, 4K quality",
        "Beans": f"A natural photograph of different types of dry beans, only these in sharp focus, no other objects, each type arranged in a separate pile on a plain background, piles spaced apart, natural lighting, photorealistic, 4K quality",
        "Bell_peppers": f"A natural photograph of exactly {num_word} bell peppers, only these in sharp focus with no other objects present, placed at a noticeable distance from each other in random positions, sharp and clear background, natural lighting, unique and varied background setting, high resolution, photorealistic detail, 4K quality",
        "Brussel_sprouts": f"A natural photograph of exactly {num_word} Brussels sprouts, only these in sharp focus with no other objects present, placed at a noticeable distance from each other in random positions, sharp and clear background, natural lighting, unique and varied background setting, high resolution, photorealistic detail, 4K quality",
        "Galia_melons": f"A natural photograph of exactly {num_word} Galia melons, only these in sharp focus with no other objects present, placed at a noticeable distance from each other in random positions, sharp and clear background, natural lighting, unique and varied background setting, high resolution, photorealistic detail, 4K quality",
        "Passion_fruits": f"A natural photograph of exactly {num_word} whole passion fruits, only these in sharp focus, no other objects present, spaced apart in random positions, sharp clear background, natural lighting, unique varied background, high resolution, photorealistic, 4K quality",
        "Plums": f"A natural photograph of exactly {num_word} plums, only these in sharp focus, no other objects present, spaced apart in random positions, with distinct oval shape unlike cherries, sharp clear background, natural lighting, unique varied background, photorealistic, 4K",
        "Raspberries": f"A natural photograph of exactly {num_word} raspberries, only these in sharp focus, no other objects present, spaced apart in random positions, sharp clear background, natural lighting, unique varied background, high resolution, photorealistic, 4K quality",
        "Blueberries": f"A natural photograph of exactly {num_word} blueberries, only these in sharp focus, no other objects present, spaced apart in random positions, sharp clear background, natural lighting, unique varied background, high resolution, photorealistic, 4K quality",
        "Garlic": f"A natural photograph of exactly {num_word} garlic bulbs, only these in sharp focus, no other objects present, spaced apart in random positions, sharp clear background, natural lighting, unique varied background, high resolution, photorealistic, 4K",
        "Radish": f"A natural photograph of exactly {num_word} long or round white radishes, only these in sharp focus with no other objects present, placed at a noticeable distance from each other in random positions, sharp and clear background, natural lighting, unique and varied background setting, high resolution, photorealistic detail, 4K quality"
    }
    
    return prompts.get(class_name, f"A natural photograph of {num_word} {class_name}, photorealistic, 4K quality")

if __name__ == "__main__":
    main()
