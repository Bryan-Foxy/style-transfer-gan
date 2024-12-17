import os
import argparse
import jax.numpy as jnp
from jax import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import load_model
from utils import load_and_preprocess_image, denormalize

def main():
    """
    Main function to perform inference on an input image using a pre-trained CycleGAN model.
    """

    # Argument parser to parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference script for CycleGAN model.")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model parameters (e.g., checkpoints/params_F.msgpack).")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to the input image to perform inference on.")
    parser.add_argument("--output_dir", type=str, default="results/outputs",
                        help="Directory where the output image will be saved (default: 'results').")
    
    # Parse the arguments
    args = parser.parse_args()

    # Extract arguments
    model_path = args.model_path
    input_image_path = args.input_image
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the random key for JAX
    key = random.PRNGKey(0)

    # Load the pre-trained model
    print("Loading model...")
    model, params = load_model(model_path, key)

    # Preprocess the input image
    print(f"Preprocessing input image: {input_image_path}")
    image = load_and_preprocess_image(input_image_path)
    image_jnp = jnp.expand_dims(image, axis=0)  # Add batch dimension

    # Perform inference
    print("Performing inference...")
    output_jnp = model.apply(params, image_jnp)
    output_img = denormalize(np.array(output_jnp[0]))  

    # Save the predicted image
    input_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_dir, f"{input_filename}_preds.png")
    Image.fromarray(output_img).save(output_path)

    # Display the predicted image
    plt.imshow(output_img)
    plt.title("Predicted Image")
    plt.axis("off")
    plt.show()

    print(f"Predicted image saved at: {output_path}")

if __name__ == "__main__":
    main()