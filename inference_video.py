import os
import argparse
import cv2
import jax.numpy as jnp
from jax import random
import numpy as np
from models import load_model
from utils import denormalize

def process_frame(frame, model, params):
    """
    Process a single video frame using the pre-trained model.
    """
    frame_resized = cv2.resize(frame, (256, 256))
    frame_normalized = frame_resized.astype(np.float32) / 127.5 - 1.0
    frame_jnp = jnp.expand_dims(frame_normalized, axis=0)
    
    # Apply the model
    output_jnp = model.apply(params, frame_jnp)
    output_img = denormalize(np.array(output_jnp[0]))
    
    return output_img

def main():
    """
    Main function to perform inference on a video using a pre-trained CycleGAN model.
    """
    parser = argparse.ArgumentParser(description="Inference script for CycleGAN video processing.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model parameters.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--output_dir", type=str, default="results/outputs", help="Directory to save the output video.")
    
    args = parser.parse_args()
    model_path = args.model_path
    input_video_path = args.input_video
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_preds.mp4")

    # Load the model
    print("Loading model...")
    key = random.PRNGKey(0)
    model, params = load_model(model_path, key)

    # Read input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 256
    height = 256
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        output_frame = process_frame(frame, model, params)
        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR) 
        out.write(output_frame_bgr)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved at: {output_video_path}")

if __name__ == "__main__":
    main()