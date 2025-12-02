# ================================================
# Demo Script for Professional Headshot Classifier
# Synthetic Data Generation + Automatic Labeling
# ================================================

import torch
import numpy as np
from PIL import Image
import cv2

from diffusers import StableDiffusionPipeline
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------
# 1. Load Models (placeholders or real if installed)
# ------------------------------------------------

print("Initializing models...")

# Synthetic generator (Stable Diffusion or LoRA)
try:
    generator = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cpu")
    print("Loaded Stable Diffusion generator.")
except Exception:
    generator = None
    print("Generator not loaded (running in mock mode).")

# Identity model (ArcFace placeholder)
class ArcFaceMock:
    def __call__(self, img1, img2):
        return np.random.uniform(0.7, 1.0)
arcface = ArcFaceMock()

# Professionalism classifier placeholder
class ClassifierMock(torch.nn.Module):
    def forward(self, x):
        professionalism = torch.rand(1).item()
        return professionalism

classifier = ClassifierMock()

print("Models initialized.\n")

# ------------------------------------------------
# 2. Synthetic Data Generation (Mock)
# ------------------------------------------------

def generate_synthetic_variant(prompt, style="professional"):
    print(f"Generating synthetic: {style}...")

    if generator:
        img = generator(prompt).images[0]
        return img
    else:
        # fallback placeholder
        return Image.new("RGB", (512, 512), color="gray")

# ------------------------------------------------
# 3. Automatic Labeling (Mock Implementation)
# ------------------------------------------------

def auto_label(image):
    return {
        "lighting_score": np.random.uniform(0.3, 1.0),
        "background_complexity": np.random.uniform(0.1, 0.9),
        "sharpness": np.random.uniform(0.4, 1.0),
        "clip_similarity": np.random.uniform(0.5, 0.95)
    }

# ------------------------------------------------
# 4. Classifier Inference
# ------------------------------------------------

def model_inference(image):
    x = torch.rand((1, 3, 224, 224))  # mock input
    score = classifier(x)
    return score

# ------------------------------------------------
# 5. Run Example Pipeline
# ------------------------------------------------

if __name__ == "__main__":
    print("Running demo...\n")

    # Example 1: Professional variant
    img1 = generate_synthetic_variant(
        prompt="Professional corporate headshot, studio lighting, neutral background",
        style="professional"
    )

    labels1 = auto_label(img1)
    print("Labels for Example 1:", labels1)

    score1 = model_inference(img1)
    print("Professionalism Score:", score1, "\n")

    # Example 2: Casual variant
    img2 = generate_synthetic_variant(
        prompt="Casual selfie, cluttered background, phone camera",
        style="casual"
    )

    labels2 = auto_label(img2)
    print("Labels for Example 2:", labels2)

    score2 = model_inference(img2)
    print("Professionalism Score:", score2, "\n")

    print("Demo completed successfully.")
