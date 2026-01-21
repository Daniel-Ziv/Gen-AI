#!/usr/bin/env python3
"""
Synthetic Portrait Dataset Generator for ViT Training

This script generates a labeled synthetic image dataset for training a Vision Transformer
to detect professional portrait attributes. It processes FFHQ images, generates transformed
portraits using Google's Gemini/Imagen API, and creates ground-truth binary label vectors.

The pipeline is designed to be:
- Resumable: Safe to restart without reprocessing completed images
- Concurrent: Async processing with configurable parallelism
- Reproducible: Seeds stored for deterministic regeneration
- Production-ready: Comprehensive error handling, logging, and retry logic

Usage:
    python synthetic_dataset_generator.py --input_dir /path/to/ffhq --output_dir /path/to/output

Author: Generated for Gen-AI project
"""

import os
import sys
import json
import random
import hashlib
import logging
import argparse
import asyncio
import time
import io
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Iterator
from abc import ABC, abstractmethod
from enum import Enum

# Load .env file if it exists
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Third-party imports
try:
    from PIL import Image
    from google import genai
    from google.genai import types
    import numpy as np
    from datasets import load_dataset
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install pillow google-genai numpy datasets")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Central configuration for the entire pipeline.

    Separating configuration from logic allows easy adjustment of parameters
    without modifying core code. This is especially important for:
    - Tuning the probability of "perfect" (all-1s) label vectors
    - Adjusting retry behavior based on API reliability
    - Switching between development and production settings
    """

    # HuggingFace dataset configuration
    hf_dataset: str = "marcosv/ffhq-dataset"
    hf_split: str = "train"

    # Path configuration
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    log_file: Path = field(default_factory=lambda: Path("./output/processing_log.txt"))

    # API configuration
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash-image"  # Gemini 2.5 Flash with image generation
    api_timeout: int = 120  # Seconds to wait for API response

    # Concurrency configuration
    max_concurrent_requests: int = 5  # Parallel API calls
    rate_limit_delay: float = 0.1  # Delay between launching requests

    # Retry configuration - conservative defaults to handle transient failures
    max_retries: int = 3
    retry_delay_base: float = 2.0  # Exponential backoff base in seconds
    retry_delay_max: float = 60.0  # Maximum delay between retries

    # Label generation configuration - Three-bucket distribution
    # 15% perfect (all 1s) + 15% near-perfect (one 0) + 70% mixed random
    # This maximizes information per example while maintaining ~50% balance per label
    perfect_vector_probability: float = 0.15
    near_perfect_probability: float = 0.15

    # Prompt variation - prevents ViT from overfitting to exact phrasing
    prompt_variation_enabled: bool = True
    attribute_dropout_probability: float = 0.1  # Chance to drop an attribute phrase

    # Output configuration
    output_format: str = "JPEG"  # JPEG is smaller, PNG if lossless needed
    jpeg_quality: int = 95
    output_image_size: tuple[int, int] = (512, 512)  # Normalize all outputs

    # Cost tracking (Gemini 2.5 Flash image generation pricing)
    cost_per_image: float = 0.039  # $0.039 per image for Flash model

    # Logging configuration
    log_level: int = logging.INFO

    def __post_init__(self):
        """Validate and convert paths after initialization."""
        self.output_dir = Path(self.output_dir)
        self.log_file = Path(self.log_file)


# =============================================================================
# LABEL DEFINITIONS
# =============================================================================

class AttributeIndex(Enum):
    """
    Enumeration of all portrait attributes with their vector indices.

    Using an enum provides:
    - Type safety when accessing specific attributes
    - Self-documenting code that clearly maps indices to meanings
    - Easy iteration over all attributes
    """
    LIGHTING_EVEN_FRONTAL = 0
    BACKGROUND_CLEAN = 1
    BUSINESS_ATTIRE_VISIBLE = 2
    NEUTRAL_PROFESSIONAL_EXPRESSION = 3
    FACE_PROPERLY_FRAMED = 4
    IMAGE_SHARPNESS_HIGH = 5  # Applied post-generation, not via prompts


@dataclass
class AttributeDefinition:
    """
    Defines how each attribute translates to prompt text.

    Each attribute has multiple prompt variants for both positive and negative
    cases to prevent the ViT from overfitting to exact phrasing.
    """
    name: str
    display_name: str
    positive_prompts: list[str]  # Multiple variants to randomize
    negative_prompts: list[str]  # Multiple variants to randomize


# Attribute definitions with multiple phrasings to prevent overfitting
# The ViT should learn visual patterns, not specific text artifacts
ATTRIBUTE_DEFINITIONS: dict[int, AttributeDefinition] = {
    AttributeIndex.LIGHTING_EVEN_FRONTAL.value: AttributeDefinition(
        name="lighting_even_frontal",
        display_name="Lighting Even and Frontal",
        positive_prompts=[
            # Technical descriptions
            "modify the lighting to a soft butterfly lighting setup with even illumination",
            "relight the subject using large softbox lighting from the front to minimize shadows",
            "apply high-key fashion lighting with balanced fill lights",
            "ensure the face is evenly lit with a beauty dish setup, removing harsh contrast",
            "simulate natural window light coming from the front, creating soft wrapping light",
            "use a ring light effect to create shadowless, even facial illumination",
            "change to a broad lighting setup with gentle, diffused highlights",
            "lighting should be flat and flattering, typical of commercial headshots",
            "remove all deep shadows and replace with soft, wrap-around studio lighting",
            "use three-point lighting but keep the key and fill lights balanced for an even look",
        ],
        negative_prompts=[
            # IMPROVED: Very explicit bad lighting descriptions
            "Half the face must be in COMPLETE DARKNESS. Only one eye visible. The other side of the face is a black silhouette. Extreme split lighting like a thriller movie poster.",
            "Harsh overhead lighting creating DEEP BLACK shadows under the eyes, nose, and chin. The eye sockets should look like dark holes. Unflattering fluorescent office lighting.",
            "Strong side lighting from the left: the right side of the face should be completely BLACK and invisible. Dramatic film noir style with extreme contrast.",
            "Underexposed photo with the face barely visible in shadows. Very dark, moody lighting where facial features are hard to see.",
            "Harsh direct flash creating blown-out highlights on forehead and nose, with deep shadows everywhere else. Amateur snapshot lighting.",
            "One side of face brightly lit, other side in TOTAL SHADOW. At least 50% of the face should be hidden in darkness.",
        ]
    ),

    AttributeIndex.BACKGROUND_CLEAN.value: AttributeDefinition(
        name="background_clean",
        display_name="Background Clean and Non-Distracting",
        positive_prompts=[
            # Color and Texture variety prevents overfitting to "Grey"
            "replace the background with a seamless charcoal grey studio paper",
            "change the background to a clean, solid white backdrop",
            "insert a blurred professional office background (bokeh effect)",
            "place the subject against a solid dark blue textured canvas",
            "use a soft, neutral gradient background, fading from light to dark grey",
            "background should be a solid, muted beige wall with no objects",
            "clean up the background completely, replacing it with an out-of-focus abstract pattern",
            "place the subject in front of a modern, blurry glass architectural element",
            "use a minimal off-white plaster wall as the background",
            "ensure the background is purely negative space, solid black or dark grey",
        ],
        negative_prompts=[
            "place the subject in a messy living room with visible clutter",
            "change background to a busy street scene with cars and pedestrians",
            "background should be a crowded cafe with people visible behind",
            "add a complex pattern wallpaper that distracts from the face",
            "place the subject in a disorganized office with piles of paper visible",
            "background should show a dense forest with high-frequency leaf textures",
            "include bright neon signs and distracting lights in the background",
            "background is a grocery store aisle with visible products",
        ]
    ),

    AttributeIndex.BUSINESS_ATTIRE_VISIBLE.value: AttributeDefinition(
        name="business_attire_visible",
        display_name="Business or Professional Attire Visible",
        positive_prompts=[
            # Fabric and Style variety prevents overfitting to "Blue Suit"
            "digitally dress the subject in a tailored navy blue wool suit",
            "change clothing to a charcoal grey blazer with a crisp white dress shirt",
            "wearing a formal black tuxedo jacket and bowtie",
            "dressed in a smart casual beige linen blazer and button-down",
            "wearing a high-end silk blouse suitable for corporate profiling",
            "dressed in a formal pinstripe suit with a professional tie",
            "wearing a smart turtleneck sweater under a tweed jacket",
            "change outfit to a professional grey vest and dress shirt combination",
            "dressed in a dark blazer with a subtle check pattern",
            "wearing formal corporate attire, structured shoulders and clean lines",
        ],
        negative_prompts=[
            "change clothing to a graphic vintage t-shirt",
            "dressed in a casual oversized hoodie with logos",
            "wearing a sports jersey and gym clothes",
            "dressed in a denim jacket and casual flannel shirt",
            "wearing a tank top or sleeveless undershirt",
            "change outfit to a floral summer dress with spaghetti straps",
            "dressed in a heavy winter parka or outdoor coat",
            "wearing pajamas or loungewear",
        ]
    ),

    AttributeIndex.NEUTRAL_PROFESSIONAL_EXPRESSION.value: AttributeDefinition(
        name="neutral_professional_expression",
        display_name="Neutral Professional Facial Expression",
        positive_prompts=[
            "modify the expression to be calm, confident, and approachable",
            "relax the face into a neutral, serious professional gaze",
            "ensure the subject has a slight, polite smile (Duchenne smile)",
            "expression should be focused and attentive, looking at the camera",
            "soften the facial features to appear composed and trustworthy",
            "change expression to a subtle, closed-mouth smile",
            "remove any extreme emotion, leaving a blank but pleasant canvas",
            "subject should look authoritative and calm",
        ],
        negative_prompts=[
            # IMPROVED: Very explicit bad expression descriptions
            "Mouth WIDE OPEN laughing hard showing ALL teeth visible, eyes squeezed shut from laughing. Very exaggerated animated hysterical laughter.",
            "ANGRY SCOWL: eyebrows pushed down and together, eyes narrowed into slits, mouth frowning hard. Visibly upset, hostile, aggressive expression.",
            "SHOCKED SURPRISED: eyebrows raised as HIGH as possible, eyes wide open showing whites, mouth forming a big O shape. Exaggerated cartoon surprise.",
            "CRYING with visible tears on cheeks, red puffy eyes, mouth turned down in a sob. Clearly distressed and emotional.",
            "SILLY GOOFY face: tongue sticking out, one eye winking, making a ridiculous childish expression. Not professional at all.",
            "YAWNING with mouth stretched wide open, eyes half-closed, looking exhausted and bored. Unprofessional tired expression.",
        ]
    ),

    AttributeIndex.FACE_PROPERLY_FRAMED.value: AttributeDefinition(
        name="face_properly_framed",
        display_name="Face Properly Framed and Centered",
        positive_prompts=[
            "re-crop the image to a standard head-and-shoulders composition",
            "ensure the eyes are positioned at the top third line (rule of thirds)",
            "center the face perfectly in the frame with balanced headroom",
            "zoom out slightly to show the full head and shoulders comfortably",
            "adjust framing to be symmetrical and passport-style",
            "compose the shot as a classic portrait, filling the frame with the face",
        ],
        negative_prompts=[
            # IMPROVED: Very explicit bad framing descriptions
            "EXTREME CLOSE-UP: only the nose and mouth visible, forehead and eyes completely CROPPED OUT of the frame. Way too zoomed in.",
            "Face pushed to the FAR LEFT EDGE of the frame, huge empty space on the right side taking up 70% of the image. Very off-center.",
            "TOP OF HEAD CUT OFF by the frame edge. Forehead and hair not visible at all. Bad amateur cropping.",
            "TINY FACE in a huge frame: person is very far away, face takes up less than 15% of the image. Too much empty space everywhere.",
            "Face at the VERY BOTTOM of frame with massive empty space above taking up 60% of the image. Way too much headroom.",
            "Person positioned in the far right corner, face partially cut off at the edge. Awkward off-center composition.",
        ]
    ),
    
    # Sharpness handled by code, placeholders only
    AttributeIndex.IMAGE_SHARPNESS_HIGH.value: AttributeDefinition(
        name="image_sharpness_high",
        display_name="Image Sharpness High",
        positive_prompts=[""], negative_prompts=[""]
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LabelVector:
    """
    Represents the 6-element binary label vector for an image.

    This class provides convenient accessors and serialization while
    maintaining the raw vector for numerical operations.
    """
    values: list[int]

    def __post_init__(self):
        """Validate vector length and values."""
        if len(self.values) != 6:
            raise ValueError(f"Label vector must have exactly 6 elements, got {len(self.values)}")
        if not all(v in (0, 1) for v in self.values):
            raise ValueError("All label values must be 0 or 1")

    def to_binary_string(self) -> str:
        """Convert to compact binary string representation for filenames."""
        return "".join(str(v) for v in self.values)

    def is_perfect(self) -> bool:
        """Check if all attributes are professionally correct."""
        return all(v == 1 for v in self.values)

    def to_dict(self) -> dict[str, int]:
        """Convert to human-readable dictionary with attribute names."""
        return {
            ATTRIBUTE_DEFINITIONS[i].display_name: v
            for i, v in enumerate(self.values)
        }

    def __getitem__(self, index: int) -> int:
        return self.values[index]


@dataclass
class ImageMetadata:
    """
    Complete metadata for a generated image.

    This captures everything needed to:
    - Reproduce the generation (original filename, prompt, labels, seed)
    - Train the ViT (label vector)
    - Debug issues (timestamp, processing details)
    - Track costs (generation cost)
    """
    original_filename: str
    label_vector: list[int]
    label_names: dict[str, int]
    prompt_text: str
    generation_seed: int  # For reproducibility
    timestamp: str
    output_filename: str
    cost_usd: float  # Track generation cost
    blur_type: Optional[str] = None  # Type of blur applied (if sharpness=0)

    def to_json(self) -> str:
        """Serialize to formatted JSON for storage."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageMetadata":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)


# =============================================================================
# DATASET LOADING (HUGGING FACE STREAMING)
# =============================================================================

class HFDatasetStreamer:
    """
    Streams FFHQ images directly from Hugging Face Hub.

    Design considerations:
    - Zero local storage: Images are streamed on-demand over the network
    - Memory efficient: Only one image in memory at a time
    - Resumable: Uses image index for deterministic ordering
    """

    def __init__(
        self,
        dataset_name: str = "marcosv/ffhq-dataset",
        split: str = "train",
        limit: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self._dataset = None

    def _load_dataset(self):
        """Lazily load the streaming dataset."""
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=True
            )
        return self._dataset

    def stream_images(self) -> "Iterator[tuple[str, Image.Image]]":
        """
        Stream images from HuggingFace.

        Yields:
            Tuple of (image_id, PIL.Image) for each image in the dataset.
            image_id is a synthetic filename like "ffhq_00001.png"
        """
        dataset = self._load_dataset()

        if self.limit is not None:
            dataset = dataset.take(self.limit)

        for idx, item in enumerate(dataset):
            # Generate a consistent image ID for tracking
            image_id = f"ffhq_{idx:05d}.png"

            # Skip items without images (metadata files, etc.)
            if 'image' not in item or item['image'] is None:
                continue

            # Get the PIL image from the dataset
            try:
                img_data = item['image']
                if isinstance(img_data, Image.Image):
                    image = img_data.convert("RGB")
                elif hasattr(img_data, 'convert'):
                    image = img_data.convert("RGB")
                else:
                    # Try to handle different dataset formats
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")

                # Skip tiny/corrupt images that could cause API errors
                if image.size[0] < 128 or image.size[1] < 128:
                    continue

            except Exception:
                # Skip corrupt or unreadable images
                continue

            yield image_id, image

    def get_image_count(self) -> Optional[int]:
        """
        Return the limit if set, otherwise None (unknown for streaming).
        """
        return self.limit


# =============================================================================
# LABEL VECTOR GENERATION
# =============================================================================

class LabelVectorGenerator:
    """
    Generates random binary label vectors with controlled three-bucket distribution.

    Distribution strategy:
    - Perfect (15%): All labels are 1. Teaches the model what an ideal
      professional photo looks like globally.
    - Near-perfect (15%): Exactly one label is 0, the rest are 1. Forces the
      model to learn each attribute independently, even when everything else
      looks professional. This prevents relying on "global professional vibe".
    - Mixed random (70%): Random binary values with balanced per-label
      distribution. Provides diversity and breaks correlations.

    This distribution ensures:
    - Each label ~50% positive overall
    - Strong positive examples via perfect bucket
    - Clean single-attribute counterexamples via near-perfect bucket
    - General diversity via mixed bucket
    """

    NUM_LABELS = 6

    def __init__(
        self,
        perfect_probability: float = 0.15,
        near_perfect_probability: float = 0.15,
        seed: Optional[int] = None
    ):
        """
        Initialize the generator with three-bucket distribution.

        Args:
            perfect_probability: Probability of generating all-1s vector (default 15%).
            near_perfect_probability: Probability of generating exactly-one-0 vector (default 15%).
            seed: Random seed for reproducibility. If None, uses system entropy.

        The remaining probability (1 - perfect - near_perfect) goes to mixed random.
        """
        if perfect_probability + near_perfect_probability > 1.0:
            raise ValueError(
                f"perfect_probability ({perfect_probability}) + "
                f"near_perfect_probability ({near_perfect_probability}) cannot exceed 1.0"
            )

        self.perfect_probability = perfect_probability
        self.near_perfect_probability = near_perfect_probability
        self.rng = random.Random(seed)

    def generate(self) -> LabelVector:
        """
        Generate a single label vector using three-bucket distribution.

        - With probability `perfect_probability`: returns all 1s
        - With probability `near_perfect_probability`: returns exactly one 0
        - Otherwise: returns uniformly random binary values
        """
        roll = self.rng.random()

        if roll < self.perfect_probability:
            # Perfect bucket: all attributes are professionally correct
            values = [1] * self.NUM_LABELS

        elif roll < self.perfect_probability + self.near_perfect_probability:
            # Near-perfect bucket: exactly one failing attribute
            # Pick which label is 0 uniformly at random
            failing_index = self.rng.randint(0, self.NUM_LABELS - 1)
            values = [1] * self.NUM_LABELS
            values[failing_index] = 0

        else:
            # Mixed random bucket: independent 50% chance per attribute
            values = [self.rng.randint(0, 1) for _ in range(self.NUM_LABELS)]

        return LabelVector(values)

    def generate_batch(self, count: int) -> list[LabelVector]:
        """Generate multiple label vectors."""
        return [self.generate() for _ in range(count)]


# =============================================================================
# PROMPT CONSTRUCTION WITH VARIATION
# =============================================================================

class PromptBuilder:
    """
    Constructs API prompts from label vectors with randomization.

    Key features:
    - Random selection from multiple phrasings per attribute
    - Optional attribute dropout to prevent overfitting
    - Maintains visual clarity while adding variation
    """

    # Base prompt establishing the overall intent
    # EXPANDED BASE PROMPTS (15 variants across 4 "Personas")
    BASE_PROMPTS = [
        # Persona 1: The Corporate Re-brander (Focus on professional standards)
        "Using the reference image as the source identity, modify this portrait to meet strict corporate profile standards. ",
        "Update the provided headshot to be suitable for a Fortune 500 company website, keeping the subject's identity intact. ",
        "Transform this casual photo into a high-end executive portrait, strictly maintaining the subject's facial structure. ",
        "Edit the reference photo to create a polished LinkedIn profile picture that conveys trustworthiness and competence. ",

        # Persona 2: The Studio Photographer (Focus on lighting/camera)
        "Retaining the subject's identity, re-imagine this photo as if it were taken in a high-end portrait studio. ",
        "Apply a professional studio photography treatment to the attached reference image, improving composition and fidelity. ",
        "Edit this image to simulate a professional photoshoot with an 85mm portrait lens, keeping the face exactly as is. ",
        "Refine the visual quality of this reference image to match the standards of high-fidelity commercial photography. ",

        # Persona 3: The Magazine Editor (Focus on aesthetics/style)
        "Give this reference image a professional editorial makeover suitable for a business magazine feature. ",
        "Upgrade the aesthetic of this photo to look like a premium personal branding shot, without altering the person's features. ",
        "Process the reference image to create a clean, modern, and approachable professional headshot. ",
        "Enhance the provided image to look like a high-quality author bio photo on a book jacket. ",

        # Persona 4: The Technical Retoucher (Focus on fixing flaws)
        "Fix the amateur elements of this reference photo to create a clean professional output, locking the facial identity. ",
        "Digitally remaster the attached image into a formal business portrait, correcting environment and lighting issues. ",
        "Convert the style of this reference image into a formal corporate headshot, focusing on clarity and professionalism. ",
    ]

    def __init__(
        self,
        enable_variation: bool = True,
        dropout_probability: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the prompt builder.

        Args:
            enable_variation: If True, randomly selects from prompt variants
            dropout_probability: Chance to drop each attribute phrase (0.1 = 10%)
            seed: Random seed for reproducibility
        """
        self.enable_variation = enable_variation
        self.dropout_probability = dropout_probability
        self.rng = random.Random(seed)

    def build_prompt(self, labels: LabelVector, seed: Optional[int] = None) -> str:
        """
        Construct a complete prompt from a label vector.

        If seed is provided, uses it for deterministic prompt generation.
        Otherwise uses the builder's RNG.
        """
        rng = random.Random(seed) if seed is not None else self.rng

        # Select base prompt
        if self.enable_variation:
            base = rng.choice(self.BASE_PROMPTS)
        else:
            base = self.BASE_PROMPTS[0]

        prompt_parts = [base, "The portrait shows: "]
        attribute_phrases = []

        # Add attribute-specific descriptions with variation
        for index, value in enumerate(labels.values):
            # Skip sharpness attribute entirely - always generate sharp images
            # Blur is applied post-generation programmatically
            if index == AttributeIndex.IMAGE_SHARPNESS_HIGH.value:
                continue

            # Skip framing attribute for negative values - always generate well-framed
            # Bad framing is applied post-generation programmatically
            if index == AttributeIndex.FACE_PROPERLY_FRAMED.value and value == 0:
                continue

            # Optional dropout: skip this attribute phrase
            if self.enable_variation and rng.random() < self.dropout_probability:
                continue

            definition = ATTRIBUTE_DEFINITIONS[index]

            if self.enable_variation:
                if value == 1:
                    phrase = rng.choice(definition.positive_prompts)
                else:
                    phrase = rng.choice(definition.negative_prompts)
            else:
                if value == 1:
                    phrase = definition.positive_prompts[0]
                else:
                    phrase = definition.negative_prompts[0]

            attribute_phrases.append(phrase)

        # Join with appropriate punctuation
        if attribute_phrases:
            prompt_parts.append(", ".join(attribute_phrases))
            prompt_parts.append(".")
        else:
            # Fallback if all attributes were dropped
            prompt_parts.append("a professional looking person.")

        return "".join(prompt_parts)


# =============================================================================
# BLUR AUGMENTATION FOR SHARPNESS LABEL
# =============================================================================

class BlurType(Enum):
    """Types of blur to apply for sharpness=0 images."""
    GAUSSIAN = "gaussian"
    MOTION = "motion"
    DEFOCUS = "defocus"
    DOWNSCALE_UPSCALE = "downscale_upscale"


class BlurAugmenter:
    """
    Applies realistic blur to images when sharpness label is 0.

    This is applied POST-GENERATION to ensure clean supervision.
    Using multiple blur types prevents the ViT from learning
    specific blur kernels instead of true perceptual sharpness.

    Blur types:
    - Gaussian: Standard soft blur (out of focus simulation)
    - Motion: Directional blur (camera shake / subject movement)
    - Defocus: Circular bokeh-style blur (lens defocus)
    - Downscale-upscale: Resolution degradation blur
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def apply_blur(self, image: Image.Image, seed: Optional[int] = None) -> tuple[Image.Image, str]:
        """
        Apply a random blur type to the image.

        Args:
            image: PIL Image to blur
            seed: Optional seed for deterministic blur selection

        Returns:
            Tuple of (blurred_image, blur_type_name)
        """
        rng = random.Random(seed) if seed is not None else self.rng

        # Randomly select blur type
        blur_type = rng.choice(list(BlurType))

        if blur_type == BlurType.GAUSSIAN:
            return self._apply_gaussian_blur(image, rng), blur_type.value
        elif blur_type == BlurType.MOTION:
            return self._apply_motion_blur(image, rng), blur_type.value
        elif blur_type == BlurType.DEFOCUS:
            return self._apply_defocus_blur(image, rng), blur_type.value
        else:  # DOWNSCALE_UPSCALE
            return self._apply_downscale_upscale_blur(image, rng), blur_type.value

    def _apply_gaussian_blur(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply Gaussian blur with random radius."""
        from PIL import ImageFilter

        # Radius 3.5-6.0: Obviously blurry but not destroyed
        # Avoids "ambiguous zone" where subtle blur could confuse the model
        radius = rng.uniform(3.5, 6.0)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_motion_blur(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply motion blur in a random direction."""
        from PIL import ImageFilter

        # Create motion blur kernel
        # Size 15-30: Obviously motion-blurred, not subtle
        size = rng.randint(15, 30)
        angle = rng.uniform(0, 180)  # Random angle

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        center = size // 2

        # Draw a line through the center at the given angle
        angle_rad = np.deg2rad(angle)
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1

        # Normalize kernel
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel

        # Apply using convolution
        kernel_flat = kernel.flatten().tolist()
        motion_filter = ImageFilter.Kernel(
            size=(size, size),
            kernel=kernel_flat,
            scale=1,
            offset=0
        )

        try:
            return image.filter(motion_filter)
        except ValueError:
            # Fallback to gaussian if kernel fails
            return self._apply_gaussian_blur(image, rng)

    def _apply_defocus_blur(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply circular defocus blur (bokeh-like)."""
        from PIL import ImageFilter

        # Create circular kernel for defocus blur
        # Size 13-21: Obviously out-of-focus, not subtle
        size = rng.randint(13, 21)  # Must be odd
        if size % 2 == 0:
            size += 1

        center = size // 2
        radius = center - 1

        kernel = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                    kernel[y, x] = 1

        # Normalize
        kernel = kernel / kernel.sum()

        kernel_flat = kernel.flatten().tolist()
        defocus_filter = ImageFilter.Kernel(
            size=(size, size),
            kernel=kernel_flat,
            scale=1,
            offset=0
        )

        try:
            return image.filter(defocus_filter)
        except ValueError:
            # Fallback to gaussian if kernel fails
            return self._apply_gaussian_blur(image, rng)

    def _apply_downscale_upscale_blur(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply blur by downscaling then upscaling (resolution degradation)."""
        original_size = image.size

        # Downscale factor 3x-5x: Obviously pixelated/soft, not subtle
        factor = rng.uniform(3.0, 5.0)
        small_size = (int(original_size[0] / factor), int(original_size[1] / factor))

        # Ensure minimum size
        small_size = (max(small_size[0], 32), max(small_size[1], 32))

        # Downscale with low quality resampling
        small = image.resize(small_size, Image.Resampling.BILINEAR)

        # Upscale back to original size
        return small.resize(original_size, Image.Resampling.BILINEAR)


# =============================================================================
# FRAMING AUGMENTATION FOR FRAMING LABEL
# =============================================================================

class FramingType(Enum):
    """Types of bad framing to apply for framing=0 images."""
    OFF_CENTER_LEFT = "off_center_left"
    OFF_CENTER_RIGHT = "off_center_right"
    OFF_CENTER_TOP = "off_center_top"
    EXTREME_CLOSEUP = "extreme_closeup"
    TOO_FAR = "too_far"
    HEAD_CUTOFF = "head_cutoff"


class FramingAugmenter:
    """
    Applies bad framing to images when framing label is 0.

    This is applied POST-GENERATION to ensure clean supervision.
    The generator often ignores bad framing prompts and centers the face anyway.
    By applying bad framing programmatically, we guarantee the label is correct.

    Framing types:
    - Off-center (left/right/top): Shift the face away from center
    - Extreme close-up: Crop to just part of the face
    - Too far: Add padding to make face small in frame
    - Head cutoff: Crop the top of the head
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def apply_bad_framing(self, image: Image.Image, seed: Optional[int] = None) -> tuple[Image.Image, str]:
        """
        Apply a random bad framing type to the image.

        Args:
            image: PIL Image to reframe
            seed: Optional seed for deterministic framing selection

        Returns:
            Tuple of (badly_framed_image, framing_type_name)
        """
        rng = random.Random(seed) if seed is not None else self.rng

        # Randomly select framing type
        framing_type = rng.choice(list(FramingType))

        if framing_type == FramingType.OFF_CENTER_LEFT:
            return self._apply_off_center(image, rng, direction="left"), framing_type.value
        elif framing_type == FramingType.OFF_CENTER_RIGHT:
            return self._apply_off_center(image, rng, direction="right"), framing_type.value
        elif framing_type == FramingType.OFF_CENTER_TOP:
            return self._apply_off_center(image, rng, direction="top"), framing_type.value
        elif framing_type == FramingType.EXTREME_CLOSEUP:
            return self._apply_extreme_closeup(image, rng), framing_type.value
        elif framing_type == FramingType.TOO_FAR:
            return self._apply_too_far(image, rng), framing_type.value
        else:  # HEAD_CUTOFF
            return self._apply_head_cutoff(image, rng), framing_type.value

    def _apply_off_center(self, image: Image.Image, rng: random.Random, direction: str) -> Image.Image:
        """Shift the subject off-center by cropping and padding."""
        w, h = image.size

        # Shift by 25-40% of the image dimension
        shift_pct = rng.uniform(0.25, 0.40)

        if direction == "left":
            # Crop right side, pad left side (face ends up on right)
            crop_amount = int(w * shift_pct)
            cropped = image.crop((crop_amount, 0, w, h))
            new_img = Image.new("RGB", (w, h), color=(128, 128, 128))
            new_img.paste(cropped, (0, 0))

        elif direction == "right":
            # Crop left side, pad right side (face ends up on left)
            crop_amount = int(w * shift_pct)
            cropped = image.crop((0, 0, w - crop_amount, h))
            new_img = Image.new("RGB", (w, h), color=(128, 128, 128))
            new_img.paste(cropped, (crop_amount, 0))

        else:  # top - face ends up at bottom
            crop_amount = int(h * shift_pct)
            cropped = image.crop((0, crop_amount, w, h))
            new_img = Image.new("RGB", (w, h), color=(128, 128, 128))
            new_img.paste(cropped, (0, 0))

        return new_img

    def _apply_extreme_closeup(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Crop to extreme close-up showing only part of face."""
        w, h = image.size

        # Crop to 30-45% of original size, centered on a random part
        crop_ratio = rng.uniform(0.30, 0.45)
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)

        # Random offset but biased toward center (where face likely is)
        max_offset_x = (w - crop_w) // 2
        max_offset_y = (h - crop_h) // 2

        offset_x = w // 2 - crop_w // 2 + rng.randint(-max_offset_x // 2, max_offset_x // 2)
        offset_y = h // 2 - crop_h // 2 + rng.randint(-max_offset_y // 2, max_offset_y // 2)

        # Ensure bounds
        offset_x = max(0, min(offset_x, w - crop_w))
        offset_y = max(0, min(offset_y, h - crop_h))

        cropped = image.crop((offset_x, offset_y, offset_x + crop_w, offset_y + crop_h))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)

    def _apply_too_far(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Make the subject appear too far away by adding padding."""
        w, h = image.size

        # Shrink to 35-50% of original size
        shrink_ratio = rng.uniform(0.35, 0.50)
        new_w = int(w * shrink_ratio)
        new_h = int(h * shrink_ratio)

        shrunk = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Place in a larger canvas with random background color (neutral tones)
        bg_color = (
            rng.randint(100, 180),
            rng.randint(100, 180),
            rng.randint(100, 180)
        )
        new_img = Image.new("RGB", (w, h), color=bg_color)

        # Random position (not centered, to make framing worse)
        pos_x = rng.randint(0, w - new_w)
        pos_y = rng.randint(0, h - new_h)

        new_img.paste(shrunk, (pos_x, pos_y))
        return new_img

    def _apply_head_cutoff(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Cut off the top of the head."""
        w, h = image.size

        # Cut 20-35% from the top
        cutoff_pct = rng.uniform(0.20, 0.35)
        cutoff_amount = int(h * cutoff_pct)

        # Crop from below the cutoff to bottom
        cropped = image.crop((0, cutoff_amount, w, h))

        # Resize back to original dimensions
        return cropped.resize((w, h), Image.Resampling.LANCZOS)


# =============================================================================
# API CLIENT - GEMINI/IMAGEN
# =============================================================================

class ImageGenerationError(Exception):
    """Raised when image generation fails after all retries."""
    pass


class ImageGenerationAPI(ABC):
    """
    Abstract base class for image generation APIs.

    This abstraction allows swapping different APIs without changing
    the rest of the pipeline.
    """

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate an image from a prompt, optionally using a reference."""
        pass


class NanoBananaProAPI(ImageGenerationAPI):
    """
    Implementation for Google's Nano Banana Pro (Gemini 3 Pro Image) API.

    Uses the new google-genai SDK with the unified client pattern.
    Supports:
    - Text-to-image generation with Gemini 3 Pro Image
    - High resolution output (1K, 2K, 4K)
    - Reference image input for style guidance
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-image",
        timeout: int = 120
    ):
        self.model_name = model_name
        self.timeout = timeout

        # Initialize the new unified client
        self.client = genai.Client(api_key=api_key)

    async def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image using Nano Banana Pro (Gemini 3 Pro Image).

        Args:
            prompt: Text description of the image to generate
            reference_image: Optional reference image for style/content guidance
            seed: Random seed for deterministic generation (used in prompt for consistency)
        """
        loop = asyncio.get_event_loop()

        try:
            # Build contents - text prompt, optionally with reference image
            contents = []

            if reference_image is not None:
                # Convert PIL image to bytes for the API
                img_bytes = io.BytesIO()
                reference_image.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Add reference image as input
                contents.append(types.Part.from_bytes(
                    data=img_bytes.getvalue(),
                    mime_type="image/png"
                ))
                # Add instruction to use reference
                contents.append(f"Using the provided reference image as a style guide, create: {prompt}")
            else:
                contents.append(prompt)

            # Add seed hint to prompt for reproducibility
            if seed is not None:
                if isinstance(contents[-1], str):
                    contents[-1] = f"{contents[-1]} [seed:{seed}]"

            # Configure image generation
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1"  # Flash model defaults to 1024x1024
                )
            )

            # Run generation in thread pool (SDK is synchronous)
            def _generate():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )

            response = await loop.run_in_executor(None, _generate)

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Data is raw bytes, not base64 encoded
                    image_bytes = part.inline_data.data
                    return Image.open(io.BytesIO(image_bytes))

            raise ImageGenerationError("No image returned from API")

        except Exception as e:
            raise ImageGenerationError(f"Nano Banana Pro API error: {e}") from e


# Alias for backwards compatibility
GeminiImagenAPI = NanoBananaProAPI


class MockImageGenerationAPI(ImageGenerationAPI):
    """
    Mock API for testing without actual API calls.

    Generates simple placeholder images that encode the seed visually.
    Useful for:
    - Testing the pipeline logic
    - Development without API costs
    - Validating file naming and metadata
    """

    def __init__(self, delay: float = 0.05):
        self.delay = delay  # Simulate API latency

    async def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a placeholder image with visual indicators."""
        await asyncio.sleep(self.delay)

        # Use seed for consistent colors, or derive from prompt
        if seed is None:
            seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)

        rng = random.Random(seed)

        # Create a simple gradient image as placeholder
        width, height = 512, 512
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        r_base = rng.randint(50, 200)
        g_base = rng.randint(50, 200)
        b_base = rng.randint(50, 200)

        for y in range(height):
            for x in range(width):
                r = (r_base + x // 8) % 256
                g = (g_base + y // 8) % 256
                b = (b_base + (x + y) // 16) % 256
                pixels[x, y] = (r, g, b)

        return img


# =============================================================================
# RESILIENT API CLIENT WITH RETRY AND CONCURRENCY
# =============================================================================

class ResilientAPIClient:
    """
    Wraps an image generation API with retry logic, concurrency control,
    and rate limiting.

    Features:
    - Semaphore-based concurrency limiting
    - Exponential backoff for retries
    - Rate limiting between requests
    """

    def __init__(
        self,
        api: ImageGenerationAPI,
        max_concurrent: int = 5,
        max_retries: int = 3,
        retry_delay_base: float = 2.0,
        retry_delay_max: float = 60.0,
        rate_limit_delay: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        self.api = api
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.retry_delay_max = retry_delay_max
        self.rate_limit_delay = rate_limit_delay
        self.logger = logger or logging.getLogger(__name__)

    async def generate_image(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image with concurrency control and automatic retry.

        Uses exponential backoff: delay = min(base * 2^attempt, max_delay)
        """
        async with self.semaphore:
            # Rate limiting delay
            await asyncio.sleep(self.rate_limit_delay)

            last_error = None

            for attempt in range(self.max_retries + 1):
                try:
                    return await self.api.generate_image(prompt, reference_image, seed)

                except Exception as e:
                    last_error = e

                    if attempt == self.max_retries:
                        self.logger.error(
                            f"Image generation failed after {self.max_retries + 1} attempts: {e}"
                        )
                        break

                    # Calculate exponential backoff delay
                    delay = min(
                        self.retry_delay_base * (2 ** attempt),
                        self.retry_delay_max
                    )

                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            raise ImageGenerationError(
                f"Failed to generate image after {self.max_retries + 1} attempts"
            ) from last_error


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

class OutputManager:
    """
    Handles all file output operations for generated images and metadata.

    Responsibilities:
    - Create consistent output filenames that encode the label vector
    - Save images in appropriate format (JPEG for efficiency)
    - Save metadata JSON alongside images
    - Normalize image sizes
    - Ensure atomic writes to prevent partial files
    """

    def __init__(
        self,
        output_dir: Path,
        output_format: str = "JPEG",
        jpeg_quality: int = 95,
        target_size: tuple[int, int] = (512, 512)
    ):
        self.output_dir = output_dir
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        self.target_size = target_size

        self.images_dir = output_dir / "images"
        self.metadata_dir = output_dir / "metadata"

        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_extension(self) -> str:
        """Get file extension based on output format."""
        return ".jpg" if self.output_format.upper() == "JPEG" else ".png"

    def generate_output_filename(
        self,
        original_filename: str,
        labels: LabelVector
    ) -> str:
        """
        Create output filename encoding the original name and labels.

        Format: {original_stem}_{binary_string}.{ext}
        Example: 00001_1101011010.jpg
        """
        stem = Path(original_filename).stem
        binary_str = labels.to_binary_string()
        return f"{stem}_{binary_str}{self._get_extension()}"

    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target size with high-quality resampling."""
        if image.size != self.target_size:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    def save_image(self, image: Image.Image, filename: str) -> Path:
        """
        Save an image with atomic write semantics.

        Writes to a temporary file first, then renames to final location.
        This prevents partial files if the process is interrupted.
        """
        # Normalize size
        image = self._normalize_image(image)

        final_path = self.images_dir / filename
        temp_path = final_path.with_suffix(".tmp")

        try:
            save_kwargs = {"format": self.output_format}
            if self.output_format.upper() == "JPEG":
                save_kwargs["quality"] = self.jpeg_quality
                save_kwargs["optimize"] = True
            else:
                save_kwargs["optimize"] = True

            image.save(temp_path, **save_kwargs)
            temp_path.rename(final_path)
            return final_path
        except Exception:
            # Clean up temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_metadata(self, metadata: ImageMetadata, filename: str) -> Path:
        """Save metadata JSON file with atomic write."""
        json_filename = Path(filename).stem + ".json"
        final_path = self.metadata_dir / json_filename
        temp_path = final_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w") as f:
                f.write(metadata.to_json())
            temp_path.rename(final_path)
            return final_path
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_generated_image(
        self,
        image: Image.Image,
        original_filename: str,
        labels: LabelVector,
        prompt: str,
        seed: int,
        cost: float,
        blur_type: Optional[str] = None
    ) -> tuple[Path, Path]:
        """
        Complete save operation for a generated image.

        Creates both the image file and its metadata file.
        Returns paths to both files.
        """
        output_filename = self.generate_output_filename(original_filename, labels)

        # Create metadata
        metadata = ImageMetadata(
            original_filename=original_filename,
            label_vector=labels.values,
            label_names=labels.to_dict(),
            prompt_text=prompt,
            generation_seed=seed,
            timestamp=datetime.now().isoformat(),
            output_filename=output_filename,
            cost_usd=cost,
            blur_type=blur_type
        )

        # Save both files
        image_path = self.save_image(image, output_filename)
        metadata_path = self.save_metadata(metadata, output_filename)

        return image_path, metadata_path


# =============================================================================
# PROGRESS TRACKING AND RESUME SUPPORT
# =============================================================================

class ProgressTracker:
    """
    Tracks processing progress for safe resumption.

    Thread-safe implementation using file locking for concurrent writes.
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._processed: Optional[set[str]] = None
        self._lock = asyncio.Lock()

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_processed(self) -> set[str]:
        """Load the set of processed filenames from disk."""
        if self._processed is not None:
            return self._processed

        self._processed = set()

        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        self._processed.add(filename)

        return self._processed

    def is_processed(self, filename: str) -> bool:
        """Check if a filename has already been processed."""
        return filename in self._load_processed()

    async def mark_processed(self, filename: str) -> None:
        """
        Mark a filename as successfully processed.

        Uses async lock for thread safety in concurrent processing.
        """
        async with self._lock:
            self._load_processed().add(filename)

            # Append immediately for durability
            with open(self.log_file, "a") as f:
                f.write(f"{filename}\n")

    def get_processed_count(self) -> int:
        """Return the number of processed images."""
        return len(self._load_processed())

    def get_processed_set(self) -> set[str]:
        """Return a copy of the processed filenames set."""
        return self._load_processed().copy()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class SyntheticDatasetPipeline:
    """
    Orchestrates the complete synthetic dataset generation pipeline.

    Features:
    - Async concurrent processing
    - Resumable operation
    - Cost tracking
    - Comprehensive logging
    """

    def __init__(
        self,
        config: PipelineConfig,
        api: Optional[ImageGenerationAPI] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or self._setup_logger()

        # Initialize components
        self.dataset_streamer = HFDatasetStreamer(
            dataset_name=config.hf_dataset,
            split=config.hf_split
        )
        self.label_generator = LabelVectorGenerator(
            perfect_probability=config.perfect_vector_probability,
            near_perfect_probability=config.near_perfect_probability
        )
        self.prompt_builder = PromptBuilder(
            enable_variation=config.prompt_variation_enabled,
            dropout_probability=config.attribute_dropout_probability
        )
        self.output_manager = OutputManager(
            config.output_dir,
            output_format=config.output_format,
            jpeg_quality=config.jpeg_quality,
            target_size=config.output_image_size
        )
        self.progress_tracker = ProgressTracker(config.log_file)

        # Initialize blur augmenter for sharpness=0 images
        self.blur_augmenter = BlurAugmenter()

        # Initialize framing augmenter for framing=0 images
        self.framing_augmenter = FramingAugmenter()

        # Initialize API client with concurrency control
        if api is None:
            if config.api_key:
                api = NanoBananaProAPI(
                    api_key=config.api_key,
                    model_name=config.model_name,
                    timeout=config.api_timeout
                )
            else:
                self.logger.warning("No API key provided, using mock API")
                api = MockImageGenerationAPI()

        self.api_client = ResilientAPIClient(
            api=api,
            max_concurrent=config.max_concurrent_requests,
            max_retries=config.max_retries,
            retry_delay_base=config.retry_delay_base,
            retry_delay_max=config.retry_delay_max,
            rate_limit_delay=config.rate_limit_delay,
            logger=self.logger
        )

        # Statistics tracking
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "perfect_vectors": 0,
            "near_perfect_vectors": 0,  # Exactly one label is 0
            "dual_saves": 0,  # When sharpness=0, we save both sharp and blurred
            "blurred_images": 0,
            "bad_framing_images": 0,  # When framing=0, we apply programmatic bad framing
            "total_cost_usd": 0.0
        }
        self._stats_lock = asyncio.Lock()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the pipeline."""
        logger = logging.getLogger("SyntheticDatasetPipeline")
        logger.setLevel(self.config.log_level)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Console handler with formatting
        handler = logging.StreamHandler()
        handler.setLevel(self.config.log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # File handler for persistent logs
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    async def process_single_image(self, image_id: str, original_image: Image.Image) -> bool:
        """
        Process a single image through the complete pipeline.

        Args:
            image_id: Unique identifier for the image (e.g., "ffhq_00001.png")
            original_image: PIL Image streamed from HuggingFace

        Returns True if processing succeeded, False otherwise.
        """
        # Check if already processed
        if self.progress_tracker.is_processed(image_id):
            async with self._stats_lock:
                self.stats["skipped"] += 1
            return True

        try:
            # Generate deterministic seed from image_id for reproducibility
            base_seed = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)

            # Generate label vector
            labels = self.label_generator.generate()
            if labels.is_perfect():
                async with self._stats_lock:
                    self.stats["perfect_vectors"] += 1
            elif sum(labels.values) == 5:  # Exactly one 0 = near-perfect
                async with self._stats_lock:
                    self.stats["near_perfect_vectors"] += 1

            # Build prompt with seed for reproducibility
            # Note: Prompt builder skips sharpness and bad framing - always generates clean/well-framed
            prompt = self.prompt_builder.build_prompt(labels, seed=base_seed)

            # Generate new image (always sharp and well-framed - augmentations applied post-generation)
            generated_image = await self.api_client.generate_image(
                prompt=prompt,
                reference_image=original_image,
                seed=base_seed
            )

            # Get indices for augmented attributes
            sharpness_index = AttributeIndex.IMAGE_SHARPNESS_HIGH.value
            framing_index = AttributeIndex.FACE_PROPERLY_FRAMED.value

            original_sharpness = labels[sharpness_index]
            original_framing = labels[framing_index]

            # Get base name without extension for file naming
            image_stem = Path(image_id).stem

            # Create "perfect" labels (all augmentable attributes set to 1)
            perfect_labels = LabelVector(labels.values.copy())
            perfect_labels.values[sharpness_index] = 1
            perfect_labels.values[framing_index] = 1

            # Always save the "perfect" version (sharp + well-framed)
            perfect_path, _ = self.output_manager.save_generated_image(
                image=generated_image,
                original_filename=image_id,
                labels=perfect_labels,
                prompt=prompt,
                seed=base_seed,
                cost=self.config.cost_per_image,
                blur_type=None
            )

            saved_versions = [f"{perfect_path.name} ({perfect_labels.to_binary_string()})"]

            # Handle sharpness=0: apply blur and save blurred version
            if original_sharpness == 0:
                blurred_image, blur_type = self.blur_augmenter.apply_blur(
                    generated_image,
                    seed=base_seed
                )

                blur_labels = LabelVector(perfect_labels.values.copy())
                blur_labels.values[sharpness_index] = 0

                blur_path, _ = self.output_manager.save_generated_image(
                    image=blurred_image,
                    original_filename=f"{image_stem}_blur",
                    labels=blur_labels,
                    prompt=prompt,
                    seed=base_seed,
                    cost=0.0,
                    blur_type=blur_type
                )

                async with self._stats_lock:
                    self.stats["dual_saves"] += 1
                    self.stats["blurred_images"] += 1

                saved_versions.append(f"{blur_path.name} (blur:{blur_type})")

            # Handle framing=0: apply bad framing and save badly-framed version
            if original_framing == 0:
                bad_framed_image, framing_type = self.framing_augmenter.apply_bad_framing(
                    generated_image,
                    seed=base_seed + 1  # Different seed for framing variation
                )

                bad_frame_labels = LabelVector(perfect_labels.values.copy())
                bad_frame_labels.values[framing_index] = 0

                bad_frame_path, _ = self.output_manager.save_generated_image(
                    image=bad_framed_image,
                    original_filename=f"{image_stem}_badframe",
                    labels=bad_frame_labels,
                    prompt=prompt,
                    seed=base_seed,
                    cost=0.0,
                    blur_type=None  # Could add framing_type to metadata in future
                )

                async with self._stats_lock:
                    self.stats["bad_framing_images"] += 1

                saved_versions.append(f"{bad_frame_path.name} (frame:{framing_type})")

            self.logger.info(
                f"Processed {image_id} -> {len(saved_versions)} versions: {', '.join(saved_versions)}"
            )

            # Mark as processed (only the original image_id)
            await self.progress_tracker.mark_processed(image_id)

            async with self._stats_lock:
                self.stats["processed"] += 1
                self.stats["total_cost_usd"] += self.config.cost_per_image

            return True

        except ImageGenerationError as e:
            self.logger.error(f"Failed to process {image_id}: {e}")
            async with self._stats_lock:
                self.stats["failed"] += 1
            return False

        except Exception as e:
            self.logger.exception(f"Unexpected error processing {image_id}: {e}")
            async with self._stats_lock:
                self.stats["failed"] += 1
            return False

    async def run(self, limit: Optional[int] = None) -> dict:
        """
        Run the complete pipeline by streaming images from HuggingFace.

        Args:
            limit: Optional maximum number of images to process (for testing)

        Returns:
            Dictionary of processing statistics
        """
        self.logger.info(f"Streaming from HuggingFace: {self.config.hf_dataset}")
        self.logger.info(
            f"Already processed: {self.progress_tracker.get_processed_count()}"
        )
        self.logger.info(f"Max concurrent requests: {self.config.max_concurrent_requests}")

        if limit is not None:
            self.logger.info(f"Processing limited to {limit} images")

        start_time = time.time()

        # Stream images and create tasks with bounded concurrency
        # We use a semaphore to limit how many images we hold in memory
        pending_tasks: set = set()
        max_pending = self.config.max_concurrent_requests * 2  # Buffer for concurrency
        streamed_count = 0

        for image_id, image in self.dataset_streamer.stream_images():
            # Apply limit
            if limit is not None and streamed_count >= limit:
                break

            streamed_count += 1

            # Create task for this image
            task = asyncio.create_task(self.process_single_image(image_id, image))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            # If we have too many pending tasks, wait for some to complete
            while len(pending_tasks) >= max_pending:
                _, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = set(pending_tasks)

            # Progress reporting every 100 images streamed
            if streamed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = streamed_count / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"Progress: streamed {streamed_count}"
                    + (f"/{limit}" if limit else "") +
                    f" ({rate:.1f} img/s, "
                    f"{self.stats['processed']} new, "
                    f"{self.stats['skipped']} skipped, "
                    f"{self.stats['failed']} failed, "
                    f"${self.stats['total_cost_usd']:.2f} spent)"
                )

        # Wait for all remaining tasks to complete
        if pending_tasks:
            await asyncio.wait(pending_tasks)

        # Final statistics
        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = elapsed
        self.stats["total_images"] = streamed_count
        self.stats["images_per_second"] = (
            (self.stats["processed"] + self.stats["skipped"]) / elapsed
            if elapsed > 0 else 0
        )

        self.logger.info("=" * 60)
        self.logger.info("Pipeline complete!")
        self.logger.info(f"Total time: {elapsed:.1f}s ({self.stats['images_per_second']:.1f} img/s)")
        self.logger.info(f"Streamed: {streamed_count}")
        self.logger.info(f"Processed: {self.stats['processed']}")
        self.logger.info(f"Skipped (already done): {self.stats['skipped']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Perfect vectors (all 1s): {self.stats['perfect_vectors']}")
        self.logger.info(f"Near-perfect vectors (one 0): {self.stats['near_perfect_vectors']}")
        self.logger.info(f"Blurred images (sharpness=0): {self.stats['blurred_images']}")
        self.logger.info(f"Bad framing images (framing=0): {self.stats['bad_framing_images']}")
        total_output = self.stats['processed'] + self.stats['blurred_images'] + self.stats['bad_framing_images']
        self.logger.info(f"Total output images: {total_output}")
        self.logger.info(f"Total cost: ${self.stats['total_cost_usd']:.2f}")

        return self.stats


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic portrait dataset for ViT training (streams from HuggingFace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with mock API (no API calls), process 100 images
    python synthetic_dataset_generator.py -o ./output --mock --limit 100

    # Production with Gemini API
    python synthetic_dataset_generator.py -o ./output --api_key YOUR_KEY --limit 5000

    # High concurrency for faster processing
    python synthetic_dataset_generator.py -o ./output --api_key KEY --concurrency 10 --limit 5000

    # Use a different HuggingFace dataset
    python synthetic_dataset_generator.py -o ./output --hf_dataset rosasalberto/ffhq --mock --limit 100
        """
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Directory for output images and metadata"
    )

    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="marcosv/ffhq-dataset",
        help="HuggingFace dataset to stream from (default: marcosv/ffhq-dataset)"
    )

    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key for Gemini (or set GOOGLE_API_KEY env var)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-image",
        help="Gemini model name (default: gemini-2.5-flash-image)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock API for testing (no actual API calls)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent API requests (default: 5)"
    )

    parser.add_argument(
        "--perfect_prob",
        type=float,
        default=0.15,
        help="Probability of generating all-1s (perfect) label vectors (default: 0.15)"
    )

    parser.add_argument(
        "--near_perfect_prob",
        type=float,
        default=0.15,
        help="Probability of generating exactly-one-0 (near-perfect) label vectors (default: 0.15)"
    )

    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Disable prompt variation (use fixed phrasing)"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum API retry attempts (default: 3)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["JPEG", "PNG"],
        default="JPEG",
        help="Output image format (default: JPEG)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def async_main():
    """Async entry point for the script."""
    args = parse_args()

    # Build configuration from arguments
    config = PipelineConfig(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        output_dir=args.output_dir,
        log_file=args.output_dir / "processing_log.txt",
        api_key=args.api_key,
        model_name=args.model,
        max_concurrent_requests=args.concurrency,
        max_retries=args.max_retries,
        perfect_vector_probability=args.perfect_prob,
        near_perfect_probability=args.near_perfect_prob,
        prompt_variation_enabled=not args.no_variation,
        output_format=args.format,
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )

    # Select API implementation
    api = None
    if args.mock:
        api = MockImageGenerationAPI()
        print(f"Running in mock mode, streaming from: {args.hf_dataset}")
    elif not args.api_key:
        print("Warning: No API key provided. Use --api_key or set GOOGLE_API_KEY")
        print("Falling back to mock mode for testing.")
        api = MockImageGenerationAPI()

    # Create and run pipeline
    pipeline = SyntheticDatasetPipeline(config=config, api=api)

    try:
        stats = await pipeline.run(limit=args.limit)

        # Exit with error code if any failures occurred
        if stats["failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved.")
        print("Run again to resume from where you left off.")
        sys.exit(130)


def main():
    """Main entry point for the script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
