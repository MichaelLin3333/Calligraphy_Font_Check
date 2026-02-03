import os
import io
import json
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.ExifTags import TAGS
import random
import argparse
from train_model import ForkedResNet50, font_to_label

def load_char_map(char_map_path='char_map.json'):
    """Load character mapping from JSON file"""
    with open(char_map_path, 'r', encoding='utf-8') as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char

def load_model(checkpoint_path, num_chars, device):
    """Load model from checkpoint"""
    model = ForkedResNet50(num_chars)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    return model.to(device)

def infer_image(model, image_path, device, transform, idx_to_char, augment=False, augment_count=5):
    """Perform inference on a single image"""
    try:
        # Preprocess (handle EXIF orientation, ensure RGB), then convert to grayscale-processed RGB
        image = preprocess_image(image_path)

        # If not augmenting, single forward
        if not augment:
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                char_logits, style_logits = model(image_tensor)

            char_prob = torch.softmax(char_logits, dim=1)[0]
            style_prob = torch.softmax(style_logits, dim=1)[0]

        else:
            # Generate light augmentations and average probabilities
            probs_char = None
            probs_style = None
            aug_images = generate_augmentations(image, augment_count)
            with torch.no_grad():
                for aug in aug_images:
                    t = transform(aug).unsqueeze(0).to(device)
                    char_logits, style_logits = model(t)
                    c = torch.softmax(char_logits, dim=1)[0]
                    s = torch.softmax(style_logits, dim=1)[0]
                    if probs_char is None:
                        probs_char = c.cpu()
                        probs_style = s.cpu()
                    else:
                        probs_char += c.cpu()
                        probs_style += s.cpu()

            probs_char = probs_char / len(aug_images)
            probs_style = probs_style / len(aug_images)

            char_prob = probs_char
            style_prob = probs_style

        # Pick best predictions from averaged probs
        char_pred = int(torch.argmax(char_prob).item())
        style_pred = int(torch.argmax(style_prob).item())

        char_confidence = float(char_prob[char_pred].item())
        style_confidence = float(style_prob[style_pred].item())

        char_name = idx_to_char.get(char_pred, "Unknown")
        style_mapping = {0: 'cs', 1: 'ks', 2: 'ls', 3: 'xs', 4: 'zs'}
        style_name = style_mapping.get(style_pred, "Unknown")

        return char_name, char_confidence, style_name, style_confidence
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, 0, None, 0


def preprocess_image(image_path: str) -> Image.Image:
    """Open image, fix EXIF orientation and ensure RGB (similar to main.py preprocess)."""
    with open(image_path, 'rb') as f:
        data = f.read()

    try:
        image = Image.open(io.BytesIO(data))

        # Handle EXIF orientation if present
        try:
            exif = image._getexif()
            if exif:
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    if decoded == 'Orientation':
                        if value == 3:
                            image = image.rotate(180, expand=True)
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                        elif value == 8:
                            image = image.rotate(90, expand=True)
                        break
        except Exception:
            pass

        # Convert to grayscale and apply equalization/denoise/enhancement
        try:
            gray = image.convert('L')
            # Equalize histogram to reduce color/lighting bias
            eq = ImageOps.equalize(gray)
            # Median filter to reduce small noise
            denoised = eq.filter(ImageFilter.MedianFilter(size=3))

            # Slight contrast and sharpness boost
            enhancer = ImageEnhance.Contrast(denoised)
            denoised = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Sharpness(denoised)
            denoised = enhancer.enhance(1.05)

            # Convert back to RGB since model expects 3 channels
            image = denoised.convert('RGB')
        except Exception:
            # Fallback to RGB conversion
            if image.mode != 'RGB':
                image = image.convert('RGB')

        return image

    except Exception as e:
        raise RuntimeError(f"Cannot preprocess image {image_path}: {e}")

def generate_augmentations(image: Image.Image, count: int = 5):
    """Generate light, label-preserving augmentations. Returns `count` PIL RGB images."""
    aug_images = []
    max_shift = 4
    for i in range(count):
        # include original as first augmentation for stability
        if i == 0:
            aug_images.append(image.copy())
            continue

        img = image.copy()

        # small rotation
        angle = random.uniform(-8.0, 8.0)
        img = img.rotate(angle, resample=Image.BILINEAR)

        # brightness / contrast / sharpness jitter
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.95, 1.05))

        # occasional mild gaussian blur to simulate soft images
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0)))

        # slight translation by padding and cropping
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        if dx != 0 or dy != 0:
            padded = ImageOps.expand(img, border=max_shift, fill=(255, 255, 255))
            w, h = img.size
            left = max_shift + dx
            top = max_shift + dy
            img = padded.crop((left, top, left + w, top + h))

        # occasional minor scaling with center-crop or pad back
        if random.random() < 0.2:
            scale = random.uniform(0.95, 1.05)
            w, h = img.size
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((nw, nh), resample=Image.BILINEAR)
            if nw >= w and nh >= h:
                left = (nw - w) // 2
                top = (nh - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:
                img = ImageOps.pad(img, (w, h), color=(255, 255, 255))

        aug_images.append(img)

    return aug_images

def main():
    parser = argparse.ArgumentParser(description='Test calligraphy recognition model on images or a directory of images.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_50.pth', help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to an image file or a directory of images')
    parser.add_argument('--char-map', type=str, default='char_map.json', help='Path to character map JSON')
    parser.add_argument('--augment', action='store_true', help='Perform light augmentations and average predictions')
    parser.add_argument('--augment-count', type=int, default=5, help='Number of augmentations to average when --augment used')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load character mapping
    char_to_idx, idx_to_char = load_char_map(args.char_map)
    num_chars = len(char_to_idx)
    print(f"Loaded {num_chars} character classes")

    # Load model
    model = load_model(args.checkpoint, num_chars, device)
    if model is None:
        return

    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Test inference on single image
    print("\nStarting inference...")
    print('-' * 80)
    
    # Normalize path to handle both forward and backward slashes
    input_path = os.path.normpath(args.input)

    # Supported image extensions
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')

    if os.path.isdir(input_path):
        files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path)
                        if f.lower().endswith(exts)])

        if not files:
            print(f"No supported image files found in directory: {input_path}")
        else:
            for img in files:
                pred_char, char_conf, pred_style, style_conf = infer_image(
                    model, img, device, transform, idx_to_char, augment=args.augment, augment_count=args.augment_count
                )
                if pred_char is not None:
                    print(f"\nImage: {img}")
                    print(f"Predicted Character: {pred_char} (Confidence: {char_conf:.4f})")
                    print(f"Predicted Style: {pred_style} (Confidence: {style_conf:.4f})")
    elif os.path.isfile(input_path):
        pred_char, char_conf, pred_style, style_conf = infer_image(
            model, input_path, device, transform, idx_to_char, augment=args.augment, augment_count=args.augment_count
        )

        if pred_char is not None:
            print(f"\nImage: {input_path}")
            print(f"Predicted Character: {pred_char} (Confidence: {char_conf:.4f})")
            print(f"Predicted Style: {pred_style} (Confidence: {style_conf:.4f})")
    else:
        print(f"Error: Input path not found: {input_path}")

    print("-" * 80)

if __name__ == '__main__':
    main()
