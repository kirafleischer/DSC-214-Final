from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from gudhi import CubicalComplex

def compute_ph_map(image_tensor):
    """
    Compute PH map from grayscale tensor using Gudhi CubicalComplex.
    Returns a (H, W) numpy array of lifetime values.
    """
    grayscale = image_tensor.mean(dim=0).numpy()
    h, w = grayscale.shape
    cubical_complex = CubicalComplex(top_dimensional_cells=grayscale)
    cubical_complex.persistence()

    diag = cubical_complex.persistence()
    ph_map = np.zeros((h, w), dtype=np.float32)

    for dim, (birth, death) in diag:
        if death == float('inf') or dim > 1:
            continue
        lifetime = death - birth
        x, y = np.random.randint(0, h), np.random.randint(0, w)
        ph_map[x, y] += lifetime

    ph_map /= (ph_map.max() + 1e-6)
    return ph_map

def generate_and_save_ph_maps(processed_dir="processed_data/faces/faces"):
    processed_dir = Path(processed_dir).expanduser()
    image_files = list(processed_dir.rglob("*.png"))

    for img_path in image_files:
        try:
            # Skip if PH map already exists
            ph_path = img_path.with_suffix(".ph.npy")
            if ph_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            tensor_img = TF.to_tensor(img)  # (3, H, W)
            ph_map = compute_ph_map(tensor_img)  # (H, W)

            # Save as .npy
            ph_path = img_path.with_suffix(".ph.npy")
            np.save(ph_path, ph_map)
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    generate_and_save_ph_maps("processed_data/faces/faces")
