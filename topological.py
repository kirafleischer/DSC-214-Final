import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from topological_load import *
from transformation import *
from resnet18_model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from Datasets import *
from ripser import ripser
from persim import plot_diagrams, wasserstein, bottleneck
from PIL import Image
import cv2
from gudhi import CubicalComplex

# Ensure output directory exists
os.makedirs("topological", exist_ok=True)

results = []  # Global list to store experiment results

def load_grayscale_image(path, resize=(128, 128)):
    img = Image.open(path).convert('L')  # convert to grayscale
    img = img.resize(resize)
    return np.array(img)

def invert_image(img):
    return np.rot90(img, k=2)  # 180-degree rotation

def to_log_polar(img):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    max_radius = np.linalg.norm(center)
    log_polar_img = cv2.logPolar(img.astype(np.float32), center, 
                                 M=max_radius / np.log(max_radius), 
                                 flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    return log_polar_img

def image_to_point_cloud(img, threshold=50):
    # Extract (x, y) coordinates where intensity > threshold
    coords = np.argwhere(img > threshold)
    return coords.astype(np.float32)

def subsample(points, max_points=1000):
    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        return points[indices]
    return points

def compute_ph(img):
    # Subsample + restrict to H1 + limit max distance
    pc = subsample(image_to_point_cloud(img))
    dgms = ripser(pc, maxdim=1)['dgms'] #, thresh=50.0
    return dgms

def persistence_image_from_dgm(dgm, resolution=(128, 128), sigma=0.1, weight_fn=None):
    """
    Convert a persistence diagram (birth, death) into a persistence image.
    """
    if dgm is None or len(dgm) == 0:
        return np.zeros(resolution)

    # Remove points at infinity
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(dgm) == 0:
        return np.zeros(resolution)

    birth = dgm[:, 0]
    pers = dgm[:, 1] - dgm[:, 0]
    points = np.stack([birth, pers], axis=1)

    # Apply weighting
    weights = pers if weight_fn is None else weight_fn(birth, pers)

    # Define grid
    x_range = (np.min(birth), np.max(birth))
    y_range = (0, np.max(pers))  # persistence is always ≥ 0
    x = np.linspace(*x_range, resolution[0])
    y = np.linspace(*y_range, resolution[1])
    X, Y = np.meshgrid(x, y)
    img = np.zeros_like(X)

    for (bx, py), w in zip(points, weights):
        gauss = np.exp(-((X - bx)**2 + (Y - py)**2) / (2 * sigma**2))
        img += w * gauss

    return img

def generate_input_tensor(img_path, resize_shape=(128, 128), resolution=(128, 128), sigma=0.1):
    """
    Complete pipeline: image -> PH -> persistence image -> stacked tensor
    """
    # Load and process image
    img = load_grayscale_image(img_path, resize_shape)
    dgms = compute_ph(logpolar_img)
    
    # Get H1 diagram (you can also experiment with H0)
    H1_dgm = dgms[1] if len(dgms) > 1 else np.array([])

    # Convert PH diagram to image
    pers_img = persistence_image_from_dgm(H1_dgm, resolution=resolution, sigma=sigma)

    # Resize image to match persistence image resolution
    resized_img = resize(img, resolution, anti_aliasing=True)

    # Stack as channels: shape = [2, H, W]
    tensor = np.stack([resized_img, pers_img], axis=0)
    return tensor

def compute_ph_map(img):
    """
    Compute a 2D persistence map from an image using GUDHI's CubicalComplex.
    Since we can't localize features spatially, we create a uniform heatmap
    with aggregate persistence values.
    """
    flipped = 255 - img.astype(np.float32)
    cubical = CubicalComplex(top_dimensional_cells=flipped)
    cubical.compute_persistence()

    h1 = cubical.persistence_intervals_in_dimension(1)
    h, w = flipped.shape
    ph_map = np.zeros((h, w), dtype=np.float32)

    if len(h1) > 0:
        # Total lifetime of H1 features
        total_persistence = np.sum(h1[:, 1] - h1[:, 0])
        ph_map += total_persistence / (h * w)

    return ph_map

def plot_ph_map(ph_map, title, out_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(ph_map, cmap='inferno')
    plt.colorbar(label="Persistence")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def compare_images(img1, img2, title1, title2, idx, experiment_type):
    dgm1 = compute_ph(img1)
    dgm2 = compute_ph(img2)

    # Create a 3x2 grid plot (image, diagram, barcode)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # --- Row 1: Original images ---
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title(f"{title1} Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title(f"{title2} Image")
    axes[0, 1].axis('off')

    # --- Row 2: Persistence diagrams ---
    plot_diagrams(dgm1, ax=axes[1, 0], title=f"{title1} Diagram")
    plot_diagrams(dgm2, ax=axes[1, 1], title=f"{title2} Diagram")

    # --- Row 3: Persistence barcodes ---
    def plot_combined_barcodes(ax, dgm, title):
        h0, h1 = dgm[0], dgm[1]
        offset_h1 = len(h0) + 2  # Offset H1 bars above H0
        for i, (birth, death) in enumerate(h0):
            if death == float('inf'):
                death = max([d for b, d in h0 if d != float('inf')], default=1.0) + 1
            ax.hlines(y=i, xmin=birth, xmax=death, color='blue', lw=2, label='H0' if i == 0 else "")
        for i, (birth, death) in enumerate(h1):
            if death == float('inf'):
                death = max([d for b, d in h1 if d != float('inf')], default=1.0) + 1
            ax.hlines(y=offset_h1 + i, xmin=birth, xmax=death, color='red', lw=2, label='H1' if i == 0 else "")

        ax.set_title(title)
        #ax.set_xlabel("Filtration Value")
        #ax.set_ylabel("Feature Index")
        ax.legend(loc='upper right')
        ax.grid(True)
    
    
    plot_combined_barcodes(axes[2, 0], dgm1, f"{title1} Barcode")
    plot_combined_barcodes(axes[2, 1], dgm2, f"{title2} Barcode")
    
    axes[2, 0].set_title(f"{title1} Barcode")
    #axes[2, 0].set_xlabel("Filtration Value")
    #axes[2, 0].set_ylabel("Feature Index")

    axes[2, 1].set_title(f"{title2} Barcode")
    #axes[2, 1].set_xlabel("Filtration Value")
    #axes[2, 1].set_ylabel("Feature Index")

    # Save combined figure
    plt.tight_layout()
    out_path = os.path.join("topological", f"{title1}_vs_{title2}_{idx}_combined.png")
    plt.savefig(out_path)
    plt.close()

    # Compute distances
    bottleneck_dist = bottleneck(dgm1[1], dgm2[1])
    wasserstein_dist = wasserstein(dgm1[1], dgm2[1])

    # Store results
    results.append({
        "index": idx,
        "experiment": experiment_type,
        "image_1": title1,
        "image_2": title2,
        "bottleneck_distance_H1": bottleneck_dist,
        "wasserstein_distance_H1": wasserstein_dist
    })

# === Experiments ===
def experiments(img_paths):
    for idx, img_path in enumerate(img_paths):
        img = load_grayscale_image(img_path)
        img_inv = invert_image(img)
        logpolar_img = to_log_polar(img)
        logpolar_inv = to_log_polar(img_inv)
        # 1. Normal vs. Inverted
        compare_images(img, img_inv, "Normal", "Inverted", idx, experiment_type="Normal_vs_Inverted")
        # 2. Log-polar vs. Inverted Log-polar
        compare_images(logpolar_img, logpolar_inv, "Log-polar", "Inverted_Log-polar", idx, experiment_type="Logpolar_vs_Inverted")
        # 3. Normal vs. Log-polar
        compare_images(img, logpolar_img, "Normal", "Log-polar", idx, experiment_type="Normal_vs_Logpolar")
        # 4. Inverted vs. Log-polar Inverted
        compare_images(img_inv, logpolar_inv, "Inverted", "Inverted_Log-polar", idx, experiment_type="Inverted_vs_LogpolarInverted")
        # 5. Save and view PH maps
        '''
        variants = {
            "Normal": img,
            "Inverted": img_inv,
            "Logpolar": logpolar_img,
            "Inverted_Logpolar": logpolar_inv,
        }
        for name, variant_img in variants.items():
            try:
                ph_map = compute_ph_map(variant_img)
                out_path = os.path.join("topological", f"{name}_{idx}_phmap.png")
                plot_ph_map(ph_map, title=f"{name} PH Map", out_path=out_path)
            except Exception as e:
                print(f"PH map failed for {name} at index {idx}: {e}")
        '''
def main():
    img_paths = get_first_image_paths_per_identity(
        root_dir="data", 
        num_identities=4, 
        split="train"
    )
    experiments(img_paths)

    # Save distances to CSV
    df = pd.DataFrame(results)
    df.to_csv("topological/distances.csv", index=False)
    print("Saved topological distances to topological/distances.csv")
    
if __name__ == "__main__":
    main()