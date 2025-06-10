import os
from PIL import Image
import torchvision.transforms.functional as TF

def get_first_image_paths_per_identity(root_dir: str, num_identities: int, split: str = "train"):
    """
    Returns a list of paths to the first image in each identity folder.
    """
    data_dir = os.path.join(
        root_dir, 
        "faces", 
        "faces", 
        f"{num_identities}_identities", 
        split
    )
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    image_paths = []

    for celeb in classes:
        celeb_dir = os.path.join(data_dir, celeb)
        image_files = sorted(
            f for f in os.listdir(celeb_dir) if f.lower().endswith(".jpg") or f.lower().endswith(".png")
        )
        if image_files:
            first_img_path = os.path.join(celeb_dir, image_files[0])
            image_paths.append(first_img_path)

    return image_paths
