import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from transformation import *
from topological_transform import *

print("Environment Initialized")

# Default dataset name and identity counts for 'faces'
DEFAULT_DATASET = "faces"
IDENTITY_COUNTS = [4, 8, 16, 32, 64, 128]

def process_dataset(
    dataset: str = DEFAULT_DATASET,
    root_dir: str = "data",
    processed_dir: str = "topological_processed_data"
):
    """
    Processes images under `root_dir` into `processed_dir`:
      - CelebA_HQ_facial_identity_dataset:
          ~/data/CelebA_HQ_facial_identity_dataset/{train,valid,test}/<label>/images
      - faces:
          ~/data/faces/faces/{n}_identities/{train,valid,test}/<label>/images

    For each image:
      1. four_random_crops (tensor-only)
      2. rotate (random ±15° or 180° if split=='test')
      3. compute_persistence_image
      4. foveation
      5. logpolar_manual

    Outputs mirror input structure inside `processed_dir`.
    """
    root = Path(root_dir).expanduser()
    dest = Path(processed_dir).expanduser()

    # Determine subdirectories to process
    if dataset == "faces":
        base = root / dataset / dataset
        sub_dirs = [base / f"{n}_identities" for n in IDENTITY_COUNTS]
    else:
        base = root / dataset
        sub_dirs = [base]

    splits = ["train", "valid", "test"]

    for sub in sub_dirs:
        print(f"Now processing sub directory {sub}.")
        for split in splits:
            input_split = sub / split
            if not input_split.exists():
                continue

            # Build corresponding output directory
            rel = sub.relative_to(root)
            output_split = dest / rel / split
            output_split.mkdir(parents=True, exist_ok=True)

            for label_dir in input_split.iterdir():
                if not label_dir.is_dir():
                    continue
                out_label = output_split / label_dir.name
                out_label.mkdir(exist_ok=True)

                for img_file in label_dir.iterdir():
                    if not img_file.is_file():
                        continue
                    try:
                        img = Image.open(img_file).convert("RGB")
                    except Exception:
                        continue

                    # Convert to tensor and get four random crops
                    tensor_img = TF.to_tensor(img)
                    crops = four_random_crops(tensor_img)

                    for i, tensor in enumerate(crops):
                        # rotate: 180° for test, random ±15° otherwise
                        tensor = rotate(tensor, inverse=(split == "test"))
                        # persistence image
                        pers_tensor = compute_persistence_image(tensor)
                        # save persistence tensor
                        pt_filename = f"{img_file.stem}_proc{i}.pt"
                        torch.save(pers_tensor, out_label / pt_filename)
                        # foveation
                        tensor = foveation(tensor)
                        # log-polar transform
                        C, H, W = tensor.shape
                        tensor = logpolar_manual(tensor, (H, W), (H, W))

                        # Save
                        out_img = TF.to_pil_image(tensor.clamp(0, 1))
                        filename = f"{img_file.stem}_proc{i}.png"
                        out_img.save(out_label / filename)


if __name__ == "__main__":
    Path(processed_dir:="topological_processed_data").mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        process_dataset(sys.argv[1])
    else:
        process_dataset()
