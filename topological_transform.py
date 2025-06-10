import numpy as np
from ripser import ripser
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def convert_to_grayscale(image_tensor):
    """Converts a PyTorch image tensor to grayscale."""
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    grayscale_image = grayscale_transform(image_tensor)
    return grayscale_image

def image_to_point_cloud(img, threshold=0):
    # Extract (x, y) coordinates where intensity > threshold
    coords = np.argwhere(img > threshold)
    return coords.astype(np.float32)

def subsample(points, max_points=100):
    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        return points[indices]
    return points

def compute_ph(img):
    # Subsample + restrict to H1 + limit max distance
    pc = subsample(image_to_point_cloud(img))
    dgms = ripser(pc, maxdim=1)['dgms'] 
    return dgms

def persistence_image_from_dgm(image_shape, dgm, sigma=0.1, weight_fn=None, x_range=(0, 25), y_range=(0, 25)):
    if dgm is None or len(dgm) == 0:
        return np.zeros(image_shape)

    dgm = dgm[np.isfinite(dgm[:, 1])]
    if len(dgm) == 0:
        return np.zeros(image_shape)

    birth = dgm[:, 0]
    pers = dgm[:, 1] - dgm[:, 0]
    points = np.stack([birth, pers], axis=1)

    weights = pers if weight_fn is None else weight_fn(birth, pers)

    #x_range = (np.min(birth), np.max(birth))
    #y_range = (0, np.max(pers))

    H, W = image_shape
    x = np.linspace(*x_range, W)
    y = np.linspace(*y_range, H)
    X, Y = np.meshgrid(x, y)
    img = np.zeros_like(X)

    for (bx, py), w in zip(points, weights):
        gauss = np.exp(-((X - bx)**2 + (Y - py)**2) / (2 * sigma**2))
        img += w * gauss

    return img

def compute_persistence_image(tensor_img, sigma=0.1):
    """
    Complete pipeline: image -> PH -> persistence image -> stacked tensor
    Assumes input tensor_img has shape [B, C, H, W] or [C, H, W].
    Computes PH on channel 0, returns tensor of shape [B, 1, 32, 32] if batched,
    or [1, 32, 32] if single image.
    """
    if tensor_img.ndim == 4:
        batch = True
        B, C, H, W = tensor_img.shape
        device = tensor_img.device
        pers_imgs = []
        for i in range(B):
            img = tensor_img[i].detach().cpu()
            pers_imgs.append(compute_persistence_image(img, sigma).squeeze(0))  # shape [32, 32]
        pers_stack = torch.stack(pers_imgs).unsqueeze(1).to(device)  # [B, 1, 32, 32]
        return pers_stack

    # Single image case: [C, H, W]
    assert tensor_img.ndim == 3
    C, H, W = tensor_img.shape
    device = tensor_img.device

    if C == 1:
        grayscale = tensor_img[0]
    else:
        r, g, b = tensor_img[0], tensor_img[1], tensor_img[2]
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

    img_array = grayscale.detach().cpu().numpy()

    dgms = compute_ph(img_array)
    H1_dgm = dgms[1] if len(dgms) > 1 else np.array([])

    pers_img = persistence_image_from_dgm((32, 32), H1_dgm, sigma=sigma, x_range=(0, 25), y_range=(0, 25))
    pers_img = pers_img / np.max(pers_img) if np.max(pers_img) > 0 else pers_img
    pers_tensor = torch.tensor(pers_img, dtype=torch.float32).unsqueeze(0)  # [1, 32, 32]

    return pers_tensor.to(device)