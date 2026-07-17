"""
bird_dataset.py
Custom PyTorch Dataset for bird image classification.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
from PIL import Image
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    """
    BirdDataset is a custom PyTorch Dataset class designed for supervised image classification of birds.
    It provides full control and transparency over how image samples are loaded, labeled, and transformed.
    This version mirrors torchvision.datasets.ImageFolder but is explicitly documented for reproducibility.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)

        if not self.root_dir.exists():
            raise RuntimeError(f"[BirdDataset] root_dir does not exist: {self.root_dir}")

        # Discover all class folders
        class_dirs = [d for d in sorted(self.root_dir.iterdir()) if d.is_dir()]
        if len(class_dirs) == 0:
            raise RuntimeError(f"[BirdDataset] No class subfolders found in {self.root_dir}")

        # Assign numeric labels to each class
        self.class_to_idx: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
        self.idx_to_class: Dict[int, str] = {i: d.name for i, d in enumerate(class_dirs)}

        # Gather all image paths and their labels
        samples: List[Tuple[Path, int]] = []
        for class_dir in class_dirs:
            label_idx = self.class_to_idx[class_dir.name]
            for file in sorted(class_dir.iterdir()):
                if file.is_file() and file.suffix.lower() in self.allowed_extensions:
                    samples.append((file, label_idx))

        if len(samples) == 0:
            raise RuntimeError(f"[BirdDataset] No valid images found under {self.root_dir}")

        self.samples = samples

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Return a single sample (image, label) given its index.
        The transform is applied dynamically during each access.
        """
        img_path, label_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label_idx


# # ---------------------------------------------------------
# # Demo Section
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     print("[DEMO] Running BirdDataset test...")

#     # Path to your dataset folder (adjust to your actual path)
#     dataset_path = Path("sample_dataset")

#     # Define a simple transform
#     demo_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     try:
#         dataset = BirdDataset(root_dir=dataset_path, transform=demo_transform)
#         print(f"Total samples: {len(dataset)}")
#         print(f"Classes found: {dataset.class_to_idx}")

#         # Create a DataLoader for inspection
#         loader = DataLoader(dataset, batch_size=4, shuffle=True)

#         images, labels = next(iter(loader))
#         print(f"Batch shape: {images.shape}")
#         print(f"Labels: {labels.tolist()}")
#         print(f"Class names: {[dataset.idx_to_class[i] for i in labels.tolist()]}")

#     except Exception as e:
#         print(f"[ERROR] {e}")