"""
bird_imagefolder_dataset.py
Alternative dataset loader using torchvision.datasets.ImageFolder for bird classification.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_imagefolder_datasets(
    root_train: str,
    root_val: str,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """
    Create PyTorch DataLoaders for training and validation using ImageFolder.
    Each subdirectory in the root paths is treated as a class label.
    """

    # Define transformation pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(root=root_train, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=root_val, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, train_loader, val_loader


# # ---------------------------------------------------------
# # Demo Section
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     print("[DEMO] Running ImageFolder test...")

#     # Define dataset roots (replace with actual paths)
#     root_train = "sample_dataset/train"
#     root_val = "sample_dataset/val"

#     try:
#         train_dataset, val_dataset, train_loader, val_loader = create_imagefolder_datasets(
#             root_train=root_train,
#             root_val=root_val,
#             batch_size=4,
#             num_workers=0,
#         )

#         print(f"Train classes: {train_dataset.class_to_idx}")
#         print(f"Validation classes: {val_dataset.class_to_idx}")
#         print(f"Total training images: {len(train_dataset)}")
#         print(f"Total validation images: {len(val_dataset)}")

#         # Fetch one batch for inspection
#         images, labels = next(iter(train_loader))
#         print(f"Batch shape: {images.shape}")
#         print(f"Labels: {labels.tolist()}")
#         print(f"Class names: {[train_dataset.classes[i] for i in labels.tolist()]}")

#     except Exception as e:
#         print(f"[ERROR] {e}")