import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from torchvision import transforms

# --- IMPORT YOUR FUNCTIONS ---
from bird_detector import (
    load_detector,
    preprocess_image_for_detector,
    run_detector_on_image,
)
from sl_methods import extract_all_features_torch
from sl_dataframe_construction import detect_and_crop_birds   # the function we wrote


# -------------------------------------------------------
# CONFIG: Adjust to your dataset
# -------------------------------------------------------
TEST_IMAGES_FOLDER = Path("F:/Univalle/01_TG/nuevo_dataset_split/train/anisognathus_somptuosus")

# Set how many images to test
N_IMAGES = 3

RESIZE_TO = (224, 224)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Device:", device)

# -------------------------------------------------------
# Load detector ONCE
# -------------------------------------------------------
detector_model, coco_classes = load_detector(device)

# Transform to tensor + resize
to_tensor_and_resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_TO, antialias=True),
])

rows = []
sample_id = 0

# -------------------------------------------------------
# Loop over a few images ONLY
# -------------------------------------------------------
for img_path in list(TEST_IMAGES_FOLDER.glob("*.jpg"))[:N_IMAGES]:
    print(f"\n[INFO] Processing {img_path.name}")

    # Load PIL image
    pil_img = Image.open(img_path).convert("RGB")

    # Detect and crop (max 2 birds)
    crops = detect_and_crop_birds(
        pil_img,
        detector_model,
        coco_classes,
        device=device,
        max_birds=2,
        conf_threshold=0.7,
    )

    print(f"  Detected crops: {len(crops)}")

    for crop_pil, bird_idx in crops:
        sample_name = f"{img_path.stem}_{bird_idx}"

        # Convert to tensor
        crop_t = to_tensor_and_resize(crop_pil)

        # Extract SL features
        feats = extract_all_features_torch(crop_t).tolist()
        print(f"  → features length = {len(feats)}")

        # Build row
        row = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "species": TEST_IMAGES_FOLDER.name,
            "split": "train",
        }
        for i, f in enumerate(feats):
            row[f"f{i}"] = f

        rows.append(row)
        sample_id += 1


# -------------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------------
df_test = pd.DataFrame(rows)
print("\n[RESULT] Test DataFrame head:\n")
print(df_test.head())

print("\n[INFO] Columns count:", df_test.shape[1])
print("[INFO] Total samples:", df_test.shape[0])

# Save small test CSV
df_test.to_csv("sl_test_output.csv", index=False)
print("\n[INFO] Saved sl_test_output.csv")
