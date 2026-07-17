# import torch
# from torchvision import transforms, models
# from PIL import Image
# from pathlib import Path
# import tkinter as tk
# from tkinter import filedialog
# import matplotlib.pyplot as plt

# def pick_file_dialog():
#     """
#     Open a file dialog window so the user can click/choose an image.
#     Returns a Path or None if canceled.
#     """
#     root = tk.Tk()
#     root.withdraw()  # don't show the full Tk window

#     filetypes = [
#         ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
#         ("All files", "*.*")
#     ]

#     filename = filedialog.askopenfilename(title="Select an image to classify", filetypes=filetypes)

#     if not filename:
#         return None
    
#     return Path(filename)

# def load_model(model_path, device):
#     """
#     Load the trained ResNet18 model and return (model, class_names).
#     """
#     checkpoint = torch.load(model_path, map_location=device)
#     class_names = checkpoint["class_names"]

#     num_classes = len(class_names)
#     model = models.resnet18(weights=None)
#     in_features = model.fc.in_features
#     model.fc = torch.nn.Linear(in_features, num_classes)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()

#     return model, class_names


# def preprocess_image(image_path, device):
#     """
#     Open an image from disk and apply the same preprocessing
#     used during training/validation.
#     """
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])

#     img = Image.open(image_path).convert("RGB")
#     tensor = transform(img).unsqueeze(0).to(device)  # shape [1, 3, 224, 224]
#     return img, tensor  # return both PIL image (for display) and tensor (for model)


# def predict_one(model, class_names, image_tensor):
#     """
#     Run one forward pass and return (predicted_label, confidence, probs_vector).
#     """
#     with torch.no_grad():
#         outputs = model(image_tensor)             # shape [1, num_classes]
#         probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # shape [num_classes]

#     conf, idx = torch.max(probs, dim=0)
#     predicted_label = class_names[idx.item()]
#     confidence = conf.item()

#     return predicted_label, confidence, probs.cpu().numpy()



# def show_result(pil_img, species_name, confidence):
#     """
#     Display the image and overlay the prediction in the title.
#     """
#     plt.figure(figsize=(5,5))
#     plt.imshow(pil_img)
#     plt.axis("off")
#     plt.title(f"{species_name} ({confidence*100:.2f}% confidence)")
#     plt.tight_layout()
#     plt.show()


# def main():
#     # -------- UPDATE THIS PATH to your .pth checkpoint --------
#     model_path = r"C:/Users/Daniel/OneDrive - correounivalle.edu.co/Univalle/Proyecto condensador/Avance personal/Python/bird_species_resnet18.pth"

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[INFO] Using device: {device}")

#     # 1. Ask user to choose an image file interactively
#     image_path = pick_file_dialog()
#     if image_path is None:
#         print("[INFO] No file selected. Exiting.")
#         return

#     print(f"[INFO] You selected: {image_path}")

#     # 2. Load model
#     model, class_names = load_model(model_path, device)

#     # 3. Preprocess chosen image
#     pil_img, img_tensor = preprocess_image(image_path, device)

#     # 4. Predict
#     species_name, confidence, _ = predict_one(model, class_names, img_tensor)

#     # 5. Report
#     print(f"\nPrediction:")
#     print(f"  Species: {species_name}")
#     print(f"  Confidence: {confidence*100:.2f}%")

#     # 6. Show image + prediction visually
#     show_result(pil_img, species_name, confidence)


# if __name__ == "__main__":
#     main()
