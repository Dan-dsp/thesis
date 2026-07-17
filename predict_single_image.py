import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


def predict_image(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    # Define the same preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    # Load model
    num_classes = len(class_names)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    print(f"Predicted class: {class_names[top_idx]}")
    print(f"Confidence: {top_prob.item()*100:.2f}%")

    return class_names[top_idx], top_prob.item()


if __name__ == "__main__":
    model_path = r"C:/Users/Daniel/OneDrive - correounivalle.edu.co/Univalle/Proyecto condensador/Avance personal/Python/bird_species_resnet18.pth"
    # image_path = r"F:/Univalle/01_TG/nuevo_dataset_split/test/chlorophanes_spiza/ML640987207.jpg"
    image_path = r"G:/Usuarios/Daniel/Descargas/large.jpeg"

    predict_image(model_path, image_path)
