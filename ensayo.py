import torch
from torchvision import transforms
from PIL import Image
import timm

img_path = 'F:\dataset\Anisognathus_somptuosus\o_idea.png'

import os

folder = r"F:\dataset\Anisognathus_somptuosus"
print(os.listdir(folder))  # Esto te dice qué imágenes hay realmente ahí


import os
print(os.path.exists(img_path))  # Esto debe imprimir: True


image = Image.open(img_path).convert('RGB')
image.show()


# Cargar el modelo preentrenado en iNaturalist
model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
model.eval()

# Imagen de entrada
# img_path = 'F:\dataset\Anisognathus_somptuosus\Anisognathus_somptuosus.jpg'

img_path = 'F:\dataset\Anisognathus_somptuosus\o_idea.png'
image = Image.open(img_path).convert('RGB')

# Preprocesamiento (mismo que usaron en el entrenamiento)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)

# Clasificación
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5 = torch.topk(probs, 5)

print("Top 5 predicciones:")
for idx, prob in zip(top5.indices, top5.values):
    print(f"Clase {idx.item()} con probabilidad {prob.item():.4f}")

