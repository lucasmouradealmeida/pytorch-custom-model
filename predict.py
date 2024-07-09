import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Definir transformações e DataLoader
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Transformações para o treinamento e teste
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Redimensiona para 224x224
        transforms.ToTensor(),  # Converte para tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normaliza
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Carrega os datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Definir modelo CNN personalizado
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Inicializa o modelo
model = CustomModel(num_classes=len(train_dataset.classes))
model.load_state_dict(torch.load("custom_model.pth"))
model.eval()


# Função para pré-processar a imagem
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = test_transforms(image).unsqueeze(0)  # Adiciona dimensão do batch
    return image


# Função para prever a classe da imagem
# Função para prever a classe da imagem com probabilidade
def predict_image_class(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        class_index = predicted.item()
        class_name = train_dataset.classes[class_index]
        probability = probabilities[
            0, class_index
        ].item()  # Probabilidade da classe predita
    return class_name, probability


# Caminho da imagem de entrada
image_path = "car.webp"

# Faz a predição
predicted_class, porcentagem = predict_image_class(model, image_path)
print(
    f"Objeto identificado na imagem: {predicted_class},"
    + f"Probabilidade: {porcentagem:.2f}"
)
