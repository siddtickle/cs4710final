import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from flask import Flask, render_template, request
from PIL import Image
import torchvision.models as models
import pickle

# Load CIFAR-100 label names
label_names_path = './data/cifar-100-python/meta'
with open(label_names_path, 'rb') as f:
    label_names = pickle.load(f)['fine_label_names']

# Example label_names: ['apple', 'aquarium_fish', 'baby', ...]


app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 100  # CIFAR-100 has 100 classes
lr = 0.003
image_size = 32

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final fully connected layer
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Data loading and splitting for CIFAR-100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-100
])

# Assume you have the CIFAR-100 dataset saved in "./data" directory
full_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)

# Split the dataset into training, validation, and testing sets
train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Flask app code
@app.route('/')
def home():
    return render_template('index.html')

# Flask app code
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result='No selected file')

    # Read and preprocess the image
    image = Image.open(file)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    # Get the human-readable label for the predicted class
    predicted_label = label_names[prediction]

    # Display the result
    return render_template('index.html', result=f'Prediction: {predicted_label}')


if __name__ == '__main__':
    app.run(debug=True)
