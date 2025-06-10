import torch
import torch.nn as nn
import timm
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")

class TeacherModel:
    def __init__(self, model_path, num_classes, device='cuda'):
        print(f"Loading teacher model from {model_path}")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model('resnetv2_50x1_bitm', pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class StudentModel:
    def __init__(self, model_path, num_classes, dataset_name, device='cuda'):
        print(f"Loading student model from {model_path}")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Select architecture based on dataset
        if dataset_name.lower() == 'fashion-mnist':
            self.model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=num_classes)
        else:  # food-101
            self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def load_class_labels(dataset_name):
    if dataset_name.lower() == 'fashion-mnist':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name.lower() == 'food-101':
        print("Loading Food-101 classes...")
        # Load Food-101 classes
        food_classes = []
        with open('data/food-101/meta/classes.txt', 'r') as f:
            food_classes = [line.strip() for line in f.readlines()]
        return food_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_random_image(dataset_name):
    print(f"Loading {dataset_name} dataset and selecting random image...")
    if dataset_name.lower() == 'fashion-mnist':
        # Load Fashion-MNIST test dataset
        test_dataset = datasets.FashionMNIST(
            root='data', 
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3)  # Convert to RGB
            ])
        )
    else:  # food-101
        # Load Food-101 test dataset
        test_dataset = datasets.Food101(
            root='data',
            split='test',
            download=True,
            transform=transforms.Resize((224, 224))
        )
    
    # Get random index
    idx = random.randint(0, len(test_dataset) - 1)
    print(f"Selected image index: {idx}")
    image, label = test_dataset[idx]
    return image, label

def predict_image(image, model, class_labels, model_type):
    # Ensure image is in RGB format
    if isinstance(image, torch.Tensor):
        input_tensor = image.unsqueeze(0)
        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = model.transform(image).unsqueeze(0)
    
    input_tensor = input_tensor.to(model.device)

    # Warm-up run to initialize CUDA and warm up the model
    with torch.no_grad():
        _ = model.model(input_tensor)
    if model.device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure inference time
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        output = model.model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item()
    
    print(f'{model_type} Prediction: {class_labels[pred_idx]} ({confidence*100:.1f}%) | Speed: {inference_time:.2f}ms')
    return pred_idx, confidence, inference_time

def save_result_image(image, true_label, class_labels, teacher_pred, student_pred, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    
    true_label_name = class_labels[true_label]
    teacher_label = class_labels[teacher_pred[0]]
    student_label = class_labels[student_pred[0]]
    
    title = f'True Label: {true_label_name}\n'
    title += f'Teacher: {teacher_label} ({teacher_pred[1]*100:.1f}%) [{teacher_pred[2]:.1f}ms] -> {"Correct" if teacher_pred[0] == true_label else "Incorrect"}\n'
    title += f'Student: {student_label} ({student_pred[1]*100:.1f}%) [{student_pred[2]:.1f}ms] -> {"Correct" if student_pred[0] == true_label else "Incorrect"}'
    
    plt.title(title)
    plt.tight_layout()
    
    save_path = f"results/prediction_{dataset_name}.png"
    plt.savefig(save_path)
    print(f"\nResult image saved to: {save_path}")
    plt.close()

def predict_for_dataset(dataset_name):
    print(f"\n--- Making predictions for {dataset_name} ---")
    
    num_classes = 10 if dataset_name.lower() == 'fashion-mnist' else 101
    
    if dataset_name.lower() == 'fashion-mnist':
        teacher_model_path = 'teacher models/bitm_resnet50x1_FMnist.pth'
        student_model_path = 'student_models_mobilenet/ranked_model_1.pth'
    else:
        teacher_model_path = 'teacher models/bitm_resnet50x1_F101.pth'
        student_model_path = 'student models/best_student_B0_f101.pth'
    
    teacher_model = TeacherModel(teacher_model_path, num_classes)
    student_model = StudentModel(student_model_path, num_classes, dataset_name)
    
    class_labels = load_class_labels(dataset_name)

    # Load the test dataset once
    if dataset_name.lower() == 'fashion-mnist':
        test_dataset = datasets.FashionMNIST(
            root='data', 
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3)
            ])
        )
    else:
        test_dataset = datasets.Food101(
            root='data',
            split='test',
            download=True,
            transform=transforms.Resize((224, 224))
        )

    for i in range(3):
        print(f"\n--- Sample {i+1}/3 for {dataset_name} ---")
        
        idx = random.randint(0, len(test_dataset) - 1)
        image, true_label = test_dataset[idx]
        
        print(f"True Label: {class_labels[true_label]}")
        
        teacher_pred = predict_image(image, teacher_model, class_labels, "Teacher")
        student_pred = predict_image(image, student_model, class_labels, "Student")
        
        # Save each result with a unique filename
        save_result_image(image, true_label, class_labels, teacher_pred, student_pred, f"{dataset_name}_sample_{i+1}")

def main():
    print("Starting prediction script...")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Set random seed for reproducibility
    random.seed(40)
    torch.manual_seed(40)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Predict for both datasets
    datasets = ['fashion-mnist', 'food-101']
    for dataset in datasets:
        predict_for_dataset(dataset)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 