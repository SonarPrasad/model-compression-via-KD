import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Helper function to evaluate model
def evaluate_model(model, data_loader, dataset_name, model_name):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # To store probabilities
    total_time = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {model_name} on {dataset_name}"):
            images, labels = images.to(device), labels.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())  # Get probabilities

    accuracy = 100 * correct / total
    avg_inference_time = total_time / total

    # Concatenate all probabilities
    all_probs = np.concatenate(all_probs)

    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  # Use probabilities

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name} on {dataset_name}")
    plt.savefig(f"results/confusion_matrix_{dataset_name.lower()}_{model_name.lower()}.png")
    plt.close()

    return accuracy, avg_inference_time, precision, recall, f1, auc_roc

def main():
    with open("results/evaluation_results.txt", "w") as result_file, open("results/inference_time.txt", "w") as time_file:
        # Fashion MNIST
        transform_fmnist = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_loader_fmnist = DataLoader(
            datasets.FashionMNIST(root='data', train=False, transform=transform_fmnist, download=True),
            batch_size=100, shuffle=False)

        # Food101
        transform_f101 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_loader_f101 = DataLoader(
            datasets.Food101(root='data', split='test', transform=transform_f101, download=True),
            batch_size=101, shuffle=False)

        # Fashion MNIST Teacher
        fmnist_teacher = timm.create_model('resnetv2_50x1_bitm', pretrained=False, num_classes=10)
        fmnist_teacher.load_state_dict(torch.load(os.path.join('teacher models', 'bitm_resnet50x1_FMnist.pth')))
        fmnist_teacher.to(device)
        acc, time_taken, precision, recall, f1, auc_roc = evaluate_model(fmnist_teacher, test_loader_fmnist, "FashionMNIST", "Teacher")
        result_file.write(f"FashionMNIST Teacher Accuracy: {acc:.2f}%\n")
        time_file.write(f"FashionMNIST Teacher Inference Time: {time_taken:.6f} sec/image\n")
        result_file.write(f"FashionMNIST Teacher Precision: {precision:.4f}\n")
        result_file.write(f"FashionMNIST Teacher Recall: {recall:.4f}\n")
        result_file.write(f"FashionMNIST Teacher F1-score: {f1:.4f}\n")
        result_file.write(f"FashionMNIST Teacher AUC-ROC: {auc_roc:.4f}\n")

        # Fashion MNIST Student
        fmnist_student = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=10)
        fmnist_student.load_state_dict(torch.load(os.path.join('student_models_mobilenet', 'ranked_model_1.pth')))
        fmnist_student.to(device)
        acc, time_taken, precision, recall, f1, auc_roc = evaluate_model(fmnist_student, test_loader_fmnist, "FashionMNIST", "Student")
        result_file.write(f"FashionMNIST Student Accuracy: {acc:.2f}%\n")
        time_file.write(f"FashionMNIST Student Inference Time: {time_taken:.6f} sec/image\n")
        result_file.write(f"FashionMNIST Student Precision: {precision:.4f}\n")
        result_file.write(f"FashionMNIST Student Recall: {recall:.4f}\n")
        result_file.write(f"FashionMNIST Student F1-score: {f1:.4f}\n")
        result_file.write(f"FashionMNIST Student AUC-ROC: {auc_roc:.4f}\n")

        # Food101 Teacher
        f101_teacher = timm.create_model('resnetv2_50x1_bitm', pretrained=False, num_classes=101)
        f101_teacher.load_state_dict(torch.load(os.path.join('teacher models', 'bitm_resnet50x1_F101.pth')))
        f101_teacher.to(device)
        acc, time_taken, precision, recall, f1, auc_roc = evaluate_model(f101_teacher, test_loader_f101, "Food101", "Teacher")
        result_file.write(f"Food101 Teacher Accuracy: {acc:.2f}%\n")
        time_file.write(f"Food101 Teacher Inference Time: {time_taken:.6f} sec/image\n")
        result_file.write(f"Food101 Teacher Precision: {precision:.4f}\n")
        result_file.write(f"Food101 Teacher Recall: {recall:.4f}\n")
        result_file.write(f"Food101 Teacher F1-score: {f1:.4f}\n")
        result_file.write(f"Food101 Teacher AUC-ROC: {auc_roc:.4f}\n")

        # Food101 Student
        f101_student = timm.create_model('efficientnet_b0', pretrained=False, num_classes=101)
        f101_student.load_state_dict(torch.load(os.path.join('student models', 'best_student_B0_f101.pth')))
        f101_student.to(device)
        acc, time_taken, precision, recall, f1, auc_roc = evaluate_model(f101_student, test_loader_f101, "Food101", "Student")
        result_file.write(f"Food101 Student Accuracy: {acc:.2f}%\n")
        time_file.write(f"Food101 Student Inference Time: {time_taken:.6f} sec/image\n")
        result_file.write(f"Food101 Student Precision: {precision:.4f}\n")
        result_file.write(f"Food101 Student Recall: {recall:.4f}\n")
        result_file.write(f"Food101 Student F1-score: {f1:.4f}\n")
        result_file.write(f"Food101 Student AUC-ROC: {auc_roc:.4f}\n")

if __name__ == "__main__":
    main()