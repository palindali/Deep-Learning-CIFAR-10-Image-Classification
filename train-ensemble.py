# Deep Learning Data Challenge 1 - Ali ElSaid

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import os
import json
from torchensemble import VotingClassifier

from models import AliNet, VGG11

now = datetime.now()

current_time = now.strftime("%B %d, %Y %H;%M;%S")

if __name__ == '__main__':
    # Params
    #   Model params
    in_channels = 3
    num_classes = 10

    #   Compute params
    batch_size = 32
    num_workers = 4
    
    #   Learning params
    num_epochs = 15
    learning_rate = 0.001
    weight_decay = 0.001

    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and transform data
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),     #Rotates the image to a specified angel
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Training data
    # Test data
    train_data = datasets.ImageFolder('./data/train', transform=transform)
    test_data = datasets.ImageFolder('./data/test', transform=test_transform)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory=True
    )

    # VGG11
    # base_classifier = VGG11().to(device)

    # Define the ensemble
    ensemble = GradientBoostingClassifier(
        estimator=VGG11,               # estimator is your pytorch model
        n_estimators=5,                        # number of base estimators
        cuda=True
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()
    ensemble.set_criterion(criterion)

    # Set the optimizer
    ensemble.set_optimizer(
        "Adam",                                 # type of parameter optimizer
        lr=learning_rate,                       # learning rate of parameter optimizer
        weight_decay=weight_decay,              # weight decay of parameter optimizer
    )

    # Set the learning rate scheduler
    ensemble.set_scheduler(
        "CosineAnnealingLR",                    # type of learning rate scheduler
        T_max=num_epochs,                           # additional arguments on the scheduler
    )

    # ADVANCE EXPERIMENT INDEX
    with open(f"./stuff/model_checkpoints/ind.json") as file:
        exp_ind = json.load(file)
    exp_ind = str(int(exp_ind) + 1).zfill(3)
    with open(f"./stuff/model_checkpoints/ind.json", "w") as file:
        json.dump(exp_ind, file)

    # CREATE MODEL DIRECTORY
    model_dir = f"{exp_ind} - {current_time}"
    os.makedirs(f"./stuff/model_checkpoints/{model_dir}")

    # Train the ensemble
    ensemble.fit(
        train_loader,
        epochs=num_epochs,                          # number of training epochs
        save_model=True,
        # save_dir=f"./stuff/model_checkpoints/{model_dir}/ensemble.pt",
    )

    # Evaluate the ensemble
    train_acc = ensemble.evaluate(train_loader)
    print(f"Ensemble Train Accuracy: {train_acc}")
    test_acc = ensemble.evaluate(test_loader)
    print(f"Ensemble Test Accuracy: {test_acc}")

    res_dict = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
    }
    with open(f"./stuff/model_checkpoints/{model_dir}/accuracies.json", "w") as write_file:
        json.dump(res_dict, write_file, indent=4)

    # # SAVE OUTPUT
    # torch.save(ensemble.state_dict(), f"./stuff/model_checkpoints/{model_dir}/ensemble_tst{test_acc}_trn;{train_acc}.pt")
