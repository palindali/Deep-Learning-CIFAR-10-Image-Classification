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

from models import AliNet, VGG11, Wide_ResNet

now = datetime.now()

current_time = now.strftime("%B %d, %Y %H;%M;%S")

if __name__ == '__main__':
    # Params
    #   Model params
    in_channels = 3
    num_classes = 10

    #   Compute params
    batch_size = 128
    num_workers = 4
    
    #   Learning params
    num_epochs = 30
    learning_rate = 0.0001
    weight_decay = 0.001

    early_stop = 5

    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and transform data
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4), 
        # transforms.RandomHorizontalFlip(),
        # transforms.GaussianBlur(5, (0.1, 5)), 
        # transforms.RandomRotation(10),     #Rotates the image to a specified angel
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Training data
    train_data = datasets.ImageFolder('./data/train', transform=transform)
    # Test data
    test_data = datasets.ImageFolder('./data/test', transform=test_transform)
    
    # Save class labeling
    with open(f"./stuff/internal_class_labeling.json", "w") as write_file:
        json.dump(train_data.class_to_idx, write_file, indent=4)

    # # Split the data
    # data_len = len(train_data)

    # valid_len = int(data_len * 0)
    # train_len = data_len - valid_len

    # train_data, valid_data = torch.utils.data.random_split(train_data, [train_len, valid_len])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
        pin_memory=True
    )

    # valid_loader = torch.utils.data.DataLoader(
    #     valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
    #     pin_memory=True
    # )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory=True
    )

    # Load model
    # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    
    # # EfficientNetv2
    # model = torchvision.models.efficientnet_v2_s(weights=None)
    # model.classifier[1] = nn.Linear(1280, num_classes)
    # model = model.to(device)

    # # Resnet50
    # model = torchvision.models.resnet50(weights=None)
    # model.fc = nn.Linear(2048, num_classes)
    # model = model.to(device)
    
    
    # VGG11
    model = VGG11().to(device)

    # # Wide ResNet
    # model = Wide_ResNet(28, 10, 0.3, 10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    test_critereon = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay = weight_decay, 
        momentum = 0.9
    )  
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=learning_rate, 
    # )

    # Training the model
    total_step = len(train_loader)

    trn_accu = []
    val_accu = []
    tst_accu = []
    model_state_dicts = []

    # Advance experiment index
    with open(f"./stuff/model_checkpoints/ind.json") as file:
        exp_ind = json.load(file)
    exp_ind = str(int(exp_ind) + 1).zfill(3)
    with open(f"./stuff/model_checkpoints/ind.json", "w") as file:
        json.dump(exp_ind, file)

    model_dir = f"{exp_ind} - {current_time}"
    os.makedirs(f"./stuff/model_checkpoints/{model_dir}")

    best_test_accuracy = -1
    for epoch in range(num_epochs):
        
        # Training
        model.train()
        print(f"Epoch {epoch+1}:")
        for i, (images, labels) in enumerate(tqdm(train_loader, "Training", leave=False)):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass on the model to get outputs and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward propagate the gradient of the loss and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # model_state_dicts.append(model.state_dict())
        # torch.save(model.state_dict(), f"../model_checkpoints/Q1/vgg11_epoch{epoch+1}.pt")
        
        model.eval()
        # Train error
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0
            for images, labels in tqdm(train_loader, "Testing on Training Set", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = test_critereon(outputs, labels)
                running_loss += loss.item()
                del images, labels, outputs

            accuracy = 100 * correct / total
            loss = running_loss / total
            print(f"Accuracy of the network on the {total} training images: {accuracy} %") 
            print(f"Training Loss: {loss}")
            trn_accu.append(accuracy)
        
        # # Validation error
        # with torch.no_grad():
        #     correct = 0
        #     total = 0
        #     running_loss = 0
        #     for images, labels in tqdm(valid_loader, "Testing on Valid Set", leave=False):
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         outputs = model(images)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
                
        #         loss = test_critereon(outputs, labels)
        #         running_loss += loss.item()
        #         del images, labels, outputs

        #     accuracy = 100 * correct / total
        #     loss = running_loss / total
        #     print(f"Accuracy of the network on the {total} validation images: {accuracy} %") 
        #     print(f"Testing Loss: {loss}")
        #     val_accu.append(accuracy)
        
        # Test error
        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0
            for images, labels in tqdm(test_loader, "Testing on Test Set", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = test_critereon(outputs, labels)
                running_loss += loss.item()
                del images, labels, outputs

            test_accuracy = 100 * correct / total
            test_loss = running_loss / total
            print(f"Accuracy of the network on the {total} testing images: {test_accuracy} %") 
            print(f"Testing Loss: {test_loss}")
            tst_accu.append(test_accuracy)
        
        # Save output
        # torch.save(model.state_dict(), f"./stuff/model_checkpoints/{model_dir}/ep{epoch+1}_tst{tst_accu[-1]}_val;{val_accu[-1]}_trn;{trn_accu[-1]}.pt")
        # torch.save(model.state_dict(), f"./stuff/model_checkpoints/{model_dir}/ep{epoch+1}_tst{tst_accu[-1]}_trn;{trn_accu[-1]}.pt")

        # Check if this epoch's validation loss is the best so far
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            early_stopping_counter = 0
            # Save the model's state dictionary
            print("Best validation accuracy saving epoch ", epoch)
            torch.save(model.state_dict(), f"./stuff/model_checkpoints/{model_dir}/ep{epoch+1}_tst{tst_accu[-1]}_trn;{trn_accu[-1]}.pt")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stop:
                print('Early stopping')
                break
        print(early_stopping_counter)

    res_dict = {
        'Train Accuracy': trn_accu,
        # 'Validation Accuracy': val_accu,
        'Test Accuracy': tst_accu,
    }
    with open(f"./stuff/model_checkpoints/{model_dir}/accuracies.json", "w") as write_file:
        json.dump(res_dict, write_file, indent=4)
    
