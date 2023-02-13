# Generate predictions

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchensemble.utils import io
from torchensemble import VotingClassifier, BaggingClassifier


from models import AliNet, VGG11
import params, utils

# !!!!!!!!!CHANGE HERE!!!!!!!!!!!!
model_name = '29'

if __name__ == '__main__':
    # Params
    batch_size = 128
    num_workers = 0

    # GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and transform data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # dataset = datasets.ImageFolder('./data/test_bkp', transform=transform)
    dataset = utils.ImagesDataset(
        './data/test_labels.csv',
        './data/test_bkp/test',
        transform=test_transform)

    plus = 4
    for i in range(len(dataset)):
        image, label, image_name = dataset[i+plus]
        
        # print(i, image, label)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i+plus))
        ax.axis('off')
        # show_landmarks(**sample)
        plt.imshow(image.permute(1, 2, 0))
        
        if i + plus == 3 + plus:
            plt.show()
            break
        # break
    
    # Dataloader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        pin_memory=True
    )

    # Models
    # Ensemble
    ensemble = BaggingClassifier(
        estimator=VGG11,               # estimator is your pytorch model
        n_estimators=7,                        # number of base estimators
        cuda=True
    )
    
    io.load(ensemble, f"./stuff/model_checkpoints/best")  # reload
        
    predictions = []
    with torch.no_grad():
        for image_id in tqdm(range(10000), "Predicting on Test"):
            img_path = './data/test_bkp/test/'+str(image_id)+'.jpg'
            img = Image.open(img_path)
            image = test_transform(img)
            image = image.to("cuda").unsqueeze(0)
            outputs = ensemble.predict(image)

            _, predicted = torch.max(outputs.data, 1)
            predictions += predicted.tolist()
        
    # Make preds dataframe
    df = pd.DataFrame(list(zip(
            list(range(len(predictions))), predictions
            )), columns=['id','label'])

    # df.to_csv(f'./stuff/submissions/submission_{submission}_notmapped.csv', index=False)

    # Load internal label representation
    with open(f"./stuff/internal_class_labeling.json") as file:
        int_labeling = json.load(file)
    int_labeling = {v: k for k, v in int_labeling.items()}

    
    # Load competition label representation
    mapping = {
        0: "deer",
        1: "horse",
        2: "car",
        3: "truck",
        4: "small mammal",
        5: "flower",
        6: "tree",
        7: "aquatic mammal",
        8: "fish",
        9: "ship",
    }
    mapping = {v: k for k, v in mapping.items()}

    # Map df values from internal label rep to competition one
    df['label'] = df['label'].map(int_labeling)
    df['label'] = df['label'].map(mapping)
    
    # Advance submission index
    with open(f"./stuff/submissions/ind.json") as file:
        sub_ind = json.load(file)
    sub_ind = str(int(sub_ind) + 1).zfill(3)
    with open(f"./stuff/submissions/ind.json", "w") as file:
        json.dump(sub_ind, file)

    with open(f"./stuff/model_checkpoints/ind.json") as file:
        exp_ind = json.load(file)

    df.to_csv(f'./stuff/submissions/predictions_{sub_ind}(exp_{model_name}).csv', index=False)
