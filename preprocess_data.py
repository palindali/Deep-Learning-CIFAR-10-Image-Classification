# Preprocess training data

import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm


name = 'test'

labels = pd.read_csv(f"./data/{name}_labels.csv")
labels.describe()

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
labels['actual_label'] = labels['label'].map(mapping)


label_list = np.sort(labels['label'].unique())

for label in label_list:
    images = (labels.loc[labels['label'] == label,'id'].astype('string') + '.jpg').tolist()
    
    new_dir = f'./data/{name}/' + mapping[label]
    os.makedirs(new_dir, exist_ok=True)
    for image in tqdm(images, str(label)):
        old_path = f'./data/{name}_bkp/{name}/' + image
        new_path = new_dir + '/' + image
        shutil.copy(old_path, new_path)
