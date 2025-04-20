

#!/usr/bin/env python
#coding=utf-8

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import annotations
from __future__ import print_function, division


# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        # print(os.path.join(dirname, filename))





# In[3]:


# Import libraries
import json
import os
import pickle
import random
import time

import cv2

# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import PIL 
from tqdm import tqdm

from torchmetrics.classification import MulticlassAccuracy
import timm

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from skimage import io, transform
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import v2 as transforms_v2

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch dataset
from torchvision import datasets, models, transforms, utils
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar

import time
import os
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, roc_curve
import sklearn
import matplotlib.pyplot as plt
# import scikitplot as skplt

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)


# In[5]:


import cv2
import numpy as np
import PIL.Image

def segment_blood_vessels(image):
    """Extracts blood vessels using morphological operations and thresholding."""

    # Convert PIL Image to NumPy array
    image_np = np.array(image)

    # Extract green channel (best for vessel contrast)
    green_channel = image_np[:, :, 1]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)

    # Apply Morphological Top-Hat Transformation to enhance vessels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # Apply Otsu's thresholding
    _, binary_vessels = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Convert single-channel grayscale image to 3-channel RGB
    vessel_image = PIL.Image.fromarray(np.stack([binary_vessels] * 3, axis=-1))


    return vessel_image


# In[6]:


class OcularDiseaseRecognition(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, use_clahe=True):
        self.csv = csv_file
        self.img_dir = root_dir
        self.transform = transform
        self.use_clahe = use_clahe
        self.image_names = self.csv[:]['ImageName']
        self.labels = np.array(self.csv.drop(['ImageName'], axis=1))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        filename = self.image_names[index]
        name, ext = os.path.splitext(filename)

        jpg_path = os.path.join(self.img_dir, name + '.jpg')
        png_path = os.path.join(self.img_dir, name + '.png')

        try:
            if os.path.exists(jpg_path):
                image = PIL.Image.open(jpg_path).convert("RGB")
            elif os.path.exists(png_path):
                image = PIL.Image.open(png_path).convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found for index {index}: {jpg_path} or {png_path}")

            image_np = np.array(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            clahe_gray = clahe.apply(gray)
            grey_clahe = np.stack([clahe_gray] * 3, axis=-1)  # Convert to 3 channels

            r, g, b = cv2.split(image_np)
            r_clahe, g_clahe, b_clahe = clahe.apply(r), clahe.apply(g), clahe.apply(b)
            rgb_clahe = cv2.merge([r_clahe, g_clahe, b_clahe])
            green_clahe = cv2.merge([b, g_clahe, r])

            # **Blood Vessel Extraction**
            vessel_img = cv2.medianBlur(g_clahe, 5)
            vessel_img = cv2.adaptiveThreshold(vessel_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY_INV, 15, 5)
            kernel = np.ones((2, 2), np.uint8)
            vessel_img = cv2.morphologyEx(vessel_img, cv2.MORPH_OPEN, kernel, iterations=1)

            # **🔹 Convert vessel image from grayscale to 3-channel**
            vessel_img = np.stack([vessel_img] * 3, axis=-1)  # Convert to (H, W, 3)

            # Convert NumPy images to PIL
            rgb_clahe = PIL.Image.fromarray(rgb_clahe)
            green_clahe = PIL.Image.fromarray(green_clahe)
            grey_clahe = PIL.Image.fromarray(grey_clahe)
            vessel_img = PIL.Image.fromarray(vessel_img)  # Ensure it's 3-channel

            if self.transform:
                rgb_clahe = self.transform(rgb_clahe)
                green_clahe = self.transform(green_clahe)
                grey_clahe = self.transform(grey_clahe)
                vessel_img = self.transform(vessel_img)

            return {
                'image1': rgb_clahe,
                'image2': green_clahe,
                'image3': grey_clahe,
                'vessel': vessel_img,
                'labels': self.labels[index],
            }
        except FileNotFoundError as e:
            print(e)
            return None



# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import cv2

def show_5_specific_samples(dataset, indices):
    fig, axes = plt.subplots(len(indices), 4, figsize=(15, 15))  # 4 columns: original, RGB CLAHE, Green CLAHE, Blood Vessel

    for i, idx in enumerate(indices):
        sample = dataset[idx]  # Get the sample

        if sample is None:
            print(f"Skipping index {idx} due to missing image.")
            continue

        # Load original image
        filename = dataset.image_names[idx]
        name, _ = os.path.splitext(filename)
        jpg_path = os.path.join(dataset.img_dir, name + '.jpg')
        png_path = os.path.join(dataset.img_dir, name + '.png')

        if os.path.exists(jpg_path):
            file_path = jpg_path
        elif os.path.exists(png_path):
            file_path = png_path
        else:
            print(f"Warning: Image not found for index {idx}. Skipping.")
            continue

        original_img = np.array(PIL.Image.open(file_path).convert('RGB'))  # Convert to NumPy array

        # Convert to grayscale and apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        clahe_gray = clahe.apply(gray)
        grey_clahe = np.stack([clahe_gray] * 3, axis=-1)  # Convert single-channel to RGB

        # Apply CLAHE to each RGB channel separately
        r, g, b = cv2.split(original_img)
        r_clahe, g_clahe, b_clahe = clahe.apply(r), clahe.apply(g), clahe.apply(b)

        # RGB CLAHE (Enhance all channels)
        rgb_clahe = cv2.merge([r_clahe, g_clahe, b_clahe])

        # Green CLAHE (Enhance only green channel)
        green_clahe = cv2.merge([b, g_clahe, r])  # Keeping Red & Blue unchanged

        # **Blood Vessel Extraction using Morphological Operations**
        vessel_img = cv2.medianBlur(g_clahe, 5)  # Smooth noise
        vessel_img = cv2.adaptiveThreshold(vessel_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 15, 5)  # Enhance vessels

        # Morphological operations to refine vessels
        kernel = np.ones((2, 2), np.uint8)
        vessel_img = cv2.morphologyEx(vessel_img, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise

        # Display Original Image
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original (Idx {idx})")
        axes[i, 0].axis("off")

        # Display RGB CLAHE
        axes[i, 1].imshow(rgb_clahe)
        axes[i, 1].set_title("RGB CLAHE")
        axes[i, 1].axis("off")

        # Display Green CLAHE
        axes[i, 2].imshow(green_clahe)
        axes[i, 2].set_title("Green CLAHE")
        axes[i, 2].axis("off")

        # Display Blood Vessel Extraction
        axes[i, 3].imshow(vessel_img, cmap="gray")  # Show blood vessels in black/white
        axes[i, 3].set_title("Blood Vessels")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


# In[8]:


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, path="vessels_10_2.pth"):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            verbose (bool): If True, prints messages for each validation improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    import os

    def save_checkpoint(self, model):
        directory = os.path.dirname(self.path)

        # Check if directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(model.state_dict(), self.path)

        if self.verbose:
            print(f"Validation loss improved. Model saved to {self.path}.")


# In[9]:


train_dir = '/home/monetai/Desktop/Nhi/Dataset/train'
valid_dir = '/home/monetai/Desktop/Nhi/Dataset/valid'
test_dir = '/home/monetai/Desktop/Nhi/Dataset/test'
train_label_dir = '/home/monetai/Desktop/Nhi/Dataset/label_train.csv'
valid_label_dir = '/home/monetai/Desktop/Nhi/Dataset/label_test_on.csv'
test_label_dir = '/home/monetai/Desktop/Nhi/Dataset/label_test_off.csv'

df_train = pd.read_csv(train_label_dir)
df_val = pd.read_csv(valid_label_dir)
df_test = pd.read_csv(test_label_dir)

# Preprocess data
# Preprocess data
transform_train = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(degrees=15),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])


transform_valid = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

train_dataset  = OcularDiseaseRecognition(csv_file=df_train, 
                                      root_dir=train_dir, 
                                      transform=transform_train
                                     )

valid_dataset  = OcularDiseaseRecognition(csv_file=df_val, 
                                      root_dir=valid_dir, 
                                      transform=transform_valid
                                     )

test_dataset  = OcularDiseaseRecognition(csv_file=df_test, 
                                      root_dir=test_dir, 
                                      transform=transform_valid
                                     )


# In[10]:


# Day la cach de in anh, em doi index 0 thanh nhung so khac de lay nhung anh khac nhe
# sample = train_dataset[3]

import random
num0 = random.randint(0, 7000)  
num1 = random.randint(0, 7000)  
num2 = random.randint(0, 7000)  
num3 = random.randint(0, 7000)  
num4 = random.randint(0, 7000)  
indices = [num0, num1, num2, num3, num4]

# sample_index = random.randint(0, len(train_dataset) - 1)  # Pick a random image
# print(f"🔍 Checking image at index {sample_index}...")

# _ = train_dataset[sample_index]  # Fetch one sample to trigger the prints and displays



show_5_specific_samples(train_dataset, indices)



# In[11]:


print("Number of samples in training dataset :", len(train_dataset))
print("Number of samples in validating dataset :",len(valid_dataset))
print("Number of samples in testing dataset :",len(test_dataset))


# In[12]:


batch_size = 4
num_workers = 2
import cv2 as cv

cutmix = transforms_v2.CutMix(num_classes=7)
mixup = transforms_v2.MixUp(num_classes=7)
cutmix_or_mixup = transforms_v2.RandomChoice([cutmix, mixup])
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=num_workers)


# In[13]:


##for gpu##

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

print(device_name)


# In[14]:


class ODIRClassifier(nn.Module):
    def __init__(self, n_classes=7):
        super(ODIRClassifier, self).__init__()

        # Feature Extractors (CoaT-Small model)
        self.feature_extractor_1 = timm.create_model("coat_small", pretrained=True, num_classes=0)

        # Get feature size
        feature_dim = self.feature_extractor_1.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 4, 512),  # Increased to accommodate vessel features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, x1, x2, x3, vessel):
        f1 = self.feature_extractor_1(x1)
        f2 = self.feature_extractor_1(x2)
        f3 = self.feature_extractor_1(x3)
        f_vessel = self.feature_extractor_1(vessel)

        # Concatenate features from all three models and vessel model
        features = torch.cat([f1, f2, f3, f_vessel], dim=1)

        # Classification
        return self.classifier(features)


# In[15]:


def train(model, train_loader, val_loader, num_epochs, device, use_mixup, use_clahe, patience=20):
    model = model.to(device)
    best_loss = float('inf')

    early_stopping = EarlyStopping(patience=patience, path="/home/monetai/Desktop/Nhi/models/vessels_10_2.pth")
    use_tqdm = True  

    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        data_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", position=0, leave=True) if use_tqdm else train_loader

        for batch_idx, batch in enumerate(data_iter):
            x1, x2, x3, vessel, y = (
                batch['image1'].to(device),
                batch['image2'].to(device),
                batch['image3'].to(device),
                batch['vessel'].to(device),  # ✅ Correct key name
                batch['labels'].to(device),
            )

            optimizer.zero_grad()
            z = model(x1, x2, x3, vessel)  # ✅ Pass vessel image
            loss = criterion(z, y.float())
            loss.backward()
            optimizer.step()

            predicted = torch.where(z > 0.0, 1.0, 0.0)
            train_total += y.size(0) * z.shape[1]
            train_correct += predicted.eq(y).sum().item()
            train_loss += loss.item()

            if use_tqdm:
                data_iter.set_postfix({"Train Loss": train_loss/(batch_idx+1), "Train Accuracy": 100. * train_correct / train_total})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total
        print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f} Train Accuracy: {avg_train_acc:.2f}%")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            dataval_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", position=0, leave=True) if use_tqdm else val_loader
            for batch_idx, batch in enumerate(dataval_iter):
                x1, x2, x3, vessel, y = (
                    batch['image1'].to(device),
                    batch['image2'].to(device),
                    batch['image3'].to(device),
                    batch['vessel'].to(device),  # ✅ Include vessel here
                    batch['labels'].to(device),
                )

                z = model(x1, x2, x3, vessel)  # ✅ Pass vessel image

                loss = criterion(z, y.float())

                predicted = torch.where(z > 0.0, 1.0, 0.0)
                val_loss += loss.item()
                val_correct += predicted.eq(y).sum().item()
                val_total += y.size(0) * z.shape[1]

                if use_tqdm:
                    dataval_iter.set_postfix({"Val Loss": val_loss/(batch_idx+1), "Val Accuracy": 100. * val_correct / val_total})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {avg_val_acc:.2f}%")

        scheduler.step()

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break


# In[16]:


# ##################Use when running out of memory#################
# import gc

# # Delete objects
# del model
# del optimizer
# del criterion

# # Collect garbage
# gc.collect()
#################################################################


# In[ ]:


torch.cuda.empty_cache()
n_classes = 7

# Initialize the model
model = ODIRClassifier(n_classes=n_classes)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])

# Loss function and metrics
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

# Train the model
num_epochs = 1
train(model, train_loader, valid_loader, num_epochs, device, use_mixup=False, use_clahe=True)


# In[ ]:


def MaxProduct(gr, score, threshold=None):
    accuracy = 0.0
    sensi = 0.0
    speci = 0.0
    if threshold == None:
        fpr, tpr, thresholds = roc_curve(gr, score, pos_label=1)
        sensitivity = tpr
            # print(sensitivity)
        specificity = np.ones(fpr.shape) - fpr
        max_product = 0.0
        best_k = 0
        thresh = 0
        for k in range(len(fpr)):
            pred = np.where(score>=thresholds[k],1.0,0.0)
            acc = accuracy_score(gr.flatten(), pred.flatten())
            sen = sensitivity[k]
            spec = specificity[k]
            product = sen*spec 
            if product > max_product:
                max_product = product
                accuracy = acc
                sensi = sen
                speci = spec
                best_k = k
                thresh = thresholds[k]
                # kappa = quadratic_weighted_kappa(gr.flatten(), pred.flatten())
                # # pred[pred==0] = -1
                # mcc = matthews_corrcoef(gr.flatten(), pred.flatten())
                # f1 = f1_score(gr.flatten(), pred.flatten())
                # print('acc ', acc)
        # print(thresh)
    else:
        pred = np.where(score>=threshold,1.0,0.0)
        accuracy = accuracy_score(gr.flatten(), pred.flatten())
        sensi = sensitivity_score(gr.flatten(), pred.flatten())
        speci = specificity_score(gr.flatten(), pred.flatten())
        thresh = threshold
    #print('acc, sen, spec: ',accuracy, sensi, speci)
    return accuracy, sensi, speci, thresh

def test(epoch):
    print('Testing...')
    global best_acc_test
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            x1 = data['image1'].to(device)
            x2 = data['image2'].to(device)
            x3 = data['image3'].to(device)
            vessel = data['vessel'].to(device)  # Added vessel image
            labels = data['labels'].to(device)

            outputs = model(x1, x2, x3, vessel)  # Pass vessel into model
            loss = criterion(outputs, labels.float())

            test_loss += loss.item()
            predicted = torch.where(outputs > 0.0, 1.0, 0.0)
            total += labels.size(0) * outputs.shape[1]
            correct += predicted.eq(labels).sum().item()

            for i in range(outputs.shape[0]):
                out = outputs[i].cpu().numpy()
                pred.append(out)

    return test_loss, np.asarray(pred)

model = ODIRClassifier(n_classes=7)
model.load_state_dict(torch.load("/home/monetai/Desktop/Nhi/models/vessels_10_2.pth"))
# test_results = test(model.to(device), test_loader, device)
model.to(device)

auc_list = []

best_auc_test = 0.
id_best_test = 0
best_auc_val = 0.
id_best_val = 0

accuracies = []
sensitivities = []
specificities = []

ground_truth = np.array(df_test.drop(['ImageName'], axis=1))
test_loss,score = test(1)
auc = roc_auc_score(ground_truth, score, average=None)
m = auc.mean()
print('mean auc', m)
auc_list.append(auc.mean())
for i in range(score.shape[1]):
    print(i)
    fpr, tpr, thresholds = roc_curve(ground_truth[:, i], score[:, i], pos_label=1)
    accuracy, sensi, speci, thresh = MaxProduct(ground_truth[:, i], score[:, i], threshold = None)

    # Print rounded metrics
    print(f"Acc: {accuracy}, Sensitivity: {sensi}, Specificity: {speci}")

    # Accumulate values for averaging
    accuracies.append(accuracy)
    sensitivities.append(sensi)
    specificities.append(speci)

# Calculate averages
avg_acc = round(np.mean(accuracies), 2)
avg_sensi = round(np.mean(sensitivities), 2)
avg_speci = round(np.mean(specificities), 2)

# Print averaged metrics
print(f"Average Acc: {avg_acc}, Average Sensitivity: {avg_sensi}, Average Specificity: {avg_speci}")


# In[ ]:




