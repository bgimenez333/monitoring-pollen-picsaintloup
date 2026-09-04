
# Train ResNet152-vased model for multiclass classification with a 10-fold cross-validation;
# Generate predictions on the corresponding Test images;
# Using PyTorch

# Libraries

import os
import shutil
from pathlib import Path
import math
import random
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re

import cv2

from sklearn.metrics import f1_score, confusion_matrix
import albumentations as A

import gc
import torch
from torch.utils.data import Dataset
from torchvision.models import ResNet152_Weights
import torchvision.models as models 
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler 
from torch.utils.data import DataLoader



# Define constants
TARGET_SIZE = 224
BATCH_SIZE = 32
NB_EPOCHS = 80 
IMG_SIZE = 224
perc_covered=0.30

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))
from config1 import CLASSIFICATION_DATA



pdump = CLASSIFICATION_DATA / "training_outputs"

liste_crossvalidation = [ "cval1","cval2","cval3" , "cval4", "cval5", "cval6", "cval7", "cval8", "cval9", "cval10"]



MODEL_DIR = str(pdump)+"/models"
LOG_DIR = str(pdump)+"/logs"
PLOT_DIR = str(pdump)+"/plots"
DUMP_DIR = str(pdump)+"/dumps"



def check_memory(device, position="position"):
    print(position)
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(device=device) / (1024 * 1024):.2f} MB")
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / (1024 * 1024):.2f} MB")
    else:
        print("CUDA is not available.")



# Function to free memory
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def path_train_val_test(path_img, cval_id, perc_covered, TRAINING_ID): # run 10 cross validations
    path_img_0 = str(path_img) + "IMAGES_augmented_padded/"
    path_img_augm = str(path_img) + "IMAGES_augmDebris_padded_10folds/"

    dico_trainvaltest = {
    "cval1": ["portion1", "portion2", "portion3", "portion4", "portion5", "portion6", "portion7", "portion8", "portion9", "portion10"],
    "cval2": ["portion2", "portion3", "portion4", "portion5", "portion6", "portion7", "portion8", "portion9", "portion10", "portion1"], 
    "cval3": ["portion3", "portion4", "portion5", "portion6", "portion7", "portion8", "portion9", "portion10", "portion1", "portion2"], 
    "cval4": ["portion4", "portion5", "portion6", "portion7", "portion8", "portion9", "portion10", "portion1", "portion2", "portion3"], 
    "cval5": ["portion5", "portion6", "portion7", "portion8", "portion9", "portion10", "portion1", "portion2", "portion3", "portion4"],
    "cval6": ["portion6", "portion7", "portion8", "portion9", "portion10", "portion1", "portion2", "portion3", "portion4", "portion5"], 
    "cval7": ["portion7", "portion8", "portion9", "portion10", "portion1", "portion2", "portion3", "portion4", "portion5", "portion6"], 
    "cval8": ["portion8", "portion9", "portion10", "portion1", "portion2", "portion3", "portion4", "portion5", "portion6", "portion7"], 
    "cval9": ["portion9", "portion10", "portion1", "portion2", "portion3", "portion4", "portion5", "portion6", "portion7", "portion8"], 
    "cval10": ["portion10", "portion1", "portion2", "portion3", "portion4", "portion5", "portion6", "portion7", "portion8", "portion9"]
    }

    li_10_portions=["portion1", "portion2", "portion3", "portion4", "portion5", "portion6", "portion7", "portion8", "portion9", "portion10"]
    li_all_classes = os.listdir(path_img_0 + "portion1/")

    path_val = path_img_0 + dico_trainvaltest[cval_id][-2]
    path_test = path_img_0 + dico_trainvaltest[cval_id][-1]
    train_parent_directories = [path_img_0 + dico_trainvaltest[cval_id][i] for i in range(8)]

    # generating artificial directory with symbolic links for all class folders 
    path_smlink_allTRAINportion = path_img + f"01_dataset_smlink_trainDIR/SMLINk_trainDIR_{cval_id}/"
    # making sure all previous files are unlinked
    for class_p in Path(path_smlink_allTRAINportion).glob('*'):
        for vi_p in class_p.glob("*.jpg"):
            os.unlink(str(vi_p))
    shutil.rmtree(path_smlink_allTRAINportion,  ignore_errors=True)
        
    os.makedirs(path_smlink_allTRAINportion, exist_ok=True)

    for train_portion_i_dir in train_parent_directories:
        for source_class_p in Path(train_portion_i_dir).glob("*"):
            path_smlink_classe = path_smlink_allTRAINportion + source_class_p.name + "/"
            os.makedirs(path_smlink_classe, exist_ok=True)
            for source_vi_p in source_class_p.glob("*.jpg"):
                destin_class_p = path_smlink_classe + source_vi_p.name
                if not os.path.exists(destin_class_p):
                    os.symlink(source_vi_p, destin_class_p)

    # add a percentage of images manually augmented
    img_augm_selected=[]
    li_train_portions = list(dico_trainvaltest[cval_id][i] for i in range(8))
    for classe_i in li_all_classes: 
        li_img_cli = []
        for portion_i in li_train_portions:
            dir_path = os.path.join(path_img_augm, portion_i, classe_i)
            for img in os.listdir(dir_path):
                li_img_cli.append(os.path.join(portion_i, classe_i, img))
        li_img_cli=sorted(li_img_cli)
        nb_select = math.ceil(len(li_img_cli) * perc_covered)
        random.seed(45)
        li_selected = random.sample(li_img_cli, nb_select)
        img_augm_selected+=li_selected
        for vi_selected in li_selected : # vi_selected = portion_i/classe_i/filename.jpg
            source_vi_p=path_img_augm + vi_selected
            destin_class_p =  path_smlink_allTRAINportion + classe_i + "/" + vi_selected.split("/")[-1]
            if not os.path.exists(destin_class_p):
                os.symlink(source_vi_p, destin_class_p)
    with open(DUMP_DIR + f'/{TRAINING_ID}_augmfiles_{str(perc_covered).replace("0.", "")}perc_train_{cval_id}.pkl', 'wb') as f:
        pickle.dump(img_augm_selected, f)


    return path_smlink_allTRAINportion, path_val, path_test


def load_image_paths_and_labels(directory, color_mode='grayscale'):
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, filename)
                    image_paths.append(image_path)
                    labels.append(class_to_idx[class_name])

    return np.array(image_paths), np.array(labels), class_names



def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# albumentations -------------------------------------------------------------------------------------
def get_transforms(img_size, is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),

            # CONTRASTS:  ---
            A.OneOf([
                A.RandomGamma(p=1, gamma_limit=(10, 400)), 
                A.RandomBrightnessContrast(p=1, brightness_limit=(0.05,0.7), contrast_limit=(0.05,0.7), brightness_by_max=False), 
                A.Sharpen( p=1, alpha=(0.1, 0.99), lightness=(0.1, 0.8)),
            ], p=0.2),

            # BLUR ---
            A.OneOf([
                A.GaussianBlur(p=0.5, sigma_limit=(0.5, 10)), 
                A.OneOf([
                    A.Defocus(p=1, radius=(1,10), alias_blur=(1, 100)), 
                    A.GlassBlur(p=0.5, sigma=1, max_delta=4, iterations=2, mode='fast'),
                ], p=0.5),
            ], p=0.2), 

            # PIXELISATION  ---
            A.Downscale(p=0.2,scale_range=(0.25, 0.75)),
            A.AdditiveNoise(p=0.2, noise_type="gaussian",spatial_mode="per_pixel"), 

            # DEBRIS-like /  adding small black squares : ---
            A.OneOf([
                A.CoarseDropout(p=1, num_holes_range=(1, 15), hole_height_range=(10, int(img_size/2)),  hole_width_range=(10, int(img_size/2)), fill="random"),
                A.Erasing(scale=(0.1, 0.3),ratio=(0.6, 2.0),p=1.0), 
                A.RandomShadow(p=1, shadow_roi=(0.05, 0.05, 0.95, 0.95),num_shadows_limit=(1,3),shadow_dimension=5, shadow_intensity_range=(0.1, 0.9)),
            ], p=0.2),

            # ADDING NOISE, LIKE SMALL DEBRIS ON TOP ---
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.8, p=1), 
                A.RandomRain(rain_type="heavy", p=0.2), 
                A.PixelDropout(dropout_prob=0.2, per_channel=True, p=1.0),
            ], p=0.2),
            A.PlasmaShadow(p=0.2), 
        
            # CROPPING ---
            A.SomeOf([
                A.Compose([A.CropAndPad(percent=([0,-0.5,0,0.5]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),]), # only right size cut
                A.Compose([A.CropAndPad(percent=([0,0.5,0,-0.5]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),]), # only left side 
                A.Compose([A.CropAndPad(percent=([0.5, 0, -0.5, 0]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),]), # only bottom size cut
                A.Compose([A.CropAndPad(percent=([-0.5, 0, 0.5, 0]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),]), # only top cut
            ], p=0.2
            ),
        
            # ROTATIONS :  ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.D4(p=0.5),
            A.Rotate(limit=45, p=1.0),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize 
            A.ToTensorV2(), # Convert to PyTorch tensor format
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])


class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
 
        return image, torch.tensor(label, dtype=torch.long)

class CustomCallback:
    def __init__(self, path_txtfile, nb_epochs, model_save_path):
        self.path_txtfile = path_txtfile
        self.nb_epochs = nb_epochs
        self.model_save_path = model_save_path
        self.best_f1 = 0.0
        
    def on_epoch_end(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy, f1, model):
        output_line = f'Epoch {epoch + 1}/{self.nb_epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f} - f1: {f1:.4f}'
        print(output_line)
        
        with open(self.path_txtfile, 'a+') as file:
            file.write("\n" + output_line)
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            torch.save(model.state_dict(), f"{self.model_save_path}_checkpoint.pth")
            with open(self.path_txtfile, 'a+') as file:
                file.write(f"\n[Saved] Better F1 score: {f1:.4f}")

def load_initial_model(img_size, num_classes):
    base_model =  models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2) 
    num_ftrs = base_model.fc.in_features
    base_model.fc =     base_model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_ftrs, num_classes)
    )

    return base_model

def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return val_loss / len(val_loader), accuracy, f1, all_preds, all_targets





def training_model(cval_id, train_loader, val_loader, num_classes, nb_epochs, img_size, training_id, path_txtfile, class_names=None):
    
    free_memory()
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Load model
    model = load_initial_model(img_size, num_classes)
    check_memory(device=device, position="model just loaded")
    model = model.to(device)
    
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    

    model_save_path = f"{MODEL_DIR}/{training_id}_{cval_id}"
    callback = CustomCallback(path_txtfile, nb_epochs, model_save_path)
    
    for epoch in range(nb_epochs):
        start_time = time.time()
        check_memory(device=device, position=f"here is the start of epoch {epoch}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            epoch_time = time.time() - start_time
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}/{nb_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | duration since epoch started : {epoch_time:.4f}')

        print(f'Epoch: {epoch+1}/{nb_epochs} | All batches done | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | epoch duration : {epoch_time:.4f}')
        # Calculate training metrics
        train_accuracy = correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_accuracy, f1, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device, num_classes)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics and model
        callback.on_epoch_end(epoch, train_loss, train_accuracy, val_loss, val_accuracy, f1, model)
        
        # Plot confusion matrix at the end of each epoch
        if epoch in [50, 80 ]: 
            cm_path = f"{PLOT_DIR}/cm_{training_id}_{cval_id}_ep{epoch+1}.png"
            plot_confusion_matrix(all_targets, all_preds, class_names, cm_path)
        current_lr = optimizer.param_groups[0]['lr']
        print(f" \n Current learning rate: {current_lr}")

        
        # Free memory
        free_memory()

    
    
    # Final evaluation
    val_loss, val_accuracy, f1, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device, num_classes)
    print(f"Final Results - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), f"{model_save_path}_final.pth")
    
    return f"Training completed - Final Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}"




def plot_training_history(log_file, PLOT_DIR, training_id, cval_id ):
    history_path= f"{PLOT_DIR}/history_{training_id}_{cval_id}.png"
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []

    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line:
                match = re.search(r"Epoch (\d+)/\d+ - loss: (\d+\.\d+) - accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+) - f1: (\d+\.\d+)", line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    accuracy = float(match.group(3))
                    val_loss = float(match.group(4))
                    val_accuracy = float(match.group(5))
                    f1 = float(match.group(6))

                    epochs.append(epoch)
                    train_losses.append(loss)
                    train_accuracies.append(accuracy)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    f1_scores.append(f1)
    # Plotting Loss
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    # Plotting Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    # Plotting F1 Score (optional)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(history_path)
    plt.close()




def get_prediction_test_set(prediction_dir, cval_id, test_loader, img_size, num_classes, path_bestweights):
    model = load_initial_model(img_size, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(path_bestweights))
    model.eval()

    y_true = []
    y_pred_scores = []
    all_image_filenames = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred_scores.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred_scores = np.array(y_pred_scores)
    y_pred_classes = np.argmax(y_pred_scores, axis=1)

    accuracy = np.mean(y_pred_classes == y_true)

    print(f'Test Accuracy: {accuracy:.4f}')
    line = f'Test Accuracy: {accuracy:.4f}'


    
    with open(os.path.join(prediction_dir, f'{cval_id}_true_classes.pkl'), 'wb') as f:
        pickle.dump(y_true, f)

    with open(os.path.join(prediction_dir, f'{cval_id}_pred_scores.pkl'), 'wb') as f:
        pickle.dump(y_pred_scores, f)

    with open(os.path.join(prediction_dir, f'{cval_id}_pred_classes.pkl'), 'wb') as f:
        pickle.dump(y_pred_classes, f)

    return line



def main(cval_id, perc_covered):
    TRAINING_ID = f"resnet152_{datetime.now().strftime('%Y%m%d_%H%M')}"
    # Ensure directories exist
    for directory in [MODEL_DIR, LOG_DIR, PLOT_DIR, DUMP_DIR]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(DUMP_DIR, TRAINING_ID), exist_ok=True)
    
    # Set experiment parameters    
    img_size = TARGET_SIZE
    batch_size = BATCH_SIZE
    nb_epochs = NB_EPOCHS
    
    # Log file setup
    path_txtfile = os.path.join(LOG_DIR, f"training_log_{TRAINING_ID}_{cval_id}.txt")
    
    lines = f"\n{cval_id} starting time is  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n"
    lines+=f'constants are TARGET_SIZE = {TARGET_SIZE}; BATCH_SIZE={BATCH_SIZE}; NB_EPOCHS={NB_EPOCHS}; IMG_SIZE={IMG_SIZE} ; perc_covered={perc_covered}; TRAINING_ID : {TRAINING_ID}\n'

    print("Starting training:", cval_id)
    
    # Load data -------------------------------------------------------------------------------------------------------
    path_smlink_allTRAINportion, path_val, path_test = path_train_val_test(CLASSIFICATION_DATA, cval_id,  perc_covered, TRAINING_ID)
    train_paths, train_labels, class_names = load_image_paths_and_labels(path_smlink_allTRAINportion)
    val_paths, val_labels, class_namesval = load_image_paths_and_labels(path_val)
    test_paths, test_labels, class_namestest = load_image_paths_and_labels(path_test)
    num_classes = len(class_names)
    
    
    print(f"Number of training samples: {len(train_paths)}")
    print(f"Number of validation samples: {len(val_paths)}")
    print(f"Number of test samples: {len(test_paths)}")

    
    # Get transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create datasets
    train_dataset = ClassificationDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ClassificationDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = ClassificationDataset(test_paths, test_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Sample batch info
    sample_batch = next(iter(train_loader))
    image_shape = sample_batch[0].shape[1:]
    batch_size_train = sample_batch[0].shape[0]
    eltrain = f"Train set shapes: {image_shape} batch size: {batch_size_train}\n"
    
    sample_batch_val = next(iter(val_loader))
    image_shape = sample_batch_val[0].shape[1:]
    batch_size_val = sample_batch_val[0].shape[0]
    elval = f"Validation set shapes: {image_shape} batch size: {batch_size_val}\n"
    
    lines += eltrain
    lines += elval
    print(elval)
    print(eltrain)
    lines+=f' before launching training {datetime.now().strftime('%Y%m%d_%H%M')}'
    
    with open(path_txtfile, 'a+') as file:
        file.seek(0, 2)
        file.write("\n" + lines)
    
    # Train model
    print("just before launching function train_model")
    train_result = training_model(
        cval_id=cval_id, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_classes=num_classes, 
        nb_epochs=nb_epochs, 
        img_size=img_size, 
        training_id=TRAINING_ID, 
        path_txtfile=path_txtfile,
        class_names=class_names
    )
    
    lines += train_result
    lines+=f' after training done {datetime.now().strftime('%Y%m%d_%H%M')}'
    
    # Test model
    prediction_dir = os.path.join(DUMP_DIR, TRAINING_ID)
    os.makedirs(prediction_dir, exist_ok=True)
    
    path_bestweights = os.path.join(MODEL_DIR, f"{TRAINING_ID}_{cval_id}_checkpoint.pth")
    test_result = get_prediction_test_set(
        prediction_dir,
        cval_id, 
        test_loader, 
        img_size, 
        num_classes, 
        path_bestweights
    )

    # Save file names of test set
    path_test_save = [el.replace(str(CLASSIFICATION_DATA), "") for el in test_paths]    
    with open(os.path.join(prediction_dir, f'{cval_id}_images.pkl'), 'wb') as f: 
        pickle.dump(path_test_save, f)

    lines+= f"\n{cval_id} ending time is  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n"
    
    lines += test_result
    
    # Save final log
    with open(path_txtfile, 'a+') as file:
        file.seek(0, 2)
        file.write("\n" + lines)

    plot_training_history(path_txtfile, PLOT_DIR, TRAINING_ID, cval_id ) # saving plot with loss acc f1 history


    path_remove = CLASSIFICATION_DATA / f"01_dataset_smlink_trainDIR/SMLINk_trainDIR_{cval_id}/"
    for class_p in Path(path_remove).glob("*"):
        for vi_p in class_p.glob("*.jpg"):
            os.unlink(str(vi_p))
    shutil.rmtree(path_remove)
    
    
    print("Training and evaluation completed!")



for cval_id in liste_crossvalidation:
    main(cval_id, perc_covered)

