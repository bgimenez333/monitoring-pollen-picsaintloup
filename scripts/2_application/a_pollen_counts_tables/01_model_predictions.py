# This script generates predictions for each images from the monitoring image dataset, using the 10 cross-validated models
# The generated predictions used in the associated publication are also available in the ZENODO repository "Monitoring-PSL" folder "2.PREDICTIONS_10models/"


# Libraries
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))
from config1 import CLASSIFICATION_DATA, MONITORING_DATA



import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models 
from torchvision.models import ResNet152_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
import gc



p_models = CLASSIFICATION_DATA / "2.MODEL_TRAINING/"
DATA_DIR = MONITORING_DATA / "IMAGES_padded/" # "1.IMAGES_application" need to be padded, using "03_padding.ipynb" from "scripts/1_classification/" from this Github repository
PRED_DIR = MONITORING_DATA / "predictions/" 


li_img_foldergroups = ["ANNU2019","ANNU2020","ANNU2021","ANNU2022","ANNU2023"] # images of each slides from a given year are grouped in the same folder (annual pollen traps in distinct sites)


class_names = ['Acacia', 'Acer', 'Alnus', 'Amarantaceae', 'Artemisia', 'Betulaceae', 'Brassicaceae', 'Buxus', 'Carduus', 'Caryophyllaceae',
               'Cichorioideae', 'Cupressaceae', 'Cyperaceae', 'Echium', 'Ericaceae', 'Fagus', 'FraxinusExcelsior', 'FraxinusOrnus', 'Gallium',
               'IndetBlurry', 'IndetCovered', 'Juglans', 'Lamiaceae', 'Liliaceae', 'Lycopodium', 'Moraceae', 'Morphotype1_smallgrains', 
               'Morphotype2_largegrains', 'Myrtaceae', 'NonPollen', 'Olea', 'Other', 'Phillyrea', 'Pinaceae', 'Pistacia', 'Plantago', 
               'Platanus', 'Poaceae', 'PopulusSp', 'QuercusDeciduous', 'QuercusIlex', 'Ranunculaceae', 'Rhamnus', 'Rosaceae', 'Rumex', 'Salix',
               'Sanguisorba', 'Tilia', 'Ulmus', 'Urtica', 'ViburnumSambucusTp', 'VitisF', 'VitisS', 'XanthiumAmbrosia']

num_classes=len(class_names)


IMG_SIZE = 224
BATCH_SIZE = 32


liste_crossvalidation = ["fold1", "fold2","fold3" , "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"]


for idx in [1,2,3,4,5,6,7,8,9,10]:
    print("idx model", idx )
    for img_foldergroup_i in li_img_foldergroups:
        print("idx model", idx , "resol_i", img_foldergroup_i)

        cval_id = "fold{idx}"
        MODEL_I=f"resnet152_{cval_id}_checkpoint.pth"
        
        
        path_bestweights = p_models + MODEL_I 
        model_id = MODEL_I.replace("_checkpoint.pth", "")
        print(cval_id, model_id,  '\n')
        
                
        p_img_monitoring =  DATA_DIR / f"{img_foldergroup_i}" 
        print(os.listdir(p_img_monitoring))
        p_dump_pred = PRED_DIR / model_id  
        
        
        
        
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
        
        def load_image_paths(directory, color_mode='grayscale'):
            image_paths = []
            li_filenames = []
            labels=[]
        
        
            for slide in os.listdir(directory):
                slide_path = os.path.join(directory, slide)
                if os.path.isdir(slide_path):
                    for filename in os.listdir(slide_path):
                        if filename.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(slide_path, filename)
                            image_paths.append(image_path)
                            filename_i=slide+"/"+filename
                            li_filenames.append(filename_i)
                            labels.append(0)
        
            return np.array(image_paths), labels, np.array(li_filenames)
        
        
        
        # albumentations ----
        def get_transforms(img_size, is_training=True):
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
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
         
                # Read image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         
                # Apply Albumentations transforms
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
         
                return image, torch.tensor(label, dtype=torch.long)
        
        
        
        def load_initial_model(img_size, num_classes):
            base_model =  models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            num_ftrs = base_model.fc.in_features
            base_model.fc =     base_model.fc = nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(num_ftrs, num_classes)
            )
            return base_model
        
        def get_prediction_test_set(p_dump_pred, model_id,resol_i, img_data_loader, img_size, class_names, path_bestweights, filenames_p):
            num_classes=len(class_names)
            
            model = load_initial_model(img_size, num_classes)
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            model.load_state_dict(torch.load(path_bestweights, map_location=device))
        
            model.to(device)
            model.eval()
        
            y_pred_scores = []
            y_labels = []
        
            with torch.no_grad():
                for inputs, labels in img_data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    outputs = model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
        
                    y_labels.extend(labels.cpu().numpy())
                    y_pred_scores.extend(probabilities.cpu().numpy())
        
            y_labels = np.array(y_labels)
            y_pred_scores = np.array(y_pred_scores)
            y_pred_int = np.argmax(y_pred_scores, axis=1)
            y_pred_classes = np.array(class_names)[y_pred_int]
            

            with open(os.path.join(p_dump_pred, f'{model_id}_{resol_i}_filenames.pkl'), 'wb') as f:
                pickle.dump(filenames_p, f)
        
            with open(os.path.join(p_dump_pred, f'{model_id}_{resol_i}_pred_scores.pkl'), 'wb') as f:
                pickle.dump(y_pred_scores, f)
        
            with open(os.path.join(p_dump_pred, f'{model_id}_{resol_i}_pred_int.pkl'), 'wb') as f:
                pickle.dump(y_pred_int, f)
        
            with open(os.path.join(p_dump_pred, f'{model_id}_{resol_i}_pred_classes.pkl'), 'wb') as f:
                pickle.dump(y_pred_classes, f)
        
        def run_get_prediction(DATA_DIR, img_size, batch_size, p_dump_pred, model_id, resol_i, class_names, path_bestweights):
            os.makedirs(p_dump_pred, exist_ok=True)
        
            img_paths, labels, filenames_p = load_image_paths(DATA_DIR)
            
            data_transform = get_transforms(img_size, is_training=False)
            
            img_dataset = ClassificationDataset(img_paths, labels, transform=data_transform)
            
            
            img_data_loader = DataLoader(
                img_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
            )
            get_prediction_test_set(p_dump_pred, model_id, resol_i, img_data_loader, img_size, class_names, path_bestweights, filenames_p)
            return "done"
        
        
        # LAUNCHING TO GET THE PREDICTIONS
        run_get_prediction(p_img_monitoring, IMG_SIZE, BATCH_SIZE,  p_dump_pred, model_id, img_foldergroup_i, class_names, path_bestweights)

