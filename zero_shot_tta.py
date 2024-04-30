
# # Sample Notebook for Zero-Shot Inference with CheXzero
# This notebook walks through how to use CheXzero to perform zero-shot inference on a chest x-ray image dataset.


# ## Import Libraries


import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval


# ## Directories and Constants


## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = 'data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = 'data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = 'checkpoints/chexzero_weights' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)
        
print(model_paths)


# ## Run Inference

## Run the model on the data set using ensembled models
from zero_shot import ensemble_models

predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
)


# save averaged preds
pred_name = "chexpert_preds.npy" # add name of preds
predictions_dir_reg = predictions_dir / pred_name
np.save(file=predictions_dir_reg, arr=y_pred_avg)


# ## (Optional) Evaluate Results
# If ground truth labels are available, compute AUC on each pathology to evaluate the performance of the zero-shot model. 
# make test_true
test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model
cxr_results = evaluate(test_pred, test_true, cxr_labels)

# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)


# display AUC with confidence intervals
print(bootstrap_results[1])

     # means computed from sample in `cxr_stats` notebook
        # Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),


# # ## Test Time Augmentation (TTA)
from zero_shot_tta_adaptions import ensemble_models_tta
import torchvision.transforms as T
from torchvision.transforms import Normalize, RandomRotation, Resize, InterpolationMode, ToTensor, RandomApply
from zero_shot_tta_adaptions import run_model_with_transforms

# apply multiple stacked transforms

transforms = [
    RandomApply([Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))], p=0.9),
    RandomApply([RandomRotation(30)], p=0.1),
    RandomApply([RandomRotation(50, expand=True)], p=0.1),
    T.RandomHorizontalFlip(p=0.1),
    T.RandomVerticalFlip(p=0.1),
    RandomApply([T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)], p=0.05),
    RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    RandomApply([T.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.2),
    RandomApply([T.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)], p=0.1),
    T.RandomSolarize(192,p=0.1),
    T.RandomErasing(p=0.1),
]

res_all = run_model_with_transforms(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    results_file="results/combined_results_all.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_all.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms
)

# ## Different types of augmentations -rotate flip (rf)

transforms_rf = [
    Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    RandomApply([RandomRotation(30)], p=0.5),
    RandomApply([RandomRotation(50, expand=True)], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
]

res_rf = run_model_with_transforms(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    results_file="results/combined_results_rf.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_rf.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms_rf
)
#
# ## Different types of augmentations - image distortions (dt)
transforms_dt = [
    Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    RandomApply([T.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
    T.RandomSolarize(192,p=0.5),
    T.RandomAutocontrast(p=0.5),
    T.RandomEqualize(p=0.5),
    T.RandomGrayscale(p=0.5),
    T.RandomInvert(p=0.5),
]

res_dt = run_model_with_transforms(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    results_file="results/combined_results_dt.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_dt.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms_dt
)

# # # ## Different types of augmentations - Resizing and cropping (rc)
transforms_rc = [
    Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    RandomApply([T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR)], p=0.5),
    RandomApply([T.CenterCrop((448, 448))], p=0.5),
    T.RandomErasing(p=0.1),
    T.RandomCrop((448, 448), padding=4),
    ]

res_rc = run_model_with_transforms(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    results_file="results/combined_results_rc.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_rc.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms_rc
)

# # ## Different types of augmentations - Random Affine (ra)
transforms_ra = [
    Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    RandomApply([T.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)], p=0.9),
    ]

res_ra = run_model_with_transforms(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    results_file="results/combined_results_ra.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_ra.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms_ra
)

# # ## Different types of augmentations - don't normailze (dn)
transforms_dn = [
    RandomApply([Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))], p=0.0),
    ]

res_dn = run_model_with_transforms(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    results_file="results/combined_results_dn.csv",
    predictions_dir=predictions_dir,
    pred_name_tta_transforms="tta_chexpert_preds_dn.npy",
    test_true=test_true,
    bootstrap_results=bootstrap_results,
    transforms=transforms_dn
)

# Append all res... arrays together
all_results = np.concatenate((res_all, res_rf, res_dt, res_rc, res_ra, res_dn))

# Label the results
result_labels = ['All', 'Rotate Flip', 'Image Distortions', 'Resizing and Cropping', 'Random Affine', 'Don\'t Normalize']

# Print the results
for i, result in enumerate(all_results):
    print(f"{result_labels[i]}: {result}")