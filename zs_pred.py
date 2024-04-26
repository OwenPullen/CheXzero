import zero_shot
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import os
from monai.transforms import Compose, Rotate, Flip, ToTensor
from monai.data.test_time_augmentation import TestTimeAugmentation
import torch
import tta_fns
import eval
## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = 'data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = 'data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = 'checkpoints/chexzero_weights' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77 # length of context for each label
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

# computes predictions for a set of images stored as a np array of probabilities for each pathology
predictions, y_pred_avg = zero_shot.ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
)

# loads in ground truth labels into memory
test_pred = y_pred_avg
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# boostrap evaluations for 95% confidence intervals
bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

# print results with confidence intervals
print(bootstrap_results[1])

# TTA
# Define transforms for TTA
# transforms = Compose([
#     Rotate(angle=30),
#     Flip(spatial_axis = 0),
#     ToTensor()
# ])
from monai.transforms import Compose, Rotate, Flip, ToTensor, RandAffine
# transforms = RandAffine(
#     rotate_range=(-30, 30),
#     translate_range=(0.1, 0.1),
#     scale_range=(0.9, 1.1),
#     shear_range=(0.1, 0.1),
#     mode=("bilinear", "nearest"),
# )

# model_pth = model_paths[0]
# model, loader = zero_shot.make(model_, cxr_filepath)

transforms = Rotate(angle=30)


predictions_tta, y_pred_avg_tta = tta_fns.ensemble_models_tta(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    transforms=transforms,
    cache_dir=cache_dir,
)
# all_augmented_inputs = torch.cat(augmented_inputs, dim=0)

# loads in ground truth labels into memory
test_pred_tta = y_pred_avg_tta
# test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results_tta: pd.DataFrame = eval.evaluate(test_pred_tta, test_true, cxr_labels) # eval on full test datset

# boostrap evaluations for 95% confidence intervals
bootstrap_results_tta: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred_tta, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

# print results with confidence intervals
print('TTA Results:')
print(bootstrap_results_tta[1])

print('Results:')
print(bootstrap_results[1])

