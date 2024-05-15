from tta_fns import ensemble_models_tta
from typing import List, Tuple, Optional
from pathlib import Path
import os
from zero_shot import *
import eval 
import monai

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
cxr_labels: List[str] = [
    'Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

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

# Ensemble model prediction without TTA
predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir,
)

test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
test_pred = torch.tensor(test_pred); test_true = torch.tensor(test_true)
# evaluate model
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset
# Ensemble model prediction with TTA
transforms = monai.transforms.compose.Compose([
    monai.transforms.RandRotated(keys ='image', prob=0.2, range_x=(-0.5,0.5), range_y=(-0.5,0.5), keep_size=True),
    monai.transforms.RandGaussianNoised(keys='image', prob=0.2),
    monai.transforms.RandFlipd(keys='image', spatial_axis=1, prob=0.2),
    monai.transforms.RandAffined(keys='image', prob = 0.2, rotate_range=(0.5,0.5), shear_range=(0.25,0.25))
])

predictions_tta, y_pred_avg_tta = ensemble_models_tta(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
    transforms=transforms,
)

test_pred_tta = y_pred_avg_tta
# test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
test_pred_tta = torch.tensor(test_pred_tta)
# ; test_true = torch.tensor(test_true)
# evaluate model
cxr_results_tta: pd.DataFrame = eval.evaluate(test_pred_tta, test_true, cxr_labels) # eval on full test datset

print(cxr_results)
print(cxr_results_tta)
from torchmetrics.classification import BinaryCalibrationError
import re

bce = BinaryCalibrationError(norm = 'l1')

class_list = []
class_list_tta = []
for i in range(14):
    err_no_tta = bce(preds = torch.tensor(test_pred[:,i]), target = torch.tensor(test_true[:,i]))
    err_tta = bce(preds = torch.tensor(test_pred_tta[:,i]), target = torch.tensor(test_true[:,i]))
    print(err_no_tta, err_tta)
    class_list.append([ #cxr_labels[i],
                        err_no_tta, err_tta])

    # class_list_tta.append(err_tta)
print(class_list)


cxr2 = cxr_results.stack().reset_index(level =1)
cxr2.columns = ['Label', 'AUC_no_tta']
print(cxr2)
cxr2_tta = cxr_results_tta.stack().reset_index(level =1)
cxr2_tta.columns = ['Label', 'AUC_tta']
gph = pd.merge(cxr2, cxr2_tta, on='Label')
gph2 = pd.DataFrame.join(gph, pd.DataFrame(np.array(class_list), columns = ['BCE_no_tta', 'BCE_tta']))
print(gph2)

# Remove "_auc" from the Label column
gph2['Label'] = gph2['Label'].apply(lambda x: re.sub('_auc', '', x))

# Round all columns to 3 decimal places
gph2 = gph2.round(3)

# Save the modified dataframe to a CSV file
gph2.to_csv('results/ensemble_tta_results.csv', index=False)


