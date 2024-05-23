import monai
import monai.data
from monai.data.test_time_augmentation import TestTimeAugmentation
import monai.transforms
from zero_shot_tta_adaptions import make
from tqdm import tqdm
import numpy as np
import zero_shot
from typing import List, Tuple, Optional
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from zero_shot import ensemble_models
import os
from pathlib import Path
import eval

cxr_true_labels_path: Optional[str] = 'data/val_labs_2.csv' # (optional for evaluation) if labels are provided, provide path
# model_path = 'checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt'
cxr_filepath = 'data/val_all.h5'
device = 'cuda'
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']
model_dir: str = 'checkpoints/chexzero_weights' # where pretrained models are saved (.pt)
predictions_dir: Path = Path('predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" #
# ---- TEMPLATES ----- #
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# model, loader = make(
#     model_path = model_path,
#     cxr_filepath = cxr_filepath
# )
# Compute class weights
def compute_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, class_weights))

# Function to calculate weighted MCC
def weighted_mcc(y_true, y_pred, class_weights):
    mcc = matthews_corrcoef(y_true, y_pred)
    weight = class_weights[1] if np.sum(y_true) > 0 else 1  # Apply weight only if positive class exists
    return mcc * weight

def find_optimal_threshold(y_true, y_proba, class_weights):
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.0
    best_weighted_mcc = -1.0  # Initialize with a very low value
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        current_mcc = weighted_mcc(y_true, y_pred, class_weights)
        if current_mcc > best_weighted_mcc:
            best_weighted_mcc = current_mcc
            best_threshold = threshold
    
    return best_threshold, best_weighted_mcc

from sklearn.metrics import precision_recall_curve, f1_score, auc

# Function to find the optimal threshold based on F1 score
def find_optimal_threshold_pr(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f1_scores[optimal_idx]


model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)

# y_pred = zero_shot.run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir.joinpath('val/'),
)

test_pred = y_pred_avg


test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
cxr_results: pd.DataFrame = zero_shot.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset
print(cxr_results)


best_thresholds = []
opt_thresh = []
opt_thresh_pr = []
thresholds_mcc = np.linspace(0, 1, 100)
# pred_labels = test_pred[:,0]
# confidences = np.max(test_pred[:0], axis = 1)
for i in range(14):
    best_mcc = -1
    best_threshold = 0.5
    for threshold in thresholds_mcc:
        preds = (test_pred[:, i] >= threshold).astype(int)
        mcc = matthews_corrcoef(test_true[:,i], preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    # class_weights = compute_class_weights(test_true[:, i])
    # best_threshold, best_mcc = find_optimal_threshold(test_true[:, i], test_pred[:, i], class_weights=class_weights)
    best_thresholds.append(round(best_threshold, 3))
    fpr, tpr, thresholds = eval.roc_curve(test_true[:,i], test_pred[:,i])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thershold = thresholds[optimal_idx]
    opt_thresh.append(optimal_thershold)
    threshold_pr, f1 = find_optimal_threshold_pr(test_true[:, i], test_pred[:, i])
    opt_thresh_pr.append(threshold_pr)


print(best_thresholds)
print(cxr_labels)
print(opt_thresh)
print(opt_thresh_pr)

  
# thresholds_dict = dict(zip(cxr_labels, best_thresholds))
# print(thresholds_dict)
# np.save('data/thresholds.npy', thresholds_dict)
# thresholds_df = pd.DataFrame(thresholds_dict.items(), columns=['Label', 'Threshold'])
# thresholds_df.to_csv('data/thresholds.csv', index=False)
