import monai
from monai.data.test_time_augmentation import TestTimeAugmentation
from zero_shot_tta_adaptions import make
from tqdm import tqdm
import numpy as np
import zero_shot
from typing import List, Tuple, Optional
import pandas as pd
import torch
from graphs import *
from zero_shot import ensemble_models, make_true_labels
from tta_fns import ensemble_models_tta
import os
from tta_fns import run_softmax_eval
import eval
from pathlib import Path
import matplotlib.pyplot as plt
from reliability_diagrams import reliability_diagrams
from experiment_utils import *


cxr_true_labels_path: Optional[str] = 'data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
predictions_dir: Path = Path('predictions')
cache_dir: str = predictions_dir / "cached" 
model_dir: str = 'checkpoints/chexzero_weights' # where pretrained models are saved (.pt)
model_path = 'checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt'
cxr_filepath = 'data/chexpert_test_5.h5'
device = 'cuda'
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)

thresholds_dict = np.load('data/threshold_mcc_values.npy', allow_pickle=True).item()
print(thresholds_dict)
# Extract values from dictionary
thresholds_values = list(thresholds_dict.values())
print(thresholds_values)

model, loader = make(
    model_path = model_path,
    cxr_filepath = cxr_filepath
)

y_pred = zero_shot.run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
test_pred = y_pred
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
cxr_results: pd.DataFrame = zero_shot.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# run the TTA

transforms = monai.transforms.compose.Compose([
    monai.transforms.RandRotated(keys ='image', prob=0.2, range_x=(-0.5,0.5), range_y=(-0.5,0.5), keep_size=True),
    monai.transforms.RandGaussianNoised(keys='image', prob=0.2),
    monai.transforms.RandFlipd(keys='image', spatial_axis=1, prob=0.2),
    monai.transforms.RandAffined(keys='image', prob = 0.2, rotate_range=(0.5,0.5), shear_range=(0.25,0.25))
])
transforms = monai.transforms.RandomOrder(transforms)
verbose = False
y_pred_tta = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template, transforms=transforms, verbose=verbose)

if verbose: 
    print('images saved')
    import sys
    sys.exit()

test_pred_tta = y_pred_tta
test_true_tta = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results_tta: pd.DataFrame = eval.evaluate(test_pred_tta, test_true_tta, cxr_labels) # eval on full test datset
# # Display the results of the model without TTA and with TTA
print(cxr_results)
print(cxr_results_tta)

# from r_diags import get_reliability_diagrams_mp
preds_binary, conf = get_binary_preds(test_pred, thresholds_values)
preds_binary_tta, conf_tta = get_binary_preds(test_pred_tta, thresholds_values)
get_reliability_diagrams(test_true, preds_binary, conf, cxr_labels, 'Best Single Model')
get_reliability_diagrams(test_true_tta, preds_binary_tta, conf, cxr_labels, 'Best Single Model TTA')
# get_reliability_diagrams_mp(test_true, preds_binary, conf, cxr_labels, 'Best Single Model')
# get_reliability_diagrams_mp(test_true_tta, preds_binary, conf, cxr_labels, 'Best Single Model TTA')
plot_roc_mp(test_true, preds_binary, cxr_labels, 'Best Single Model')
plot_roc_mp(test_true_tta, preds_binary_tta, cxr_labels, 'Best Single Model TTA')
# plot_pr_mp(test_true, preds_binary, cxr_labels, 'Best Single Model')
# plot_pr_mp(test_true_tta, preds_binary_tta, cxr_labels, 'Best Single Model TTA')


predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir.joinpath('no_tta/'),
)

test_pred = y_pred_avg
# test_pred = torch.tensor(test_pred); test_true = torch.tensor(test_true)
# evaluate model
cxr_results_ens: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# Ensemble model prediction with TTA
predictions_tta, y_pred_avg_tta = ensemble_models_tta(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir.joinpath('tta/'),
    transforms=transforms,
)

cxr_results_ens_tta: pd.DataFrame = eval.evaluate(test_pred_tta, test_true, cxr_labels) # eval on full test datset


preds_binary, conf = get_binary_preds(test_pred, thresholds_values)
preds_binary_tta, conf_tta = get_binary_preds(test_pred_tta, thresholds_values)

get_reliability_diagrams(test_true, preds_binary, conf, cxr_labels, 'Ensemble Model')
get_reliability_diagrams(test_true_tta, preds_binary_tta, conf, cxr_labels, 'Ensemble Model TTA')
# get_reliability_diagrams_mp(test_true, preds_binary, conf, cxr_labels, 'Ensemble Model')
# get_reliability_diagrams_mp(test_true_tta, preds_binary_tta, conf, cxr_labels, 'Ensemble Model TTA')
plot_roc_mp(test_true, preds_binary, cxr_labels, 'Ensemble Model')
plot_roc_mp(test_true_tta, preds_binary_tta, cxr_labels, 'Ensemble Model TTA')
# plot_pr_mp(test_true, preds_binary, cxr_labels, 'Ensemble Model')
# plot_pr_mp(test_true_tta, preds_binary_tta, cxr_labels, 'Ensemble Model TTA')


# Stacked bar graph for AUC without TTA and with TTA
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # AUC values for without TTA

# auc_no_tta = pd.melt(pd.DataFrame(cxr_results, columns=cxr_labels))
# ax1.bar(cxr_labels, auc_no_tta, label='Without TTA')

# # AUC values for with TTA
# auc_with_tta = pd.melt(pd.DataFrame(cxr_results_tta, columns=cxr_labels))
# ax1.bar(cxr_labels, auc_with_tta, bottom=auc_no_tta, label='With TTA')

# ax1.set_ylabel('AUC')
# ax1.set_title('AUC Comparison: Single Model')

# # Stacked bar graph for AUC of ensemble models without TTA and with TTA
# # AUC values for without TTA
# auc_no_tta_ens = pd.melt(pd.DataFrame(cxr_results_ens, columns=cxr_labels))
# ax2.bar(cxr_labels, auc_no_tta_ens, label='Without TTA')
# auc_with_tta_ens = pd.melt(pd.DataFrame(cxr_results_ens_tta, columns=cxr_labels))
# ax2.bar(cxr_labels, auc_with_tta_ens, bottom=auc_with_tta_ens, label='With TTA')

# ax2.set_ylabel('AUC')
# ax2.set_title('AUC Comparison: Ensemble Models')

# # Add legend
# ax1.legend()
# ax2.legend()

# # Adjust spacing between subplots
# plt.subplots_adjust(hspace=0.5)

# # Show the plot
# plt.savefig('results/plots/AUC_comparison.png')