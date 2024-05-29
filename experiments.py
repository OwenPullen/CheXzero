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

# define the TTA
transforms = monai.transforms.compose.Compose([
    monai.transforms.RandRotated(keys ='image', prob=0.05, range_x=(-0.5,0.5), range_y=(-0.5,0.5), keep_size=True),
    monai.transforms.RandGaussianNoised(keys='image', prob=0.05),
    monai.transforms.RandFlipd(keys='image', spatial_axis=1, prob=0.05),
    monai.transforms.RandAffined(keys='image', prob = 0.05, rotate_range=(0.5,0.5), shear_range=(0.25,0.25))
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
# filter positives and get confidence values
positive_preds, true_positives, conf = filter_positives(test_pred, test_true, thresholds_values)
positive_preds_tta, true_positives_tta, conf_tta = filter_positives(test_pred, test_true, thresholds_values)

# Get reliability diagrams
# get_reliability_diagrams(test_true, preds_binary, conf, cxr_labels, 'Best Single Model', 'single_model_all')
# get_reliability_diagrams(test_true_tta, preds_binary_tta, conf, cxr_labels, 'Best Single Model TTA', 'single_model_tta_all')

# plot reliability diagrams - postive predictions
# get_reliability_diagrams(true_positives, positive_preds, conf, cxr_labels, 'Best Single Model Positives', 'single_model_positives', pos_only=True)
# get_reliability_diagrams(true_positives, positive_preds_tta, conf_tta, cxr_labels, 'Best Single Model TTA Positives', 'single_model_tta_positives', pos_only=True)

# Get ROC curves
# plot_roc_mp(test_true, test_pred, cxr_labels, 'Best Single Model', 'roc')
# plot_roc_mp(test_true_tta, test_pred_tta, cxr_labels, 'Best Single Model TTA', 'roc')
# get_roc_plots(test_true, test_pred, cxr_labels, 'single_model')
# get_roc_plots(test_true, test_pred_tta, cxr_labels, 'single_model_tta')

predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir.joinpath('no_tta/'),
)

test_pred_en = y_pred_avg
# test_pred = torch.tensor(test_pred); test_true = torch.tensor(test_true)
# evaluate model
cxr_results_ens: pd.DataFrame = eval.evaluate(test_pred_en, test_true, cxr_labels) # eval on full test datset

# Ensemble model prediction with TTA
predictions_tta, y_pred_avg_tta = ensemble_models_tta(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir.joinpath('tta/'),
    transforms=transforms,
)

test_pred_tta_en = y_pred_avg_tta

cxr_results_ens_tta: pd.DataFrame = eval.evaluate(test_pred_tta_en, test_true, cxr_labels) # eval on full test datset


preds_binary_en, conf = get_binary_preds(test_pred_en, thresholds_values)
preds_binary_tta_en, conf_tta_en = get_binary_preds(test_pred_tta_en, thresholds_values)

# Extract positive preds and confidence values
positive_preds_en, true_positives_en, conf_en = filter_positives(test_pred_en, test_true, thresholds_values)
positive_preds_tta_en, true_positives_tta_en, conf_tta_en = filter_positives(test_pred_tta_en, test_true, thresholds_values)

# Plot reliability diagrams
# get_reliability_diagrams(test_true, preds_binary_en, conf, cxr_labels, 'Ensemble Model', 'ensemble_model_all')
# get_reliability_diagrams(test_true_tta, preds_binary_tta_en, conf_tta_en, cxr_labels, 'Ensemble Model TTA', 'ensemble_model_tta_all')

# plot reliability diagrams - postive predictions
# get_reliability_diagrams(true_positives_en, positive_preds_en, conf_en, cxr_labels, 'Ensemble Model Positives', 'ensemble_model_positives', pos_only=True)
# get_reliability_diagrams(true_positives_tta_en, positive_preds_tta_en, conf_tta_en, cxr_labels, 'Ensemble Model TTA Positives', 'ensemble_model_tta_positives',pos_only=True)

# get_reliability_diagrams_mp(test_true, preds_binary, conf, cxr_labels, 'Ensemble Model')
# get_reliability_diagrams_mp(test_true_tta, preds_binary_tta_en, conf_tta_en, cxr_labels, 'Ensemble Model TTA')

# Get ROC curves
# plot_roc_mp(test_true, test_pred_en, cxr_labels, 'Ensemble Model', 'roc')
# plot_roc_mp(test_true, test_pred_tta_en, cxr_labels, 'Ensemble Model TTA', 'roc')
# get_roc_plots(test_true, test_pred_en, cxr_labels, 'ensemble_model')
# get_roc_plots(test_true, test_pred_tta_en, cxr_labels, 'ensemble_model_tta')

# Create a multipanel box plot of the AUC Values for each experiment
plt.figure(figsize=(10, 6))
# Reshape the dataframe into long format
cxr_results_long = cxr_results.melt(var_name='Class', value_name='AUC').assign(Experiment='Single Model')
cxr_results_tta_long = cxr_results_tta.melt(var_name='Class', value_name='AUC').assign(Experiment='Single Model TTA')
cxr_results_ens_long = cxr_results_ens.melt(var_name='Class', value_name='AUC').assign(Experiment='Ensemble Model')
cxr_results_ens_tta_long = cxr_results_ens_tta.melt(var_name='Class', value_name='AUC').assign(Experiment='Ensemble Model TTA')

# Combine the dataframes
combined_results = pd.concat([cxr_results_long, cxr_results_tta_long, cxr_results_ens_long, cxr_results_ens_tta_long])
# Add a column for the experiment
import seaborn as sns
import re
from sklearn.metrics import roc_curve, auc
import numpy as np
# Remove "_auc" from x-axis labels
combined_results['Class'] = combined_results['Class'].apply(lambda x: re.sub('_auc', '', x))

# Plot the dotplot
plt.figure(figsize=(10, 6))
sns.stripplot(data=combined_results, x='AUC', y='Class', hue='Experiment', dodge=True, palette='Set1', marker='s')
plt.xlabel('AUC Value')
plt.title('AUC Values for Different Experiments')
plt.legend(title='Experiment', borderaxespad=0.5, frameon=True, edgecolor='black')
# plt.xticks(rotation=90)  # Rotate x-axis labels vertically
plt.tight_layout()  # Adjust the layout to prevent labels from being cut off
plt.savefig('results/plots/AUC/AUC_Dotplot.png')




import matplotlib.pyplot as plt

# Extract the AUC values for each experiment
single_model_auc = cxr_results_long['AUC'].values
single_model_tta_auc = cxr_results_tta_long['AUC'].values
ensemble_model_auc = cxr_results_ens_long['AUC'].values
ensemble_model_tta_auc = cxr_results_ens_tta_long['AUC'].values

# Create a list of colors for each experiment
colors = ['green', 'orange', 'red', 'blue']
experiment_labels = ['Single Model', 'Single Model TTA', 'Ensemble Model', 'Ensemble Model TTA']
# Set the number of classes
num_classes = len(cxr_labels)

# Create a list of angles for each class
angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
angles += angles[:1]

# Create a radar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.spines['polar'].set_visible(False)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cxr_labels, fontsize=9)
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], color="black", size=7)
plt.ylim(0,1)

# Plot the AUC values for each experiment
for i, (auc_values, color) in enumerate(zip([single_model_auc, single_model_tta_auc, ensemble_model_auc, ensemble_model_tta_auc], colors)):
    values = np.concatenate((auc_values, [auc_values[0]]))
    ax.fill(angles, values, color=color, alpha=0.5, label=f'{experiment_labels[i]}')
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')

# Add a scale
ax.set_rlabel_position(0)
ax.grid(True)

# Adjust the layout to prevent labels from being cut off
plt.tight_layout()

# Add a legend
ax.legend(loc='upper center', bbox_to_anchor=(0.53, 1.15), ncol=4)

# Show the plot
plt.savefig('results/plots/AUC/AUC_Radar.png')

chexero_list = ['Atelectasis','Cardiomegaly','Consolidation', 'Edema', 'Pleural Effusion']

# Create multipanel ROC curves of the AUC Values for each experiment matching chexzero paper
#     roc_auc_en = auc(fpr_en, tpr_en)
#     plt.plot(fpr_en, tpr_en, label='Ensemble Model (AUC = %0.2f)' % roc_auc_en)
#     plt.grid(False)  # Remove grid lines
     
#     fpr_tta_en, tpr_tta_en, _ = roc_curve(test_true[:, i], test_pred_tta_en[:, i])
#     roc_auc_tta_en = auc(fpr_tta_en, tpr_tta_en)
#     plt.plot(fpr_tta_en, tpr_tta_en, label='Ensemble Model TTA (AUC = %0.2f)' % roc_auc_tta_en)
    
#     plt.title(label)
#     plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig('results/plots/AUC/ROC_Multipanel.png')
fpr_single = dict()
tpr_single = dict()
fpr_single_tta = dict()
tpr_single_tta = dict()
fpr_ensemble = dict()
tpr_ensemble = dict()
fpr_ensemble_tta = dict()
tpr_ensemble_tta = dict()
roc_auc_single = dict()
roc_auc_single_tta = dict()
roc_auc_ensemble = dict()
roc_auc_ensemble_tta = dict()


plt.figure(figsize=(15, 10))
for i, label in enumerate(chexero_list):
    # Single Model
    fpr_single[i], tpr_single[i], _ = roc_curve(test_true[:, i], test_pred[:, i])
    roc_auc_single[i] = auc(fpr_single[i], tpr_single[i])
        
    # Single Model TTA
    fpr_single_tta[i], tpr_single_tta[i], _ = roc_curve(test_true_tta[:, i], test_pred_tta[:, i])
    roc_auc_single_tta[i] = auc(fpr_single_tta[i], tpr_single_tta[i])
    
    # Ensemble Model
    fpr_ensemble[i], tpr_ensemble[i], _ = roc_curve(test_true[:, i], test_pred_en[:, i])
    roc_auc_ensemble[i] = auc(fpr_ensemble[i], tpr_ensemble[i])
    
    # Ensemble Model TTA
    fpr_ensemble_tta[i], tpr_ensemble_tta[i], _ = roc_curve(test_true[:, i], test_pred_tta_en[:, i])
    roc_auc_ensemble_tta[i] = auc(fpr_ensemble_tta[i], tpr_ensemble_tta[i])


plt.figure(figsize=(15, 10), dpi=500)  # Increase the dpi for higher resolution
plt.subplot(2, 2, 1).set_facecolor('white')
for i, label in enumerate(chexero_list):
    plt.plot(fpr_single[i], tpr_single[i], label=f'{label} (AUC = {roc_auc_single[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Single Model')
legend1 = plt.legend(borderaxespad=0.5, frameon=True, edgecolor='black')
legend1.get_frame().set_facecolor('white')
legend1.get_frame().set_linewidth(1)
plt.grid(False)

plt.subplot(2, 2, 2).set_facecolor('white')
for i, label in enumerate(chexero_list):
    plt.plot(fpr_single_tta[i], tpr_single_tta[i], label=f'{label} (AUC = {roc_auc_single_tta[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Single Model TTA')
legend2 = plt.legend(borderaxespad=0.5, frameon=True, edgecolor='black')
legend2.get_frame().set_facecolor('white')
legend2.get_frame().set_linewidth(1)
plt.grid(False)


plt.subplot(2, 2, 3).set_facecolor('white')
for i, label in enumerate(chexero_list):
    plt.plot(fpr_ensemble[i], tpr_ensemble[i], label=f'{label} (AUC = {roc_auc_ensemble[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ensemble Model')
legend3 = plt.legend(borderaxespad=0.5, frameon=True, edgecolor='black')
legend3.get_frame().set_facecolor('white')
legend3.get_frame().set_linewidth(1)
plt.grid(False)  # Set the background color to white to match the report she

plt.subplot(2, 2, 4).set_facecolor('white')
for i, label in enumerate(chexero_list):
    plt.plot(fpr_ensemble_tta[i], tpr_ensemble_tta[i], label=f'{label} (AUC = {roc_auc_ensemble_tta[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ensemble Model TTA')
plt.grid(False)
legend4 = plt.legend(borderaxespad=0.5, frameon=True, edgecolor='black')
legend4.get_frame().set_facecolor('white')
legend4.get_frame().set_linewidth(1)
  # Set the background color to white to match the report she

plt.tight_layout()


plt.savefig('results/plots/report_plots/roc/ROC_Multipanel2.png', dpi=500)  # Increase the dpi for higher resolution

# Create a DataFrame with the AUC results
auc_results = pd.DataFrame({
    'Class': cxr_labels,
    'Single Model': cxr_results_long[cxr_results_long['Experiment'] == 'Single Model']['AUC'].round(3),
    'Single Model TTA': cxr_results_tta_long[cxr_results_tta_long['Experiment'] == 'Single Model TTA']['AUC'].round(3),
    'Ensemble Model': cxr_results_ens_long[cxr_results_ens_long['Experiment'] == 'Ensemble Model']['AUC'].round(3),
    'Ensemble Model TTA': cxr_results_ens_tta_long[cxr_results_ens_tta_long['Experiment'] == 'Ensemble Model TTA']['AUC'].round(3)
})

# Save the DataFrame as a CSV file
auc_results.to_csv('/home/kell7033/git/CheXzero/auc_results.csv', index=False)

