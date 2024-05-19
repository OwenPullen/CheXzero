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

cxr_true_labels_path: Optional[str] = 'data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
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

model, loader = make(
    model_path = model_path,
    cxr_filepath = cxr_filepath
)

# Regular single model prediction without TTA
y_pred = zero_shot.run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
test_pred = y_pred
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
cxr_results: pd.DataFrame = zero_shot.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# run the TTA
from tta_fns import run_softmax_eval
import eval
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

from torchmetrics.classification import BinaryCalibrationError
import re

bce = BinaryCalibrationError(norm = 'l1')
class_list = []
class_list_tta = []
values = []

pred_labels = test_pred[:,0]
confidences = np.max(test_pred[:0], axis = 1)
for i in range(14):
    err_no_tta = bce(preds = torch.tensor(test_pred[:,i]), target = torch.tensor(test_true[:,i]))
    err_tta = bce(preds = torch.tensor(test_pred_tta[:,i]), target = torch.tensor(test_true_tta[:,i]))
    print(err_no_tta, err_tta)
    class_list.append(err_no_tta)
    class_list_tta.append(err_tta)

    # class_list_tta.append(err_tta)

fig, ax = bce.plot(class_list)
fig.savefig('results/fig.png')
print(class_list)

import matplotlib.pyplot as plt
from reliability_diagrams import reliability_diagrams

pred_lab = []
for row in test_pred:
    threshold_row = []
    for num in row:
        if num > 0.5: threshold_row.append(1)
        else: threshold_row.append(0)
    pred_lab.append(threshold_row)

df_1 = pd.DataFrame({
    'true_label': test_true[:,0],
    'pred_label': pred_lab[:,0],
    'confidence': test_pred[:,0],
})

y_true = df_1.true_label.values
y_pred = df_1.pred_label.values
y_conf = df_1.confidence.values

plt.style.use("seaborn")
plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

title = 'Plot'
fig = reliability_diagrams(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                           draw_bin_importance="alpha", draw_averages=True,
                           title=title, figsize=(6,6), dpi=100,
                           return_fig=True
)
fig.savefig('results/reliability_diagram.png')
# ece = BinaryCalibrationError(norm='l1')
# values = []
# values_tta = []

# for _ in range(14):
#     values.append(ece(preds = torch.tensor(test_pred[:,i]), target = torch.tensor(test_true[:,i])))
#     # values_tta.append(ece.update(preds = torch.tensor(test_pred_tta[:,i]), target = torch.tensor(test_true_tta[:,i])))

# fig, ax = ece.plot(values)
# fig.savefig('results/fig.png')
# print('printing values:')
# print(values)
# fig_tta, ax_tta = ece.plot(values_tta)
    
# # join class list to cxr results and cxr_results_tta
# cxr_results = cxr_results.join(pd.DataFrame(class_list, columns = ['BCE_no_tta']))
# cxr_results = cxr_results.join(pd.DataFrame(class_list_tta, columns = ['BCE_tta']))

# print(cxr_results)
# print(cxr_results_tta)


# cxr2 = cxr_results.stack().reset_index(level =1)
# cxr2.columns = ['Label', 'AUC_no_tta']
# print(cxr2)
# cxr2_tta = cxr_results_tta.stack().reset_index(level =1)
# cxr2_tta.columns = ['Label', 'AUC_tta']
# gph = pd.merge(cxr2, cxr2_tta, on='Label')
# gph2 = pd.DataFrame.join(gph, pd.DataFrame(np.array(class_list), columns = ['BCE_no_tta', 'BCE_tta']))
# print(gph2)

# # Remove "_auc" from the Label column
# gph2['Label'] = gph2['Label'].apply(lambda x: re.sub('_auc', '', x))

# # Round all columns to 3 decimal places
# gph2 = gph2.round(3)

# Save the modified dataframe to a CSV file
# gph2.to_csv('results/tta_results.csv', index=False)

class_list = []
for i in range(14):
    err_no_tta = bce(preds = torch.tensor(test_true[:,i]).float(), target = torch.tensor(test_true[:,i]))
    err_tta = bce(preds = torch.tensor(test_true[:,i]).float(), target = torch.tensor(test_true[:,i]))
    # bce.update(preds=torch.tensor(test_true[:,i].f))
    print(err_no_tta, err_tta)
    class_list.append([ #cxr_labels[i],
                        err_no_tta, err_tta])

    # class_list_tta.append(err_tta)
print(class_list)

# fig, ax = bce.plot()

# ax.set_fontsize(fs=20)
# fig.savefig('plot.png')


# from torchmetrics.classification.calibration_error import MulticlassCalibrationError
# from torchmetrics.wrappers import ClasswiseWrapper
# calibration_error = ClasswiseWrapper(MulticlassCalibrationError(num_classes=14, norm='l1')) # L! norm makes it ECE
# test_true = np.array(test_true); 
#.to(torch.float16)#, 0)
# test_true_one_hot = torch.eye(14)[test_true]
# # ece = calibration_error(preds = test_pred, target = test_true.argmax(-1))
# print('Calibration Error without TTA: ', ece)
# ece_tta = calibration_error(torch.tensor(test_pred_tta), test_true_one_hot)
#                             #  torch.tensor(test_true_tta).argmax(-1))
# print('Calibration Error with TTA: ', ece_tta)





# cxr_results.append(cxr_results_tta).to_csv('results/tta_results.csv')

# import matplotlib.pyplot as plt

# cxr2 = cxr_results.stack().reset_index(level =1)
# cxr2.columns = ['Label', 'AUC_no_tta']
# print(cxr2)
# cxr2_tta = cxr_results_tta.stack().reset_index(level =1)
# cxr2_tta.columns = ['Label', 'AUC_tta']
# gph = pd.merge(cxr2, cxr2_tta, on='Label')
# print(gph)
# gph.to_csv('results/tta_results.csv')
# index = np.arange(len(gph))
# bar_width = 0.35
# plt.bar(gph['Label'], gph['AUC_no_tta'], label = 'No TTA')
# plt.bar(gph['Label'], gph['AUC_tta'], label='TTA')
# plt.ylabel('AUC')
# plt.title('AUC of model with and without TTA')
# plt.xticks(rotation=90)
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1.25))
# plt.tight_layout()
# plt.savefig('results/tta_results.png')
# print(gph)
# import matplotlib.pyplot as plt
# from reliability_diagrams import *

# plt.style.use("seaborn")
# plt.rc("font", size=12)
# plt.rc("axes", labelsize=12)
# plt.rc("xtick", labelsize=12)
# plt.rc("ytick", labelsize=12)
# plt.rc("legend", fontsize=12)

# plt.rc("axes", titlesize=16)
# plt.rc("figure", titlesize=16)

# title = "ECE of model with and without TTA"

# y_pred = test_pred
# y_true = test_true
# y_conf = class_list[

# fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
#                           draw_bin_importance="alpha", draw_averages=True,
#                           title=title, figsize=(6, 6), dpi=100, 
#                           return_fig=True)


