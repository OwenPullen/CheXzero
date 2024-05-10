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
verbose = True
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

cxr_results.append(cxr_results_tta).to_csv('results/tta_results.csv')

import matplotlib.pyplot as plt

cxr2 = cxr_results.stack().reset_index(level =1)
cxr2.columns = ['Label', 'AUC_no_tta']
print(cxr2)
cxr2_tta = cxr_results_tta.stack().reset_index(level =1)
cxr2_tta.columns = ['Label', 'AUC_tta']
gph = pd.merge(cxr2, cxr2_tta, on='Label')
print(gph)
gph.to_csv('results/tta_results.csv')
index = np.arange(len(gph))
bar_width = 0.35
plt.bar(gph['Label'], gph['AUC_no_tta'], label = 'No TTA')
plt.bar(gph['Label'], gph['AUC_tta'], label='TTA')
plt.ylabel('AUC')
plt.title('AUC of model with and without TTA')
plt.xticks(rotation=90)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.25))
plt.tight_layout()
plt.savefig('results/tta_results.png')
print(gph)