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
cxr_filepath = 'data/chexpert_test.h5'
device = 'cuda'
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

transforms = monai.transforms.Rotated(keys ='image', angle=0.5)
model, loader = make(
    model_path = model_path,
    cxr_filepath = cxr_filepath
)

# monai.transforms.LoadImage(dtype=np.float32, 'image', image_only=True)



# run the TTA
from tta_fns import run_softmax_eval
import eval

y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template, transforms=transforms)

test_pred = y_pred
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

print(cxr_results)