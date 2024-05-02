import zero_shot as zs
from zero_shot_tta_adaptions import make
from torchvision.transforms import Normalize
from typing import Optional, List, Tuple
from pathlib import Path
from eval import evaluate, bootstrap

# ----- DIRECTORIES ------ #
model_path = "checkpoints/chexzero_weights/best_64_0.0001_original_35000_0.864.pt" # path to seemingly best performing model
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
# means computed from sample in `cxr_stats` notebook
transforms = [
    Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]

model, loader =  make(model_path,
     cxr_filepath = cxr_filepath,
       transforms = transforms,
       pretrained = True)

y_pred = zs.run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)

## Evaluate and make AUC
# make test_true
test_true = zs.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
# evaluate model
cxr_results = evaluate(y_pred, test_true, cxr_labels)
print(cxr_results)
# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(y_pred, test_true, cxr_labels)
# print(bootstrap_results)
# print(bootstrap_results[1])