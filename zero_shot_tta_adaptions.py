from typing import List, Tuple
from pathlib import Path
import os
import numpy as np
from monai.data.test_time_augmentation import TestTimeAugmentation
import torch
import zero_shot as zs
from tqdm import tqdm
from eval import sigmoid
import matplotlib.pyplot as plt
from zero_shot import run_softmax_eval
from torchvision.transforms import Normalize, Resize, InterpolationMode, Compose

def ensemble_models_tta(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    transforms: List = [Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),],
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath,
            transforms=transforms,
        ) 
        
    #     # path to the cached prediction
    #     if cache_dir is not None:
    #         if save_name is not None: 
    #             cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
    #         else: 
    #             cache_path = Path(cache_dir) / f"{model_name}.npy"

    #     transforms = np.load(cache_path)
    #     # if prediction already cached, don't recompute prediction
    #     if cache_dir is not None and os.path.exists(cache_path): 
    #         print("Loading cached prediction for {}".format(model_name))
            # y_pred = np.load(cache_path)
        # else: # cached prediction not found, compute preds
        print("Inferring model {}".format(path))
        y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template) # transforms
        # if cache_dir is not None: 
        #         Path(cache_dir).mkdir(exist_ok=True, parents=True)
        #         np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

from zero_shot import load_clip, CXRTestDataset
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, RandomRotation

def make(
    model_path: str, 
    cxr_filepath: str,
    transforms: List,
    pretrained: bool = True, 
    context_length: bool = 77, 
):
    """
    FUNCTION: make
    -------------------------------------------
    This function makes the model, the data loader, and the ground truth labels. 
    
    args: 
        * model_path - String for directory to the weights of the trained clip model. 
        * context_length - int, max number of tokens of text inputted into the model. 
        * cxr_filepath - String for path to the chest x-ray images. 
        * cxr_labels - Python list of labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * pretrained - bool, whether or not model uses pretrained clip weights
        * cutlabels - bool, if True, will keep columns of ground truth labels that correspond
        with the labels inputted through `cxr_labels`. Otherwise, drop the first column and keep remaining.
    
    Returns model, data loader. 
    """
    # load model
    model = load_clip(
        model_path=model_path, 
        pretrained=pretrained, 
        context_length=context_length
    )

    # load data
    # Apply transformations
         # means computed from sample in `cxr_stats` notebook
        # Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    transformations = transforms

    # if using CLIP pretrained model
    if pretrained: 
        # resize to input resolution of pretrained clip model
        input_resolution = 224
        transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    
    # create dataset
    torch_dset = CXRTestDataset(
        img_path=cxr_filepath,
        transform=transform, 
    )
    loader = torch.utils.data.DataLoader(torch_dset, shuffle=False)
    
    return model, loader