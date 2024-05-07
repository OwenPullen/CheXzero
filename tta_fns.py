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
from torchvision.transforms import Normalize, Resize, InterpolationMode, Compose

def ensemble_models_tta(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    transforms,
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
        model, loader = zs.make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
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
        y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template, transforms=transforms)
        # if cache_dir is not None: 
        #         Path(cache_dir).mkdir(exist_ok=True, parents=True)
        #         np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

def run_softmax_eval(model, loader, eval_labels: list, pair_template: tuple, transforms, context_length: int = 77): 
    """
    Run softmax evaluation to obtain a single prediction from the model.
    """
     # get pos and neg phrases
    pos = pair_template[0]
    neg = pair_template[1]

    # get pos and neg predictions, (num_samples, num_classes)
    pos_pred = run_single_prediction_tta(eval_labels, pos, model, loader, transforms,
                                     softmax_eval=True, context_length=context_length) 
    neg_pred = run_single_prediction_tta(eval_labels, neg, model, loader, transforms,
                                     softmax_eval=True, context_length=context_length) 

    # compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    return y_pred

def run_single_prediction_tta(cxr_labels, template, model, loader, transforms, softmax_eval=True, context_length=77): 
    """
    FUNCTION: run_single_prediction
    --------------------------------------
    This function will make probability predictions for a single template
    (i.e. "has {}"). 
    
    args: 
        * cxr_labels - list, labels for a specific zero-shot task. (i.e. ['Atelectasis',...])
        * template - string, template to input into model. 
        * model - PyTorch model, trained clip model
        * loader - PyTorch data loader, loads in cxr images
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * context_length (optional) - int, max number of tokens of text inputted into the model.
        
    Returns list, predictions from the given template. 
    """
    cxr_phrase = [template]
    zeroshot_weights = zs.zeroshot_classifier(cxr_labels, cxr_phrase, model, context_length=context_length)
    y_pred = predict_tta(loader, model, zeroshot_weights, transforms, softmax_eval=softmax_eval)
    return y_pred

def predict_tta(loader, model, zeroshot_weights, transforms, softmax_eval=True, verbose=False): 
    """
    FUNCTION: predict
    ---------------------------------
    This function runs the cxr images through the model 
    and computes the cosine similarities between the images
    and the text embeddings. 
    
    args: 
        * loader -  PyTorch data loader, loads in cxr images
        * model - PyTorch model, trained clip model 
        * zeroshot_weights - PyTorch Tensor, outputs of text encoder for labels
        * softmax_eval (optional) - Use +/- softmax method for evaluation 
        * verbose (optional) - bool, If True, will print out intermediate tensor values for debugging.
        
    Returns numpy array, predictions on all test data samples. 
    """
    device = 'cuda'
    tta = TestTimeAugmentation(transform=transforms, batch_size=10, return_full_data=True, device=device)
    y_pred = []
    with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                images = data['img']
                images = images.to(device)
                    # tta = TestTimeAugmentation(transform=transforms, batch_size=10, return_full_data=True, device=device)
                    # img_element = images[0,0,0].item()
                images = {"image": images[0]}
                images = tta(images)
                    # import pdb; pdb.set_trace()
                    # debug goes here: directly apply the transform to the image
                    #_out = transforms(images[0])
                    #import pdb; pdb.set_trace()
                    #images  =tta()
                    # import pdb; pdb.set_trace()
                    # predict
                image_features = model.encode_image(images) 
                image_features /= image_features.norm(dim=-1, keepdim=True) # (1, 768)
                image_features = torch.tensor(image_features, dtype=torch.float32)
                # obtain logits
                zeroshot_weights = torch.tensor(zeroshot_weights, dtype=torch.float32)
                logits = image_features @ zeroshot_weights # (1, num_classes)
                logits = torch.Tensor.cpu(logits)
                logits = np.squeeze([logits.numpy()], axis=0) # (num_classes,)
                logits = np.mean(logits, axis=0)

                if softmax_eval is False: 
                    norm_logits = (logits - logits.mean()) / (logits.std())
                    logits = sigmoid(norm_logits) 
                
                y_pred.append(logits)
                
                if verbose: 
                    images = torch.Tensor.cpu(images)
                    plt.imshow(images[0][0])
                    plt.show()
                    print('images: ', images)
                    print('images size: ', images.size())
                    
                    print('image_features size: ', image_features.size())
                    print('logits: ', logits)
                    print('logits size: ', logits.size())
         
    y_pred = np.array(y_pred)
    return np.array(y_pred)

from zero_shot import load_clip, CXRTestDataset
def make(
    model_path: str, 
    cxr_filepath: str, 
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
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
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

class ImageTransform:
    def __init__(self, transform_func: callable):
        self.transform_func = transform_func
    
    def __call__(self, img):
        pass

