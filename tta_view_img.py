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
import torchvision

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
                torchvision.utils.save_image(images, f'data/tta_images/img_{i}_original.png')
                images = images.to(device)
                    # tta = TestTimeAugmentation(transform=transforms, batch_size=10, return_full_data=True, device=device)
                    # img_element = images[0,0,0].item()
                images = {"image": images[0]}
                images = tta(images)
                torchvision.utils.save_image(images, f'data/tta_images/img_{i}_tta.png')
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

                if i >= 20:
                    break
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