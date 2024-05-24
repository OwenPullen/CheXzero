import numpy as np

def get_binary_preds(test_pred, thresholds_values):
    preds_binary = np.zeros_like(test_pred)
    conf = np.zeros_like(test_pred)
    for i in range(test_pred.shape[1]):
        preds_binary[:, i] = (test_pred[:, i] > thresholds_values[i]).astype(int)
        confidence = np.copy(test_pred[:, i])
        confidence[preds_binary[:, i] == 0] = 1 - confidence[preds_binary[:, i] == 0]
        conf[:, i] = confidence
    return preds_binary, conf