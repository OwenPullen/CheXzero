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

def filter_positives(test_pred, true_labels, thresholds_values):
    # filter out the predictions and true labels for positive cases
    pred_binary, conf = get_binary_preds(test_pred, thresholds_values)
    condition = (true_labels == 1) | (pred_binary == 1)
    pred_pos = np.where(condition, pred_binary, np.nan).astype(int)
    true_pos = np.where(condition, true_labels, np.nan).astype(int)
    conf_pos = np.where(condition, conf, np.nan).astype(int)
    return pred_pos, true_pos, conf_pos


def get_binary_preds_pos(test_pred, test_true, thresholds_values):
    preds_binary = []
    conf = []
    true = []
    for i in range(test_pred.shape[1]):
        preds_binary_temp = (test_pred[:, i] > thresholds_values[i]).astype(int)
        confidence_temp = np.copy(test_pred[:, i])
        confidence_temp[preds_binary_temp == 0] = 1 - confidence_temp[preds_binary_temp == 0]
        # Create a mask for positive cases
        pos_mask = (preds_binary_temp == 1) | (test_true[:, i] == 1)
        # Use mask to filter out the predictions and true labels for positive cases
        preds_binary.append(preds_binary_temp[pos_mask])
        conf.append(confidence_temp[pos_mask])
        true.append(test_true[:, i][pos_mask])
        
    return preds_binary, true, conf


def get_binary_preds_neg(test_pred, test_true, thresholds_values):
    preds_binary = []
    conf = []
    true = []
    for i in range(test_pred.shape[1]):
        preds_binary_temp = (test_pred[:, i] > thresholds_values[i]).astype(int)
        confidence_temp = np.copy(test_pred[:, i])
        confidence_temp[preds_binary_temp == 0] = 1 - confidence_temp[preds_binary_temp == 0]
        # Create a mask for positive cases
        pos_mask = (preds_binary_temp == 0) | (test_true[:, i] == 0)
        # Use mask to filter out the predictions and true labels for positive cases
        preds_binary.append(preds_binary_temp[pos_mask])
        conf.append(confidence_temp[pos_mask])
        true.append(test_true[:, i][pos_mask])
        
    return preds_binary, true, conf