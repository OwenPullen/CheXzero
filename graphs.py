import reliability_diagrams as rd
import numpy as np
import matplotlib.pyplot as plt
from eval import plot_pr, plot_roc
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_recall_curve

def get_reliability_diagrams(test_true, preds_binary, test_pred, cxr_labels, model_label, path, pos_only=False):
    plt.style.use("seaborn")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)
    plt.title(f"Reliability Diagram {model_label} {cxr_labels[0]}")

    dict = {}
    list_gph = []
    for i in range(14):
        dict_i = {
        "true_labels" : test_true[:,i],
        "pred_labels" : preds_binary[:,i],
        "confidences" : test_pred[:,i]
        }
        dict.update({cxr_labels[i]: dict_i})
        gph = rd.reliability_diagram(true_labels=test_true[:,i],
                                    pred_labels=preds_binary[:,i],
                                    confidences=test_pred[:,i],
                                    title=cxr_labels[i],
                                    num_bins=20,
                                    dpi = 500,
                                    return_fig=True,
                                    pos_only=pos_only)
        gph.savefig(f"results/plots/{path}/reliability_diagram_3_{model_label}_{cxr_labels[i]}")
        list_gph.append(gph)
        
    plt.title(f"Reliability Diagram {model_label}")
    mp_gph = rd.reliability_diagrams(dict, num_cols = 7, num_rows = 2, num_bins=20, draw_bin_importance=True, return_fig=True)
    mp_gph.savefig(f'results/plots/multipanel_reliability_diagram_{model_label}.png')

def get_roc_plots(test_true, preds_binary, cxr_labels, path):
    for i in range(preds_binary.shape[1]):
        plot_roc(y_true=test_true[:,i], y_pred=preds_binary[:,i], roc_name=path+'/'+cxr_labels[i], plot=True)
        

def plot_roc_mp(test_true, preds_binary, cxr_labels, model_label, path ,plot=True):
    plt.figure(figsize=(14, 10))
    for i in range(preds_binary.shape[1]):
        fpr, tpr, thresholds = roc_curve(test_true[:, i], preds_binary[:, i])
        roc_auc = auc(fpr, tpr)
        if plot:
            plt.subplot(4, 4, i+1)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title(f'{model_label} {cxr_labels[i]}')
    plt.tight_layout()
    plt.savefig(f'results/plots/{path}/multipanel_roc_curve_{model_label}.png')

def plot_pr_mp(test_true, preds_binary, cxr_labels, model_label, plot=True):
    plt.figure(figsize=(14, 10))
    for i in range(preds_binary.shape[1]):
        precision, recall, thresholds = precision_recall_curve(test_true[:, i], preds_binary[:, i])
        pr_auc = auc(recall, precision)
        if plot:
            plt.subplot(4, 4, i+1)
            plt.plot(recall, precision, 'b', label='AUC = %0.2f' % pr_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{model_label} {cxr_labels[i]}')
    plt.tight_layout()
    plt.savefig(f'results/plots/multipanel_pr_curve_{model_label}.png')

# def get_reliability_diagrams_mp(test_true, preds_binary, test_pred, cxr_labels, model_label):
#     plt.style.use("seaborn")
#     plt.rc("font", size=12)
#     plt.rc("axes", labelsize=12)
#     plt.rc("xtick", labelsize=12)
#     plt.rc("ytick", labelsize=12)
#     plt.rc("legend", fontsize=12)
#     plt.title(f"Reliability Diagram {model_label}")
    
#     dict = {}
#     list_gph = []
#     fig, axs = plt.subplots(2, 7, figsize=(14, 6))
#     for i in range(14):
#         dict_i = {
#         "true_labels" : test_true[:,i],
#         "pred_labels" : preds_binary[:,i],
#         "confidences" : test_pred[:,i]
#         }
#         dict.update({cxr_labels[i]: dict_i})
#         gph = rd.reliability_diagram(true_labels=test_true[:,i],
#                         pred_labels=preds_binary[:,i],
#                         confidences=test_pred[:,i],
#                         title=cxr_labels[i],
#                         num_bins=20,
#                         dpi = 500,
#                         return_fig=True)
#         axs[i//7, i%7].plot(gph.get_paths()[0].vertices[:, 0], gph.get_paths()[0].vertices[:, 1], 'b', label='Reliability Diagram')
#         axs[i//7, i%7].plot(gph.get_paths()[1].vertices[:, 0], gph.get_paths()[1].vertices[:, 1], 'r--', label='Perfect Calibration')
#         axs[i//7, i%7].set_xlim([0, 1])
#         axs[i//7, i%7].set_ylim([0, 1])
#         axs[i//7, i%7].set_xlabel('Confidence')
#         axs[i//7, i%7].set_ylabel('Accuracy')
#         axs[i//7, i%7].set_title(f'{model_label} {cxr_labels[i]}')
#         axs[i//7, i%7].legend(loc='lower right')

#         list_gph.append(gph)
        
#     plt.tight_layout()
#     plt.savefig(f'results/plots/multipanel_reliability_diagram3_{model_label}.png')

def get_reliability_diagrams_mp(test_true, preds_binary, test_pred, cxr_labels, model_label):
    plt.style.use("seaborn")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)
    
    fig, axs = plt.subplots(2, 7, figsize=(14, 6))
    
    for i in range(14):
        ax = axs[i//7, i%7]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_label} {cxr_labels[i]}')

        # Generate the reliability diagram and extract the figure
        gph = rd.reliability_diagram(
            true_labels=test_true[:, i],
            pred_labels=preds_binary[:, i],
            confidences=test_pred[:, i],
            title=cxr_labels[i],
            num_bins=20,
            dpi=500,
            return_fig=True
        )
        
        # Extract lines from the generated figure
        lines = gph.gca().get_lines()
        for line in lines:
            ax.plot(line.get_xdata(), line.get_ydata(), line.get_linestyle(), label=line.get_label(), color=line.get_color())
        
        ax.legend(loc='lower right')
        plt.close(gph)  # Close the figure to free up memory

    plt.tight_layout()
    plt.suptitle(f"Reliability Diagram {model_label}", y=1.02)
    plt.savefig(f'results/plots/multipanel_reliability_diagram3_{model_label}.png')
    plt.show()  # Display the final plot

# Example call to the function
# get_reliability_diagrams_mp(test_true, preds_binary, test_pred, cxr_labels, model_label)


# Example call to the function
# get_reliability_diagrams_mp(test_true, preds_binary, test_pred, cxr_labels, model_label)

# Example call to the function
# get_reliability_diagrams_mp(test_true, preds_binary, test_pred, cxr_labels, model_label)
