from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import itertools


def plot_model_results(y_cardinal, pred_cardinal, normalize=False):
    non_buggy = y_cardinal[:, 0] == 0
    buggy = y_cardinal[:, 0] != 0
        
    # Results for non-buggy programs
    correct_loc_nonbuggy = sum(pred_cardinal[:, 0][non_buggy] == y_cardinal[:, 0][non_buggy])
    inc_loc_nonbuggy = sum(pred_cardinal[:, 0][non_buggy] != y_cardinal[:, 0][non_buggy])
    
    correct_rep_nonbuggy = sum(pred_cardinal[:, 1][non_buggy] == y_cardinal[:, 1][non_buggy])
    inc_rep_nonbuggy = sum(pred_cardinal[:, 1][non_buggy] != y_cardinal[:, 1][non_buggy])
    
    # Results for buggy programs
    correct_loc_buggy = sum(pred_cardinal[:, 0][buggy] == y_cardinal[:, 0][buggy])
    inc_loc_buggy = sum(pred_cardinal[:, 0][buggy] != y_cardinal[:, 0][buggy])
    
    correct_rep_buggy = sum(pred_cardinal[:, 1][buggy] == y_cardinal[:, 1][buggy])
    inc_rep_buggy = sum(pred_cardinal[:, 1][buggy] != y_cardinal[:, 1][buggy])        
    
    # Join all results
    loc_results = np.array([[correct_loc_nonbuggy, inc_loc_nonbuggy], [correct_loc_buggy, inc_loc_buggy]])
    rep_results = np.array([[correct_rep_nonbuggy, inc_rep_nonbuggy], [correct_rep_buggy, inc_rep_buggy]])
    
    fmt = 'd'
    
    if normalize:
        loc_results = loc_results.astype('float') / loc_results.sum(axis=1)[:, np.newaxis]
        rep_results = rep_results.astype('float') / rep_results.sum(axis=1)[:, np.newaxis]
        
        fmt = '.2'
    
    # Plot results in a confusion matrix
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,7))
    fig.tight_layout(w_pad=12)
    
    ticks = ['Non-buggy', 'Buggy']
    ticks2 = ['Good prediction', 'Bad prediction']
    
    loc_plot = sns.heatmap(loc_results, annot=True, annot_kws={'fontsize':15}, fmt=fmt, xticklabels=ticks2, yticklabels=ticks, cbar=False, 
                           cmap='Blues', linewidths=.5, ax=ax[0])
    rep_plot = sns.heatmap(rep_results, annot=True, annot_kws={'fontsize':15}, fmt=fmt, xticklabels=ticks2, yticklabels=ticks, cbar=False, 
                           cmap='Blues', linewidths=.5, ax=ax[1])
    
    loc_plot.set_yticklabels(labels=ticks, rotation=360, fontsize=12)
    rep_plot.set_yticklabels(labels=ticks, rotation=360, fontsize=12)
    loc_plot.set_xticklabels(labels=ticks2, fontsize=12)
    rep_plot.set_xticklabels(labels=ticks2, fontsize=12)
    #loc_plot.set_xlabel('Predicted label', fontsize=14)
    #rep_plot.set_xlabel('Predicted label', fontsize=14)
    #loc_plot.set_ylabel('True label', fontsize=14)
    #rep_plot.set_ylabel('True label', fontsize=14)
    loc_plot.set_title('Results for location', fontdict={'fontsize':18})
    rep_plot.set_title('Results for repair', fontdict={'fontsize':18})
    
    plt.show()
    
    
    
def plot_confusion_matrix(y_cardinal, pred_cardinal, normalize=False):
    non_buggy = y_cardinal[:, 0] == 0
    buggy = y_cardinal[:, 0] != 0
    
    # Results for non-buggy programs
    correct_nonbuggy = sum(pred_cardinal[:, 0][non_buggy] == 0)
    inc_nonbuggy = sum(pred_cardinal[:, 0][non_buggy] != 0)
    
    # Results for buggy programs
    correct_buggy = sum(pred_cardinal[:, 0][buggy] != 0)
    inc_buggy = sum(pred_cardinal[:, 0][buggy] == 0)
    
    # Join all results
    loc_results = np.array([[correct_nonbuggy, inc_nonbuggy], [inc_buggy, correct_buggy]])
    
    fmt = 'd'
    
    if normalize:
        loc_results = loc_results.astype('float') / loc_results.sum(axis=1)[:, np.newaxis]        
        fmt = '.2'
        
    # Plot results in a confusion matrix
    plt.figure(figsize=(8, 8))
    
    ticks = ['Non-buggy', 'Buggy']
    
    confusion_matrix = sns.heatmap(loc_results, annot=True, annot_kws={'fontsize':15}, fmt=fmt, xticklabels=ticks, yticklabels=ticks, cbar=False, 
                                   cmap='Blues', linewidths=.5)
    
    confusion_matrix.set_yticklabels(labels=ticks, rotation=360, fontsize=12)
    confusion_matrix.set_xticklabels(labels=ticks, fontsize=12)
    confusion_matrix.set_xlabel('Predicted label', fontsize=14)
    confusion_matrix.set_ylabel('True label', fontsize=14)
    confusion_matrix.set_title('Confusion matrix', fontdict={'fontsize':18})
    
    plt.show()