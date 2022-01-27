import numpy as np
from sklearn.metrics import accuracy_score


# Function to get the positions of the repair variables correctly
def cardinal(y, preds):
    y_cardinal = np.zeros(shape=y.shape[:2])
    
    for i in range(len(y_cardinal)):
        loc = np.argmax(y[i, 0])
        rep = np.argmax(preds[i, 1]) if np.argmax(preds[i, 1]) in np.argwhere(y[i, 1]) else np.argmax(y[i, 1])
        
        y_i = np.array([loc, rep])
        
        y_cardinal[i] = y_i
        
    return y_cardinal


# Calculate the metrics used in (Marko Vasic, 2019) to compare different model's performance
def model_evaluation(y_cardinal, pred_cardinal):
    # True Positive, the percentage of the bug-free programs in the ground truth classified as bug free
    non_bug = y_cardinal[:, 0] == 0
    true_positive = np.round(accuracy_score(y_cardinal[non_bug][:, 0], pred_cardinal[non_bug][:, 0]), 5)
    
    # Classification Accuracy, the percentage of total programs in the test set classified correctly as either bug free or buggy
    buggy = y_cardinal[:, 0] != 0
    
    correct_non_bug = sum(y_cardinal[:, 0][non_bug] == pred_cardinal[:, 0][non_bug])
    correct_bug = sum(pred_cardinal[:, 0][buggy] != 0)
    
    class_accuracy = np.round((correct_non_bug + correct_bug)/len(y_cardinal), 5)
    
    # Localization Accuracy, the percentage of buggy programs for which the bug location is correctly predicted by the model    
    correct_loc_bug = pred_cardinal[:, 0][buggy] == y_cardinal[:, 0][buggy]
    loc_accuracy = np.round(sum(correct_loc_bug)/sum(buggy), 5)
    
    # Localization+Repair Accuracy, the percentage of buggy programs for which both the location and repair are correctly predicted by the model
    correct_loc_and_rep_bug = pred_cardinal[:, 1][buggy][correct_loc_bug] == y_cardinal[:, 1][buggy][correct_loc_bug]
    loc_rep_accuracy = np.round(sum(correct_loc_and_rep_bug)/sum(buggy), 5)
    
    return true_positive, class_accuracy, loc_accuracy, loc_rep_accuracy