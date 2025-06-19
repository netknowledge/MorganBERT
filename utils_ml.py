from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score


def compute_metrics_reg(y_true, y_pred):
    """"""
    return {
        'eval_mse': mean_squared_error(y_true, y_pred), 
        'eval_mae': mean_absolute_error(y_true, y_pred), 
        'eval_r2':  r2_score(y_true, y_pred), 
        'eval_rmse':root_mean_squared_error(y_true, y_pred), 
        'eval_pcc': stats.pearsonr(y_pred, y_true)[0]
    }

def compute_metrics_cls(y_true, y_score, y_pred):
    """"""
    return {
        'eval_auc': roc_auc_score(y_true, y_score), 
        'eval_prec': precision_score(y_true, y_pred), 
        'eval_recall': recall_score(y_true, y_pred), 
        'eval_f1': f1_score(y_true, y_pred), 
        'eval_accuracy': accuracy_score(y_true, y_pred), 
        'eval_balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
