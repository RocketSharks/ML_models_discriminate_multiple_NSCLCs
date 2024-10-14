'''
Author: lx
Date: 2024-03-21 10:10:22
LastEditors: lx
LastEditTime: 2024-10-14 17:05:04
Description: 
FilePath: /ML_models_discriminate_multiple_NSCLCs/RF.py
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,  precision_score, recall_score, matthews_corrcoef, confusion_matrix, roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

def load_data(file_path, feature_cols, label_col):   
    """
    读取CSV文件并提取特征和标签列

    参数:
    file_path (str): CSV文件路径
    feature_cols (list): 特征列名列表
    label_col (str): 标签列名

    返回:
    tuple: 特征 (data_x) 和 标签 (data_y)
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)
    # 提取特征和标签
    data_x = data[feature_cols]
    data_y = data[label_col]
    return data_x, data_y

def load_data_split(file_path, feature_cols, label_col,test_size=0.2, random_state=42):   
    """
    读取CSV文件并提取特征和标签列

    参数:
    file_path (str): CSV文件路径
    feature_cols (list): 特征列名列表
    label_col (str): 标签列名

    返回:
    tuple: 特征 (data_x) 和 标签 (data_y)
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 提取特征和标签
    data_x = data[feature_cols]
    data_y = data[label_col]
        # 进行分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=test_size, stratify=data_y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def print_label_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    total_counts = sum(counts)
    label_distribution = dict(zip(unique, counts))
    print(f"Labels counts in {name}:")
    for label, count in label_distribution.items():
        ratio = count / total_counts
        print(f"  {label}: {count} ({ratio:.2f})")

def calculate_metrics(predicted_probs, true_labels, pos_label):
    """
    计算AUC、ACC、F1等指标

    参数:
    predicted_probs (array-like): 预测结果的概率
    true_labels (array-like): 真实标签
    pos_label (str): 正类标签

    返回:
    dict: 包含AUC、ACC、F1等指标的字典
    """
    # 将真实标签转化为二进制格式
    true_labels_binary = np.array([1 if label == pos_label else 0 for label in true_labels])

    # 提取正类概率
    pos_probs = np.array([prob[1] for prob in predicted_probs])

    # 将概率转换为预测标签
    predicted_labels_binary = np.array([1 if prob > 0.5 else 0 for prob in pos_probs])

    # 计算AUC
    auc = roc_auc_score(true_labels_binary, pos_probs)
    acc = accuracy_score(true_labels_binary, predicted_labels_binary)
    f1 = f1_score(true_labels_binary, predicted_labels_binary, pos_label=0, average='binary')
    precision = precision_score(true_labels_binary, predicted_labels_binary, pos_label=0, average='binary')
    recall = recall_score(true_labels_binary, predicted_labels_binary, pos_label=0, average='binary')
    mcc = matthews_corrcoef(true_labels_binary, predicted_labels_binary)
    cm = confusion_matrix(true_labels_binary, predicted_labels_binary, labels=[1, 0])

    return {
        'AUC': auc,
        'Accuracy': acc,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Precision': precision,
        'Recall': recall,
        'MCC': mcc
    }        
     


if __name__ == '__main__':

    val_results_df = pd.DataFrame(columns=['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC','Pre_pro',"Patient_ID"])
    test_results_df = pd.DataFrame(columns=['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC','Pre_pro','True_label'])
    random_seed_ = 66
    train_file_path = './train_data.csv'
    test_file_path = './test_data.csv'
    feature_cols = ['Cor', 'Com_num', 'trunk_num', 'Entropy']
    label_col = 'Outcome'

    train_x, val_x, train_y, val_y = load_data_split(train_file_path, feature_cols, label_col, test_size=0.2, random_state=random_seed_)
    train_x, train_y = load_data(test_file_path, feature_cols, label_col)
    test_x, test_y = load_data(test_file_path, feature_cols, label_col)
    
    rfc = RandomForestClassifier(random_state=random_seed_)
    # 定义超参数超参数空间
    param_grid = {
        "n_estimators": np.arange(1, 4, step=1),
        "max_features":  np.arange(1, 4, step=1),
        "max_depth": list(np.arange(1, 4, step=1)) + [None],
        "min_samples_split": np.arange(1, 4, step=1),
        "min_samples_leaf":  np.arange(1, 4, step=1),
        "bootstrap": [True, False],
    }
    cv = StratifiedKFold(n_splits=5)
    GS = GridSearchCV(rfc, param_grid, cv=cv, n_jobs=-1, scoring='roc_auc_ovr', return_train_score=True,refit=True)  
    GS.fit(train_x, train_y)
    print("网格搜索的最佳结果是：",GS.best_params_,GS.best_score_)
    
    val_pre_pro = GS.predict_proba(val_x)
    val_metrics = calculate_metrics(predicted_probs=val_pre_pro, true_labels=val_y, pos_label="MP")
    # print("验证集的结果：",val_metrics)
    # 保存对应的模型权重等信息
    joblib.dump(GS.best_estimator_, f"RF.pkl")
    result_row = pd.DataFrame({
    'AUC': [val_metrics['AUC']],
    'Accuracy': [val_metrics['Accuracy']],
    'F1 Score': [val_metrics['F1 Score']],
    'Precision': [val_metrics['Precision']],
    'Recall': [val_metrics['Recall']],
    'MCC': [val_metrics['MCC']],
    'Pre_pro': [val_pre_pro],
    'Patient_ID': [val_y]
    })
    val_results_df = pd.concat([val_results_df, result_row], ignore_index=True)
    
    test_pre_pro = GS.predict_proba(test_x)
    test_metrics = calculate_metrics(predicted_probs=test_pre_pro, true_labels=test_y, pos_label="MP")
    
    result_row = pd.DataFrame({
    'AUC': [test_metrics['AUC']],
    'Accuracy': [test_metrics['Accuracy']],
    'F1 Score': [test_metrics['F1 Score']],
    'Precision': [test_metrics['Precision']],
    'Recall': [test_metrics['Recall']],
    'MCC': [test_metrics['MCC']],
    'Pre_pro': [test_pre_pro],
    'True_label': [test_y]
    })
    test_results_df = pd.concat([test_results_df, result_row], ignore_index=True)
    
    # 保存结果
    val_results_df.to_csv('./RF_val_results.csv', index=False)
    test_results_df.to_csv('./RF_test_results.csv', index=False)

