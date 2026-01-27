import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def load_and_merge_data(file_paths):
    all_data = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            df['data_source'] = path.split('/')[-1]  
            all_data.append(df)
            print(f"成功读取文件 {path}，数据规模：{df.shape}")
        except Exception as e:
            print(f" 读取文件 {path} 失败：{e}")
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后总数据规模：{merged_df.shape}")
    return merged_df
def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print("\n" + "="*60)
    print("模型评估核心指标")
    print("="*60)
    print(f"准确率(Accuracy)：{accuracy:.4f}")
    print(f"精确率(Precision)：{precision:.4f}")
    print(f"召回率(Recall)：{recall:.4f}")
    print(f"F1分数(F1-Score)：{f1:.4f}")
    label_names = {
        0: "正常心跳",
        1: "异常心跳(心梗)",
        2: "室性早搏",
        3: "房性早搏",
        4: "其他异常"
    }
    unique_labels = sorted(y_true.unique())
    target_names = [label_names.get(label, f"标签{label}") for label in unique_labels]
    print("\n" + "="*60)
    print("详细分类报告（按标签）")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0  
    ))
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "="*60)
    print("混淆矩阵（行=真实标签，列=预测标签）")
    print("="*60)
    print(cm)
    n_classes = len(unique_labels)
    fpr: Dict[int, Any] = {}
    tpr: Dict[int, Any] = {}
    roc_auc: Dict[int, Any] = {}
    print("\n" + "="*60)
    print("ROC-AUC 维度检查 & 计算")
    print("="*60)
    print(f"标签类别数：{n_classes}")
    print(f"预测概率数组形状：{y_pred_proba.shape}")
    if n_classes == 1:
        print(" 标签仅含1类，ROC-AUC无实际意义")
        roc_auc = {0: 1.0}
        fpr = {0: np.array([0, 1])}
        tpr = {0: np.array([0, 1])}
    elif n_classes == 2:
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))
        fpr[1], tpr[1], _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc[1] = auc(fpr[1], tpr[1])
        print(f"二分类ROC-AUC（异常心跳）：{roc_auc[1]:.4f}")
    else:
        y_true_bin = label_binarize(y_true, classes=unique_labels)
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
        for i, label in enumerate(unique_labels):
            fpr[label], tpr[label], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])
        roc_auc_avg = np.mean(list(roc_auc.values()))
        print(f"多分类ROC-AUC（平均）：{roc_auc_avg:.4f}")
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm, "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr,
        "target_names": target_names, "n_classes": n_classes
    }
def plot_evaluation_results(eval_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cm = eval_results["confusion_matrix"]
    target_names = eval_results["target_names"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
        xticklabels=target_names, yticklabels=target_names
    )
    ax1.set_title("混淆矩阵", fontsize=14, pad=10)
    ax1.set_xlabel("预测标签", fontsize=12)
    ax1.set_ylabel("真实标签", fontsize=12)
    fpr = eval_results["fpr"]
    tpr = eval_results["tpr"]
    roc_auc = eval_results["roc_auc"]
    n_classes = eval_results["n_classes"]
    
    if n_classes == 1:
        ax2.text(0.5, 0.5, "标签仅含1类\n无有效ROC曲线", 
                 ha="center", va="center", fontsize=12)
        ax2.set_title("ROC曲线", fontsize=14, pad=10)
    else:
        for label_idx in fpr.keys():
            label_name = target_names[label_idx] if label_idx < len(target_names) else f"标签{label_idx}"
            ax2.plot(fpr[label_idx], tpr[label_idx], lw=2, 
                     label=f"{label_name} (AUC={roc_auc[label_idx]:.4f})")
        ax2.plot([0, 1], [0, 1], "k--", lw=2, label="随机猜测")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_title("ROC曲线", fontsize=14, pad=10)
        ax2.set_xlabel("假阳性率(FPR)", fontsize=12)
        ax2.set_ylabel("真阳性率(TPR)", fontsize=12)
        ax2.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("ecg_model_evaluation.png", dpi=300, bbox_inches="tight")
    print("\n 可视化结果已保存为：ecg_model_evaluation.png")
    plt.show()
def llm_baseline_advice(eval_results):
    """基于传统模型评估结果，给出LLM训练的基线建议"""
    print("\n" + "="*60)
    print("LLM训练基线建议（无人工版）")
    print("="*60)
    cm = eval_results["confusion_matrix"]
    target_names = eval_results["target_names"]
    高准确率标签 = []
    for i in range(len(cm)):
        准确率 = cm[i][i] / sum(cm[i]) if sum(cm[i]) > 0 else 0
        if 准确率 > 0.8:
            高准确率标签.append(target_names[i])
    print(f"1. 传统模型高准确率类别（LLM需优先匹配）：{高准确率标签}")
    if eval_results["n_classes"] > 2:
        roc_auc_avg = np.mean(list(eval_results["roc_auc"].values()))
        if roc_auc_avg < 0.8:
            print("2. 提示：传统模型整体AUC偏低，LLM需强化「室性早搏/房性早搏」的区分表述")
        else:
            print("2. 传统模型AUC达标，LLM可复用其标签分类逻辑")
    print("3. LLM输出校验规则：必须包含「正常心跳/异常心跳」等标准标签，禁止自定义分类")
if __name__ == "__main__":
    file_paths = [
    r"C:\Users\liuxi\Desktop\new progress\data2\mitbih_test_annotated.csv",
    r"C:\Users\liuxi\Desktop\new progress\data2\mitbih_train_annotated.csv",
    r"C:\Users\liuxi\Desktop\new progress\data2\ptbdb_abnormal_annotated.csv",
    r"C:C:\Users\liuxi\Desktop\new progress\data2\ptbdb_normal_annotated.csv"
    ]
    df = load_and_merge_data(file_paths)
    print("\n" + "="*60)
    print("异常值检查 & 处理")
    print("="*60)
    na_count = df['original_label'].isna().sum()
    inf_count = np.isinf(df['original_label']).sum()
    print(f"original_label缺失值数量：{na_count}")
    print(f"original_label无穷值数量：{inf_count}")
    df = df[df['original_label'].notna() & ~np.isinf(df['original_label'])]
    print(f"过滤异常值后，数据规模：{df.shape}")
    feature_cols = [f"ecg_feature_{i}" for i in range(1, 188)] 
    label_col = "original_label"
    X = df[feature_cols].fillna(0)  
    y = df[label_col].astype(int) 
    print("\n" + "="*60)
    print("标签实际类别检查")
    print("="*60)
    print(f"original_label的所有类别：{sorted(y.unique())}")
    print(f"类别数量：{len(y.unique())}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )
    print("\n" + "="*60)
    print("训练集/测试集划分结果")
    print("="*60)
    print(f"训练集规模：{X_train.shape}，测试集规模：{X_test.shape}")
    print(f"训练集标签分布：{dict(pd.Series(y_train).value_counts())}")
    print(f"测试集标签分布：{dict(pd.Series(y_test).value_counts())}")
    print("\n" + "="*60)
    print("模型训练中...")
    print("="*60)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) 
    eval_results: Dict[str, Any] = evaluate_model(y_test, y_pred, y_pred_proba)
    print("\n" + "="*60)
    print("生成可视化结果...")
    print("="*60)
    plot_evaluation_results(eval_results)
    print("\n所有评估流程执行完成！")
    plot_evaluation_results(eval_results)
    llm_baseline_advice(eval_results)  
    print("\n所有评估流程执行完成！")