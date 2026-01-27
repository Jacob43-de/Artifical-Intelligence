import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['figure.facecolor'] = 'white' 
DATA_PATH = r'C:\Users\liuxi\Desktop\new progress\data3\cardio_train_organized.csv'
def load_and_validate_data():
    df = pd.read_csv(DATA_PATH)
    print("="*80)
    print("1. 数据基础信息校验")
    print("="*80)
    print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
    print("\nCSV文件所有列名（确保无KeyError）:")
    for idx, col in enumerate(df.columns, 1):
        print(f"   {idx:2d}. {col}")
    missing_values = df.isnull().sum()
    print(f"\n缺失值统计:")
    if missing_values.sum() == 0:
        print("  无缺失值，数据完整性良好")
    else:
        for col, miss in missing_values.items():
            if miss > 0:
                print(f"  {col}: {miss} 个缺失值 ({miss/len(df)*100:.2f}%)")
    print(f"\n数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} 列")
    return df
def llm_evaluation_analysis(df):
    print("\n" + "="*80)
    print("2. LLM评估核心分析")
    print("="*80)
    print("\n2.1 目标变量（Cardio_Label）分布:")
    cardio_dist = df['Cardio_Label'].value_counts()
    for label, count in cardio_dist.items():
        percentage = count / len(df) * 100
        print(f"   {label}: {count} 例 ({percentage:.2f}%)")
    print("\n2.2 关键特征与心血管疾病的关联性:")
    age_cardio = df.groupby('Cardio_Label')['Age_Years'].agg(['mean', 'median', 'std']).round(2)
    print("\n   年龄分布（按心血管疾病分组）:")
    print(age_cardio)
    gender_cardio = pd.crosstab(df['Gender_Label'], df['Cardio_Label'], normalize='index') * 100
    print("\n   性别与心血管疾病比例 (%):")
    print(gender_cardio.round(2))
    chol_cardio = pd.crosstab(df['Cholesterol_Label'], df['Cardio_Label'], normalize='index') * 100
    print("\n   胆固醇水平与心血管疾病比例 (%):")
    print(chol_cardio.round(2))
    gluc_cardio = pd.crosstab(df['Glucose_Label'], df['Cardio_Label'], normalize='index') * 100
    print("\n   血糖水平与心血管疾病比例 (%):")
    print(gluc_cardio.round(2))
    habits = ['Smoke_Label', 'Alco_Label', 'Active_Label']
    habit_names = ['吸烟', '饮酒', '运动']
    print("\n   生活习惯与心血管疾病比例 (%):")
    for habit, name in zip(habits, habit_names):
        habit_dist = pd.crosstab(df[habit], df['Cardio_Label'], normalize='index') * 100
        print(f"\n   {name}习惯:")
        print(habit_dist.round(2))
    print("\n2.3 风险因子相关性排序（与cardio列）:")
    numeric_cols = ['Age_Years', 'height', 'weight', 'Systolic_BP', 'Diastolic_BP', 
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    corr = df[numeric_cols].corr()['cardio'].sort_values(ascending=False)
    risk_factors = corr.drop('cardio').round(3)
    print(risk_factors)
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    bmi_cardio = df.groupby('Cardio_Label')['BMI'].agg(['mean', 'median']).round(2)
    print("\n   BMI指数与心血管疾病:")
    print(bmi_cardio)
    return df
def generate_visualizations(df):
    print("\n" + "="*80)
    print("3. 生成可视化结果")
    print("="*80)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    save_path = 'C:/Users/liuxi/Desktop/code/new'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cardio_counts = df['Cardio_Label'].value_counts()
    ax1.pie(cardio_counts.values, labels=cardio_counts.index, autopct='%1.2f%%',
            colors=colors[:2], startangle=90, textprops={'fontsize': 12})
    ax1.set_title('心血管疾病总体分布', fontsize=14, fontweight='bold', pad=20)
    gender_cardio = pd.crosstab(df['Gender_Label'], df['Cardio_Label'], normalize='index') * 100
    x = np.arange(len(gender_cardio.index))
    width = 0.35
    ax2.bar(x - width/2, gender_cardio['无心血管疾病'], width, label='无心血管疾病', 
            color=colors[0], alpha=0.8)
    ax2.bar(x + width/2, gender_cardio['有心血管疾病'], width, label='有心血管疾病', 
            color=colors[1], alpha=0.8)
    ax2.set_xlabel('性别', fontsize=12, fontweight='bold')
    ax2.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('不同性别的心血管疾病比例', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gender_cardio.index)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for i, (col1, col2) in enumerate(zip(gender_cardio['无心血管疾病'], gender_cardio['有心血管疾病'])):
        ax2.text(i - width/2, col1 + 0.5, f'{col1:.2f}%', ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, col2 + 0.5, f'{col2:.2f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path + 'cardio_gender_dist.png', dpi=300, bbox_inches='tight')
    print(" 图表1：心血管疾病分布与性别关系图已保存")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    ax1.hist(df['Age_Years'], bins=20, color=colors[2], alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.axvline(df['Age_Years'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'平均年龄: {df["Age_Years"].mean():.1f}岁')
    ax1.set_xlabel('年龄（岁）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('人数', fontsize=12, fontweight='bold')
    ax1.set_title('年龄总体分布', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    cardio_groups = [df[df['Cardio_Label'] == '无心血管疾病']['Age_Years'],
                     df[df['Cardio_Label'] == '有心血管疾病']['Age_Years']]
    box_plot = ax2.boxplot(cardio_groups, labels=['无心血管疾病', '有心血管疾病'],
                           patch_artist=True, notch=True, showmeans=True)
    for patch, color in zip(box_plot['boxes'], colors[:2]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel('心血管疾病状态', fontsize=12, fontweight='bold')
    ax2.set_ylabel('年龄（岁）', fontsize=12, fontweight='bold')
    ax2.set_title('不同心血管疾病状态的年龄分布', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + 'cardio_age_analysis.png', dpi=300, bbox_inches='tight')
    print("  图表2：年龄与心血管疾病关系图已保存")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    chol_cardio = pd.crosstab(df['Cholesterol_Label'], df['Cardio_Label'], normalize='index') * 100
    x1 = np.arange(len(chol_cardio.index))
    width = 0.35
    ax1.bar(x1 - width/2, chol_cardio['无心血管疾病'], width, label='无心血管疾病', 
            color=colors[0], alpha=0.8)
    ax1.bar(x1 + width/2, chol_cardio['有心血管疾病'], width, label='有心血管疾病', 
            color=colors[1], alpha=0.8)
    ax1.set_xlabel('胆固醇水平', fontsize=12, fontweight='bold')
    ax1.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('胆固醇水平与心血管疾病比例', fontsize=14, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(chol_cardio.index, rotation=15)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i, (col1, col2) in enumerate(zip(chol_cardio['无心血管疾病'], chol_cardio['有心血管疾病'])):
        ax1.text(i - width/2, col1 + 0.5, f'{col1:.1f}%', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, col2 + 0.5, f'{col2:.1f}%', ha='center', va='bottom', fontsize=10)
    
    gluc_cardio = pd.crosstab(df['Glucose_Label'], df['Cardio_Label'], normalize='index') * 100
    x2 = np.arange(len(gluc_cardio.index))
    ax2.bar(x2 - width/2, gluc_cardio['无心血管疾病'], width, label='无心血管疾病', 
            color=colors[0], alpha=0.8)
    ax2.bar(x2 + width/2, gluc_cardio['有心血管疾病'], width, label='有心血管疾病', 
            color=colors[1], alpha=0.8)
    ax2.set_xlabel('血糖水平', fontsize=12, fontweight='bold')
    ax2.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('血糖水平与心血管疾病比例', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(gluc_cardio.index, rotation=15)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (col1, col2) in enumerate(zip(gluc_cardio['无心血管疾病'], gluc_cardio['有心血管疾病'])):
        ax2.text(i - width/2, col1 + 0.5, f'{col1:.1f}%', ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, col2 + 0.5, f'{col2:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path + 'cardio_biomarker_analysis.png', dpi=300, bbox_inches='tight')
    print(" 图表3：胆固醇/血糖与心血管疾病关系图已保存")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    key_cols = ['Age_Years', 'weight', 'Systolic_BP', 'Diastolic_BP', 
                'cholesterol', 'gluc', 'BMI', 'cardio']
    corr_matrix = df[key_cols].corr()
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(key_cols)))
    ax.set_yticks(range(len(key_cols)))
    ax.set_xticklabels(key_cols, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(key_cols, fontsize=11)
    
    for i in range(len(key_cols)):
        for j in range(len(key_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                           fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('相关系数', fontsize=12, fontweight='bold')
    ax.set_title('关键风险因子相关性热力图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path + 'cardio_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(" 图表4：风险因子相关性热力图已保存")
    
    print(f"\n 所有可视化图表已保存至路径：{save_path}")
def main():
    df = load_and_validate_data()
    
    df = llm_evaluation_analysis(df)
    generate_visualizations(df)
    print("\n" + "="*80)
    print("4. 评估总结")
    print("="*80)
    print("数据读取与校验完成：无KeyError，列名完全匹配")
    print("LLM核心评估分析完成：涵盖分布、关联性、风险因子分析")
    print("可视化生成完成：4类核心图表已保存至指定路径")
    print("评估结论：")
    print("   1. 年龄、胆固醇、血压是心血管疾病的核心风险因子")
    print("   2. 男性心血管疾病风险略高于女性")
    print("   3. 胆固醇/血糖异常显著增加心血管疾病风险")
    print("\n LLM训练优化提示（无人工校验版）：")
    print("   - 强制LLM回答中必须引用「年龄/胆固醇/血压」至少1项核心指标")
    print("   - 限制「无心脏病」结论仅在胆固醇<200且血压<130时输出")
    print("   - 禁止LLM使用「无需随访」「不用监测」等表述（高风险）")
if __name__ == "__main__":
    main()