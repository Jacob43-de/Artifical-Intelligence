import pandas as pd
import re
import logging
from typing import Dict, List, Any, Optional
import os
import sys
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_diagnosis_output.csv")
try:
    df = pd.read_csv(test_path, encoding="utf-8-sig")
    print(f"文件有效，列名：{df.columns.tolist()}，行数：{len(df)}")
except Exception as e:
    print(f"文件无效：{str(e)}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 所有文件路径（直接指向data1文件夹内的CSV）
MY_CSV_PATH = os.path.join(CURRENT_DIR, "标注好的完整数据文件.csv")
LLM_OUTPUT_PATH = os.path.join(CURRENT_DIR, "llm_diagnosis_output.csv")
ASSESSMENT_REPORT_PATH = os.path.join(CURRENT_DIR, "LLM诊断评估报告.csv")
OPTIMIZATION_SUGGESTIONS_PATH = os.path.join(CURRENT_DIR, "LLM训练优化建议.csv")

# 打印路径供核对（运行时可查看是否匹配）
logger.info(f"当前脚本所在路径（data1）：{CURRENT_DIR}")
logger.info(f"标注数据路径：{MY_CSV_PATH}")
logger.info(f"LLM输出文件路径：{LLM_OUTPUT_PATH}")

COLUMN_MAPPING = {
    "年龄": "年龄",
    "静息血压": "静息血压(mm Hg)",
    "血清胆固醇": "血清胆固醇(mg/dl)",
    "胸痛类型": "胸痛类型",
    "ST段压低": "ST段压低(mm)",
    "运动诱发心绞痛": "运动诱发心绞痛",
    "评估目标列": "心脏病诊断结果"
}

GUIDELINE_RULES = {
    "核心指标": {
        "正常区间": {
            "静息血压": {"低限": 0, "高限": 139},    
            "血清胆固醇": {"低限": 0, "高限": 239}   
        },
        "临界区间": {
            "静息血压": {"低限": 130, "高限": 139},  
            "血清胆固醇": {"低限": 200, "高限": 239} 
        }
    },
    "非核心指标": {
        "胸痛类型": [2, 3],        
        "ST段压低": {"高限": 2.0}, 
        "运动诱发心绞痛": 0       
    },
    "诊断映射": {
        "无心脏病": {
            "核心正常": 5,  
            "核心临界": 4,  
            "核心正常+非核心异常": 3  
        },
        "患病类": {
            "核心异常": 5, 
            "核心正常+非核心异常": 3 
        }
    }
}

STANDARD_TERMS = {
    "诊断结果": ["无心脏病", "未患病", "冠心病", "高血压", "心梗", "心血管疾病", "患病"],
    "指标术语": ["静息血压", "血清胆固醇", "ST段压低", "运动诱发心绞痛", "胸痛类型"],
    "临床表述": ["无异常", "正常", "临界异常", "轻微异常"]
}

SAFE_RULES = [
    {"匹配规则": re.compile(r"无需处理|不用监测|无风险|无需随访"), "风险等级": "中", "原因": "诊断后随访建议缺失"},
    {"匹配规则": re.compile(r"确诊.*无异常|无心脏病.*异常|正常.*患病"), "风险等级": "高", "原因": "诊断结果逻辑矛盾"},
    {"匹配规则": re.compile(r"轻微异常.*需紧急处理"), "风险等级": "高", "原因": "处置建议过度"}
]

def parse_llm_output(llm_text: str) -> Dict[str, Any]:
    """从LLM的自然语言回答中提取指标值和诊断结果"""
    extraction_result = {
        "静息血压": None,
        "血清胆固醇": None,
        "胸痛类型": 3,  
        "ST段压低": 1.0,  
        "运动诱发心绞痛": 0,  
        "诊断结果": ""
    }
    
    # 正则提取血压（匹配：血压120、静息血压130mm Hg、血压值140等）
    bp_pattern = re.compile(r"(静息血压|血压)\D*(\d{2,3})")
    bp_match = bp_pattern.search(llm_text)
    if bp_match:
        try:
            extraction_result["静息血压"] = int(bp_match.group(2))
        except ValueError:
            logger.warning(f"无法将血压值转换为整数: {bp_match.group(2)}")
    
    # 正则提取胆固醇（匹配：胆固醇200、血清胆固醇240mg/dl等）
    cholesterol_pattern = re.compile(r"(血清胆固醇|胆固醇)\D*(\d{2,3})")
    cholesterol_match = cholesterol_pattern.search(llm_text)
    if cholesterol_match:
        try:
            extraction_result["血清胆固醇"] = int(cholesterol_match.group(2))
        except ValueError:
            logger.warning(f"无法将胆固醇值转换为整数: {cholesterol_match.group(2)}")
    
    # 提取诊断结论
    for diagnosis_term in STANDARD_TERMS["诊断结果"]:
        if diagnosis_term in llm_text:
            extraction_result["诊断结果"] = diagnosis_term
            break
    
    return extraction_result

def determine_index_interval(index_value: float, interval_rules: Dict[str, int]) -> str:
    if interval_rules["低限"] <= index_value <= interval_rules["高限"]:
        return "正常" if interval_rules["低限"] == 0 else "临界"
    return "异常"

def evaluate_index_diagnosis_match(patient_index: Dict[str, Any], diagnosis_result: str) -> int:
    bp_value = patient_index["静息血压"] if patient_index["静息血压"] is not None else 120
    cholesterol_value = patient_index["血清胆固醇"] if patient_index["血清胆固醇"] is not None else 200
    
    # 判断血压和胆固醇区间
    bp_interval = determine_index_interval(
        bp_value, GUIDELINE_RULES["核心指标"]["正常区间"]["静息血压"]
    )
    cholesterol_interval = determine_index_interval(
        cholesterol_value, GUIDELINE_RULES["核心指标"]["正常区间"]["血清胆固醇"]
    )
    
    # 确定核心指标状态
    if bp_interval == "正常" and cholesterol_interval == "正常":
        core_status = "正常"
    elif bp_interval == "临界" or cholesterol_interval == "临界":
        core_status = "临界"
    else:
        core_status = "异常"
    
    # 计算非核心指标异常数
    non_core_abnormal_count = 0
    if patient_index["胸痛类型"] not in GUIDELINE_RULES["非核心指标"]["胸痛类型"]:
        non_core_abnormal_count += 1
    if patient_index["ST段压低"] >= GUIDELINE_RULES["非核心指标"]["ST段压低"]["高限"]:
        non_core_abnormal_count += 1
    if patient_index["运动诱发心绞痛"] != GUIDELINE_RULES["非核心指标"]["运动诱发心绞痛"]:
        non_core_abnormal_count += 1
    
    non_core_abnormal = non_core_abnormal_count > 0
    
    # 评估匹配度
    if "无心脏病" in diagnosis_result or "未患病" in diagnosis_result:
        if core_status == "正常" and not non_core_abnormal:
            return 5  
        elif core_status == "正常" and non_core_abnormal:
            return 3  
        elif core_status == "临界":
            return 4 
        else:
            return 2  
    elif any(term in diagnosis_result for term in ["患病", "冠心病", "高血压", "心梗"]):
        return 5 if core_status == "异常" else 3
    else:
        return 3

def evaluate_diagnosis_terminology(diagnosis_result: str) -> int:
    words = [word.strip() for word in diagnosis_result.split() if word.strip()]
    standard_terms = [term for sublist in STANDARD_TERMS.values() for term in sublist]
    non_standard_count = 0
    for word in words:
        if not any(standard_term in word or word in standard_term for standard_term in standard_terms):
            non_standard_count += 1
    
    # 确定规范得分
    if non_standard_count == 0:
        return 5
    elif non_standard_count <= 1:
        return 4
    else:
        return 3

def detect_diagnosis_risk(diagnosis_result: str) -> Dict[str, Any]:
    risk_list = []
    for rule in SAFE_RULES:
        match_result = rule["匹配规则"].search(diagnosis_result)
        if match_result:
            risk_list.append({
                "风险类型": rule["原因"],
                "风险等级": rule["风险等级"],
                "匹配内容": match_result.group()
            })
    return {"存在风险": len(risk_list) > 0, "风险详情": risk_list}

def read_and_clean_data(file_path: str) -> pd.DataFrame:
    """读取并清洗数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"标注数据文件不存在：{file_path}\n当前文件夹（data1）下的文件：{os.listdir(CURRENT_DIR)}"
        )
    
    try:
        for encoding in ["utf-8-sig", "gbk", "utf-16"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("无法解析文件编码，请将CSV文件另存为「UTF-8编码」")
        
        required_columns = list(COLUMN_MAPPING.values())
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV缺少必要列：{missing_columns}\n需包含：年龄、静息血压(mm Hg)、心脏病诊断结果等")
        
        df_clean = df.dropna(subset=["年龄", "静息血压(mm Hg)", "心脏病诊断结果"])
        df_clean = df_clean.fillna({
            "血清胆固醇(mg/dl)": 200,  
            "胸痛类型": 3,             
            "ST段压低(mm)": 1.0,       
            "运动诱发心绞痛": 0         
        })
        
        logger.info(f"成功读取标注数据，有效样本数：{len(df_clean)}")
        return df_clean
    
    except Exception as e:
        logger.error(f"数据读取失败：{str(e)}")
        raise

def evaluate_llm_output() -> None:
    """LLM评估主函数（核心逻辑）"""
    try:
        if not os.path.exists(LLM_OUTPUT_PATH):
            raise FileNotFoundError(
                f"LLM输出文件不存在：{LLM_OUTPUT_PATH}\n当前文件夹（data1）下的文件：{os.listdir(CURRENT_DIR)}"
            )
        
        annotated_data = read_and_clean_data(MY_CSV_PATH)
        
        llm_output = None
        for encoding in ["utf-8-sig", "gbk", "utf-16"]:
            try:
                llm_output = pd.read_csv(LLM_OUTPUT_PATH, encoding=encoding)
                logger.info(f"✅ 成功读取LLM输出文件，编码：{encoding}")
                break
            except UnicodeDecodeError:
                continue
        if llm_output is None:
            raise UnicodeDecodeError("❌ LLM输出文件编码不支持，建议另存为UTF-8格式")
        
        # 检查LLM输出的必要列
        required_llm_columns = ["样本序号", "LLM诊断文本"]
        missing_llm_cols = [col for col in required_llm_columns if col not in llm_output.columns]
        if missing_llm_cols:
            raise ValueError(f"❌ LLM输出文件缺少必要列：{missing_llm_cols}\n必须包含：样本序号、LLM诊断文本")
        
        assessment_results = []
        total_match_score = 0
        total_norm_score = 0
        risk_sample_count = 0
        low_score_samples = []
        
        for _, llm_row in llm_output.iterrows():
            sample_id = llm_row["样本序号"]
            llm_text = str(llm_row["LLM诊断文本"])  
            
            llm_parsed = parse_llm_output(llm_text)
            
            try:
                annotated_row = annotated_data[annotated_data.index == sample_id - 1]
                if not annotated_row.empty:
                    if llm_parsed["静息血压"] is None:
                        llm_parsed["静息血压"] = annotated_row["静息血压(mm Hg)"].values[0]
                    if llm_parsed["血清胆固醇"] is None:
                        llm_parsed["血清胆固醇"] = annotated_row["血清胆固醇(mg/dl)"].values[0]
            except Exception as e:
                logger.warning(f"⚠ 样本{sample_id}匹配标注数据出错：{str(e)}")
            
            match_score = evaluate_index_diagnosis_match(llm_parsed, llm_parsed["诊断结果"])
            norm_score = evaluate_diagnosis_terminology(llm_text)
            risk_result = detect_diagnosis_risk(llm_text)
            
            total_match_score += match_score
            total_norm_score += norm_score
            if risk_result["存在风险"]:
                risk_sample_count += 1
            
            if match_score < 4 or norm_score < 4:
                low_score_samples.append({
                    "样本序号": sample_id,
                    "LLM文本": llm_text,
                    "低分原因": "匹配度低" if match_score < 4 else "规范性低",
                    "真实指标": f"血压{llm_parsed['静息血压']}|胆固醇{llm_parsed['血清胆固醇']}",
                    "建议优化方向": "强化核心指标与诊断的关联" if match_score < 4 else "使用标准临床术语"
                })
            
            assessment_results.append({
                "样本序号": sample_id,
                "LLM诊断文本": llm_text,
                "解析后血压": llm_parsed["静息血压"],
                "解析后胆固醇": llm_parsed["血清胆固醇"],
                "匹配得分(1-5)": match_score,
                "规范得分(1-5)": norm_score,
                "是否有风险": "是" if risk_result["存在风险"] else "否",
                "风险详情": str(risk_result["风险详情"]) if risk_result["存在风险"] else "无"
            })
        
        total_samples = len(llm_output)
        avg_match_score = round(total_match_score / total_samples, 2) if total_samples > 0 else 0.0
        avg_norm_score = round(total_norm_score / total_samples, 2) if total_samples > 0 else 0.0
        risk_rate = round((risk_sample_count / total_samples) * 100, 2) if total_samples > 0 else 0.0
        
        # 保存CSV报告
        pd.DataFrame(assessment_results).to_csv(ASSESSMENT_REPORT_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame(low_score_samples).to_csv(OPTIMIZATION_SUGGESTIONS_PATH, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 80)
        print(" LLM诊断输出自动化评估总结（匹配你的data1文件夹）")
        print("=" * 80)
        print(f"1. 评估样本总数：{total_samples} 条")
        print(f"2. 平均匹配得分：{avg_match_score} 分 | 平均规范得分：{avg_norm_score} 分")
        print(f"3. 风险样本占比：{risk_rate}% （共{risk_sample_count}条）")
        print(f"4. 低分样本数：{len(low_score_samples)} 条（已生成优化建议）")
        print("5. 评估结论：{'✅ 可直接用于训练' if (avg_match_score≥4 and risk_rate≤5) else '⚠ 需先优化低分样本'}")
        print(f"\n评估报告保存至：{ASSESSMENT_REPORT_PATH}")
        print(f"训练优化建议保存至：{OPTIMIZATION_SUGGESTIONS_PATH}")
        
    except Exception as e:
        logger.error(f"评估过程出错：{str(e)}", exc_info=True)
        print(f"\n执行失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print(" 当前文件夹（data1）下的文件列表：")
    for idx, file in enumerate(os.listdir(CURRENT_DIR), 1):
        print(f"   {idx}. {file}")
    print("-" * 50)
    
    evaluate_llm_output()