import json
import os

def convert_txt_to_preference_json(txt_file_path, output_json_path):
    """
    将病例txt文件转换为偏好数据对JSON文件
    :param txt_file_path: 输入的标注好的txt文件路径
    :param output_json_path: 输出的JSON文件路径
    """
    preference_data = []
    
    # 读取txt文件
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            cases_raw = f.read().split('\n\n')  
            print(f"共读取到 {len(cases_raw)} 个病例，开始转换...")
            
            for case_idx, case in enumerate(cases_raw):
                # 跳过空行/无效病例
                if not case.strip():
                    continue
                age = ""
                gender = ""
                symptoms = ""
                ecg_result = ""
                st_depression = ""
                st_slope = ""
                vessel_lesion = ""
                final_diagnosis = ""

                lines = case.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if "年龄：" in line:
                        age = line.split("年龄：")[1].split(" ")[0]
                    if "性别：" in line:
                        gender = line.split("性别：")[1].split(" ")[0]
                    if "症状：" in line:
                        symptoms = line.split("症状：")[1].split(" ")[0]
                    if "静息心电图：" in line:
                        ecg_result = line.split("静息心电图：")[1].split(" ")[0]
                    if "ST段压低：" in line:
                        st_depression = line.split("ST段压低：")[1].split(" ")[0]
                    if "ST段斜率：" in line:
                        st_slope = line.split("ST段斜率：")[1].split(" ")[0]
                    if "血管病变数：" in line:
                        vessel_lesion = line.split("血管病变数：")[1].split(" ")[0]
                    if "最终诊断：" in line:
                        final_diagnosis = line.split("最终诊断：")[1].strip()
                prompt = f"患者{age}{gender}，{symptoms}，静息心电图显示{ecg_result}，ST段压低程度为{st_depression}、峰值斜率呈{st_slope}改变，冠状动脉荧光染色检查显示{vessel_lesion}支血管存在病变"
                
                if final_diagnosis == "未发现心脏疾病":
                    chosen = f"该患者未发现心脏疾病。依据：最终诊断为未发现心脏疾病"
                    rejected = f"该患者存在心脏疾病相关风险，符合心血管疾病早期表现。依据：患者有{symptoms}，冠状动脉荧光染色显示{vessel_lesion}支血管病变，ST段压低{st_depression}"
                else:
                    chosen = f"该患者存在心脏疾病。依据：最终诊断为{final_diagnosis}"
                    rejected = f"该患者未发现心脏疾病。依据：无明确心脏疾病特征，仅{symptoms}为非特异性表现"
                
                # 将当前病例加入列表
                preference_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        # 写入JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=4)
        
        print(f"转换完成！共生成 {len(preference_data)} 条偏好数据对")
        print(f"JSON文件已保存至：{os.path.abspath(output_json_path)}")
        
    except FileNotFoundError:
        print(f"错误：未找到txt文件，请检查路径是否正确：{txt_file_path}")
    except Exception as e:
        print(f"转换过程中出错：{str(e)}，请检查txt文件格式或联系我调整解析逻辑")

if __name__ == "__main__":
    TXT_FILE_PATH = r"E:\three step\标注好的完整数据文件.txt"
    OUTPUT_JSON_PATH = r"E:\three step\偏好数据对.json"
    
    convert_txt_to_preference_json(TXT_FILE_PATH, OUTPUT_JSON_PATH)