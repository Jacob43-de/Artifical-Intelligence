import json
import pandas as pd

def create_instruction_dataset(text_file, output_file, task_type="diagnosis"):
    df = pd.read_csv(text_file)
    instructions = []
    for idx, row in df.iterrows():
        if task_type == "diagnosis":
            instruction = "Given the following patient data, provide a cardiovascular diagnosis:"
        else:  
            instruction = "Based on the following patient data, predict the 10-year coronary heart disease risk:"

        answer = f"Patient information: {row['text']} "
        answer += f"The patient {row['label_text']}. "
        answer += "This assessment should be confirmed with additional clinical evaluation."
        
        instructions.append({
            "instruction": instruction,
            "input": row['text'],
            "output": answer,
            "label": row['label_text'],
            "task_type": task_type
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)
    
    print(f"已创建 {len(instructions)} 条指令数据，保存到 {output_file}")

create_instruction_dataset(r'E:\创新数据\转换文件\heart_text.csv', 'heart_instructions.json', 'diagnosis')
create_instruction_dataset(r'E:\创新数据\转换文件\cleveland_text.csv', 'cleveland_instructions.json', 'diagnosis')
create_instruction_dataset(r'E:\创新数据\转换文件\framingham_text.csv', 'framingham_instructions.json', 'risk_prediction')