import json
from sklearn.model_selection import train_test_split
def split_dataset(json_file, train_file, test_file, test_size=0.2):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = [d['label'] for d in data]
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    print(f"{json_file}: 总{len(data)}条，训练集{len(train_data)}条，测试集{len(test_data)}条")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

split_dataset(r'E:\创新数据\instruction JSON文件\heart_instructions.json', 'heart_train.json', 'heart_test.json')
split_dataset(r'E:\创新数据\instruction JSON文件\cleveland_instructions.json', 'cleveland_train.json', 'cleveland_test.json')
split_dataset(r'E:\创新数据\instruction JSON文件\framingham_instructions.json', 'framingham_train.json', 'framingham_test.json')