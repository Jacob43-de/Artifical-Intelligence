import pandas as pd
def cleveland_to_text(row):
    sex = "male" if row['Sex'] == 1 else "female"
    chest_pain_map = {
        1: 'typical angina',
        2: 'atypical angina',
        3: 'non-anginal pain',
        4: 'asymptomatic'
    }
    chest_pain = chest_pain_map.get(row['Chest pain type'], 'unknown')
    ekg_map = {
        0: 'normal',
        1: 'ST-T wave abnormality',
        2: 'left ventricular hypertrophy'
    }
    ekg = ekg_map.get(row['EKG results'], 'unknown')
    exercise_angina = "with" if row['Exercise angina'] == 1 else "without"
    slope_map = {
        1: 'upsloping',
        2: 'flat',
        3: 'downsloping'
    }
    slope = slope_map.get(row['Slope of ST'], 'unknown')
    thallium_map = {
        3: 'normal',
        6: 'fixed defect',
        7: 'reversible defect'
    }
    thallium = thallium_map.get(row['Thallium'], 'unknown')
    text = f"A {row['Age']}-year-old {sex} patient presents with {chest_pain}. "
    text += f"Blood pressure is {row['BP']} mmHg, cholesterol level is {row['Cholesterol']} mg/dL. "
    text += f"Fasting blood sugar is {'elevated' if row['FBS over 120'] == 1 else 'normal'}. "
    text += f"Resting EKG shows {ekg} findings. "
    text += f"Maximum heart rate achieved is {row['Max HR']} bpm, {exercise_angina} exercise-induced angina. "
    text += f"ST depression of {row['ST depression']} mm with {slope} ST segment. "
    text += f"Fluoroscopy reveals {row['Number of vessels fluro']} vessels with significant stenosis. "
    text += f"Thallium stress test shows {thallium}."
    label = "has heart disease" if row['Heart Disease'] == 'Presence' else "does not have heart disease"
    
    return text, label
df = pd.read_csv('Heart_Disease_Prediction.csv')
df['text'], df['label_text'] = zip(*df.apply(cleveland_to_text, axis=1))
df[['text', 'label_text']].to_csv('cleveland_text.csv', index=False)
print(f"转换完成，共{len(df)}条")