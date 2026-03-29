import pandas as pd
def heart_to_text(row):
    sex = "male" if row['Sex'] == 'M' else "female"
    chest_pain_map = {
        'ATA': 'atypical angina',
        'NAP': 'non-anginal pain',
        'ASY': 'asymptomatic',
        'TA': 'typical angina'
    }
    chest_pain = chest_pain_map.get(row['ChestPainType'], 'unknown')
    ecg_map = {
        'Normal': 'normal',
        'ST': 'ST-T wave abnormality',
        'LVH': 'left ventricular hypertrophy'
    }
    ecg = ecg_map.get(row['RestingECG'], 'unknown')
    exercise_angina = "with" if row['ExerciseAngina'] == 'Y' else "without"
    st_map = {
        'Up': 'upsloping',
        'Flat': 'flat',
        'Down': 'downsloping'
    }
    st_slope = st_map.get(row['ST_Slope'], 'unknown')
    fasting_bs = "elevated" if row['FastingBS'] == 1 else "normal"
    text = f"A {row['Age']}-year-old {sex} patient presents with {chest_pain}. "
    text += f"Resting blood pressure is {row['RestingBP']} mmHg, cholesterol level is {row['Cholesterol']} mg/dL. "
    text += f"Fasting blood sugar is {fasting_bs}. "
    text += f"Resting ECG shows {ecg} findings. "
    text += f"Maximum heart rate achieved is {row['MaxHR']} bpm, {exercise_angina} exercise-induced angina. "
    text += f"ST depression of {row['Oldpeak']} mm with {st_slope} ST segment."
    label = "has heart disease" if row['HeartDisease'] == 1 else "does not have heart disease"
    return text, label
df = pd.read_csv('heart.csv')
df['text'], df['label_text'] = zip(*df.apply(heart_to_text, axis=1))
df[['text', 'label_text']].to_csv('heart_text.csv', index=False)
print(f"转换完成，共{len(df)}条")