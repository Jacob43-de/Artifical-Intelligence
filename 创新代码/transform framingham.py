import pandas as pd
def framingham_to_text(row):
    sex = "male" if row['male'] == 1 else "female"
    smoker = "current smoker" if row['currentSmoker'] == 1 else "non-smoker"
    cigs = f", smoking {row['cigsPerDay']} cigarettes per day" if row['currentSmoker'] == 1 else ""
    bp_meds = "taking blood pressure medication" if row['BPMeds'] == 1 else "not taking blood pressure medication"
    prev_stroke = "with prior stroke" if row['prevalentStroke'] == 1 else "no prior stroke"
    prev_hyp = "with hypertension" if row['prevalentHyp'] == 1 else "no hypertension"
    diabetic = "diabetic" if row['diabetes'] == 1 else "non-diabetic"
    if pd.notna(row['BMI']):
        if row['BMI'] < 18.5:
            bmi_status = "underweight"
        elif row['BMI'] < 25:
            bmi_status = "normal weight"
        elif row['BMI'] < 30:
            bmi_status = "overweight"
        else:
            bmi_status = "obese"
    else:
        bmi_status = "unknown BMI"
    text = f"A {row['age']}-year-old {sex} patient. "
    text += f"{smoker}{cigs}. "
    text += f"{bp_meds}. "
    text += f"Medical history: {prev_stroke}, {prev_hyp}, {diabetic}. "
    text += f"Total cholesterol: {row['totChol']} mg/dL. "
    text += f"Blood pressure: {row['sysBP']}/{row['diaBP']} mmHg. "
    text += f"BMI: {row['BMI']:.1f} ({bmi_status}). "
    text += f"Heart rate: {row['heartRate']} bpm. "
    text += f"Glucose: {row['glucose']} mg/dL."
    label = "high 10-year CHD risk" if row['TenYearCHD'] == 1 else "low 10-year CHD risk"
    
    return text, label
df = pd.read_csv('framingham_clean.csv')  
df['text'], df['label_text'] = zip(*df.apply(framingham_to_text, axis=1))
df[['text', 'label_text']].to_csv('framingham_text.csv', index=False)
print(f"转换完成，共{len(df)}条")