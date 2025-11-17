# --- 0) Install dependencies ---
!pip install -q gradio openpyxl seaborn joblib

# --- 1) Imports ---
import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

import gradio as gr

sns.set(style="whitegrid")

# --- 2) Upload dataset (used for training) ---
print("Upload your Excel dataset: Updated_PCOS_Dataset_with_Risk_Scores.xlsx")
from google.colab import files as colab_files
uploaded = colab_files.upload()
if len(uploaded) == 0:
    raise FileNotFoundError("Please upload the dataset file when prompted.")
dataset_path = list(uploaded.keys())[0]

print("Loading:", dataset_path)
df = pd.read_excel(dataset_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- 3) Column configuration ---
INPUT_FEATURES = [
 'Age','Weight (kg)','Height (cm)','BMI','Cycle_length_days','Irregular_periods','Facial_hair',
 'Acne','Hair_loss','Mood_swings','Sleep_issues','Physical_activity_level','Physical_activity_time',
 'Diet_type','Stress_level','Water_intake (L/day)','PCOS_risk_percent','Cycle_Length_Risk','BMI_Risk',
 'Facial_Hair_Risk','Acne_Risk','Hair_Loss_Risk','Irregular_Periods_Risk','Physical_Activity_Level_Risk',
 'Physical_Activity_Time_Risk','Diet_Risk','Stress_Risk','Water_Intake_Risk','Total_Risk_Percentage'
]
TARGET = 'PCOS_class_label'

missing = [c for c in INPUT_FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns in dataset: {missing}")

# --- 4) Preprocess ---
df_proc = df.copy()

binary_cols = ['Irregular_periods','Facial_hair','Acne','Hair_loss','Mood_swings','Sleep_issues']
for c in binary_cols:
    if c in df_proc.columns:
        df_proc[c] = df_proc[c].replace({'Yes':1,'No':0,'Y':1,'N':0,'y':1,'n':0}).astype(float)

cat_cols = ['Physical_activity_level','Diet_type','Stress_level']
label_encoders = {}
for c in cat_cols:
    if c in df_proc.columns:
        df_proc[c] = df_proc[c].astype(str).fillna("missing")
        le = LabelEncoder()
        df_proc[c] = le.fit_transform(df_proc[c])
        label_encoders[c] = le

numeric_imputer = SimpleImputer(strategy='median')
df_proc[INPUT_FEATURES] = numeric_imputer.fit_transform(df_proc[INPUT_FEATURES])

# --- 5) Prepare X, y and split ---
X = pd.DataFrame(df_proc[INPUT_FEATURES], columns=INPUT_FEATURES)
y = df_proc[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# --- 6) Scale for SVM ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7) Train Random Forest & SVM ---
print("Training RandomForest and SVM (this may take a moment)...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
rf_time = time.time() - t0
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred) * 100.0

t0 = time.time()
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_time = time.time() - t0
svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred) * 100.0

print(f"RF acc: {rf_acc:.2f}%  time: {rf_time:.2f}s")
print(f"SVM acc: {svm_acc:.2f}% time: {svm_time:.2f}s")

# --- 8) Metrics & plots ---
cm_rf = confusion_matrix(y_test, rf_pred)
cm_svm = confusion_matrix(y_test, svm_pred)
rf_proba = rf.predict_proba(X_test)[:,1]
svm_proba = svm.predict_proba(X_test_scaled)[:,1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_proba)
rf_auc = roc_auc_score(y_test, rf_proba)
svm_auc = roc_auc_score(y_test, svm_proba)

def make_confusion_figure():
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0], cmap='Blues', cbar=False)
    axes[0].set_title(f"Random Forest (acc={rf_acc:.2f}%)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    sns.heatmap(cm_svm, annot=True, fmt='d', ax=axes[1], cmap='Greens', cbar=False)
    axes[1].set_title(f"SVM (acc={svm_acc:.2f}%)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    plt.tight_layout()
    return fig

def make_roc_figure():
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC={rf_auc:.3f})")
    ax.plot(svm_fpr, svm_tpr, label=f"SVM (AUC={svm_auc:.3f})")
    ax.plot([0,1],[0,1],'k--', linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()
    return fig

def make_feature_imp_figure(top_n=12):
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=INPUT_FEATURES).sort_values(ascending=False)[:top_n]
    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax, palette="mako")
    ax.set_title("Top Feature Importances (RF)")
    plt.tight_layout()
    return fig

# --- 9) Heuristic risk & recommendations ---
def calculate_bmi_value(weight, height):
    try: return weight / (height/100)**2
    except: return 0.0

def yn(val):
    if isinstance(val, str):
        return 1 if val.lower() in ["yes","y","1","true","t"] else 0
    try:
        return int(val)
    except:
        return 0

def assess_risk_percentage(bmi, irregular_periods, facial_hair, acne, hair_loss, physical_activity, diet, stress, water):
    risk_percentage = 0
    risk_details = []
    if bmi < 18.5: risk_percentage += 20; risk_details.append("BMI: Underweight - +20%")
    elif bmi <= 24.9: risk_percentage += 10; risk_details.append("BMI: Optimal - +10%")
    elif bmi <= 29.9: risk_percentage += 30; risk_details.append("BMI: Overweight - +30%")
    else: risk_percentage += 50; risk_details.append("BMI: Obese - +50%")
    
    symptoms_vals = [facial_hair, acne, hair_loss, irregular_periods]
    symptom_risk = 10 * sum([int(v) for v in symptoms_vals])
    risk_percentage += symptom_risk
    for i, name in enumerate(["Facial Hair","Acne","Hair Loss","Irregular Periods"]):
        if symptoms_vals[i] == 1:
            risk_details.append(f"Symptom: {name} - +10%")
    
    if str(physical_activity).lower() == "sedentary": risk_percentage += 20; risk_details.append("Physical Activity: Sedentary - +20%")
    elif str(physical_activity).lower() == "moderate": risk_percentage += 10; risk_details.append("Physical Activity: Moderate - +10%")
    
    if diet in ["Processed","High Fat"]: risk_percentage += 15; risk_details.append("Diet: Processed/High Fat - +15%")
    if str(stress).lower() == "high": risk_percentage += 15; risk_details.append("Stress: High - +15%")
    elif str(stress).lower() == "moderate": risk_percentage += 10; risk_details.append("Stress: Moderate - +10%")
    
    if float(water) < 2: risk_percentage += 10; risk_details.append("Water Intake: Low - +10%")
    elif float(water) >= 3: risk_percentage -= 5; risk_details.append("Water Intake: High - -5%")
    
    risk_percentage = int(min(max(risk_percentage, 0), 100))
    return risk_percentage, risk_details

def generate_recommendations(risk_percentage, bmi=None, irregular=None, facial=None, acne=None, hair_loss=None, stress=None):
    recs = []
    if risk_percentage >= 60:
        recs.append("High risk: consult a gynecologist/endocrinologist soon.")
        if bmi and bmi >= 30: recs.append("Consider supervised weight-loss program.")
        if facial or acne: recs.append("Hormonal evaluation may be necessary.")
        if str(stress).lower() == "high": recs.append("Try stress reduction techniques.")
    elif risk_percentage >= 40:
        recs.append("Moderate risk: lifestyle changes & monitoring advised.")
        if bmi and bmi >= 25: recs.append("Manage weight via diet & exercise.")
    else:
        recs.append("Low risk: maintain healthy habits & routine checkups.")
    return list(dict.fromkeys(recs))

# --- 11) Prepare input row (user-provided inputs required) ---
def prepare_input_row(age, weight, height, cycle_len, irreg, facial, acne, hair_loss,
                      mood, sleep, act_level, act_time, diet, stress, water):
    row = {}
    row['Age'] = float(age)
    row['Weight (kg)'] = float(weight)
    row['Height (cm)'] = float(height)
    row['BMI'] = calculate_bmi_value(float(weight), float(height))
    row['Cycle_length_days'] = float(cycle_len)
    row['Irregular_periods'] = yn(irreg)
    row['Facial_hair'] = yn(facial)
    row['Acne'] = yn(acne)
    row['Hair_loss'] = yn(hair_loss)
    row['Mood_swings'] = yn(mood)
    row['Sleep_issues'] = yn(sleep)
    def map_cat(col, val):
        if col not in label_encoders:
            try: return float(val)
            except: return 0
        le = label_encoders[col]; sval = str(val)
        for i, cls in enumerate(le.classes_):
            if cls.lower() == sval.lower(): return int(i)
        if "missing" in le.classes_: return int(np.where(le.classes_=="missing")[0][0])
        return 0
    row['Physical_activity_level'] = map_cat('Physical_activity_level', act_level)
    row['Physical_activity_time'] = float(act_time)
    row['Diet_type'] = map_cat('Diet_type', diet)
    row['Stress_level'] = map_cat('Stress_level', stress)
    row['Water_intake (L/day)'] = float(water)
    for rc in ['PCOS_risk_percent','Cycle_Length_Risk','BMI_Risk','Facial_Hair_Risk','Acne_Risk','Hair_Loss_Risk',
               'Irregular_Periods_Risk','Physical_Activity_Level_Risk','Physical_Activity_Time_Risk','Diet_Risk',
               'Stress_Risk','Water_Intake_Risk','Total_Risk_Percentage']:
        row[rc] = 0
    input_df = pd.DataFrame([row], columns=INPUT_FEATURES)
    input_scaled = scaler.transform(input_df)
    return input_df, input_scaled

# --- 12) Gradio UI functions & in-memory reports (single login screen, logout) ---
USER_CREDENTIALS = {"user":"pass123","admin":"admin123"}

def login_handler(username, password, current_user_state, reports_state):
    """Single login screen logic. Returns visibility updates and states plus admin table."""
    # default: show login
    if USER_CREDENTIALS.get(username) == password:
        # set current user
        if username == "admin":
            # admin: show form_section and admin panel and provide reports table
            reports_df = pd.DataFrame(reports_state) if reports_state else pd.DataFrame()
            return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
                    f"✅ Welcome, {username}!", username, reports_state, reports_df)
        else:
            # normal user: show form_section but hide admin panel
            return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                    f"✅ Welcome, {username}!", username, reports_state, pd.DataFrame())
    else:
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                "❌ Invalid login credentials.", "", reports_state, pd.DataFrame())

def logout_handler(current_reports_state):
    """Return UI visibility to login screen and keep reports_state intact."""
    return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            "You have logged out.", "", current_reports_state, pd.DataFrame())

def ui_predict_and_report(age, weight, height, cycle_len, irreg, facial, acne, hair_loss,
                          mood, sleep, act_level, act_time, diet, stress, water, model_choice, current_user, reports_state):
    # Prepare input row
    input_df, input_scaled = prepare_input_row(age, weight, height, cycle_len, irreg, facial, acne, hair_loss,
                                               mood, sleep, act_level, act_time, diet, stress, water)

    # Model prediction
    if model_choice == "Random Forest":
        pred = int(rf.predict(input_df)[0])
        prob = float(rf.predict_proba(input_df)[0][1] * 100)
    else:
        pred = int(svm.predict(input_scaled)[0])
        prob = float(svm.predict_proba(input_scaled)[0][1] * 100)

    # Heuristic risk
    bmi_val = float(input_df.loc[0,'BMI'])
    heuristic_score, heuristic_details = assess_risk_percentage(
        bmi_val, yn(irreg), yn(facial), yn(acne), yn(hair_loss), act_level, diet, stress, float(water)
    )

    # Final assessment
    if pred == 1 or heuristic_score >= 60:
        final_assessment = "High Risk"
    elif heuristic_score >= 40 or prob >= 50:
        final_assessment = "Moderate Risk"
    else:
        final_assessment = "Low Risk"

    # Recommendations
    recs = generate_recommendations(
        heuristic_score, bmi=bmi_val, irregular=yn(irreg), facial=yn(facial),
        acne=yn(acne), hair_loss=yn(hair_loss), stress=stress
    )

    # Patient info
    patient_info = {
        "Age": age, "Weight (kg)": weight, "Height (cm)": height, "BMI": round(bmi_val,2),
        "Cycle_length_days": cycle_len, "Irregular_periods": irreg, "Facial_hair": facial,
        "Acne": acne, "Hair_loss": hair_loss, "Physical_activity_level": act_level,
        "Diet": diet, "Stress_level": stress, "Water_intake (L/day)": water
    }

    # Report text
    report_text = (
        f"Model: {model_choice}\n"
        f"Model prediction: {'PCOS (1)' if pred==1 else 'No PCOS (0)'}\n"
        f"Model positive-class prob: {prob:.2f}%\n"
        f"Heuristic risk score: {heuristic_score}%\n"
        f"Final assessment: {final_assessment}\n\n"
        f"Risk breakdown:\n" + ("\n".join(heuristic_details) if heuristic_details else "-") + "\n\n"
        f"Recommendations:\n" + ("\n".join(recs) if recs else "-") + "\n\n"
        f"Model metrics: RF acc={rf_acc:.2f}%, SVM acc={svm_acc:.2f}%"
    )

    # Save report to in-memory session list (reports_state)
    ts_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    report_record = {
        "timestamp": ts_readable,
        "user": current_user if current_user else "anonymous",
        "age": age,
        "weight": weight,
        "height": height,
        "bmi": round(bmi_val,2),
        "model": model_choice,
        "prediction": int(pred),
        "probability_%": round(prob,2),
        "heuristic_%": heuristic_score,
        "final_assessment": final_assessment,
        "risk_breakdown": "; ".join(heuristic_details) if heuristic_details else "-",
        "recommendations": "; ".join(recs) if recs else "-"
    }

    # Append to in-memory list
    reports_state = list(reports_state)  # ensure it's a mutable list
    reports_state.append(report_record)

    # If admin is viewing, send the dataframe for display; otherwise empty DataFrame
    reports_df = pd.DataFrame(reports_state) if current_user == "admin" else pd.DataFrame()

    # Return: report text, figures, feature-imp fig, reports_df (for admin), updated reports_state
    return report_text, make_confusion_figure(), make_roc_figure(), make_feature_imp_figure(), reports_df, reports_state

# --- 13) Gradio UI (single login screen, immediate admin dashboard on login) ---
css = """
body { background: linear-gradient(135deg,#fff5fb,#f4fbff); }
#component-0 { max-width: 1000px; margin: 8px auto;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet"), css=css) as demo:
    gr.Markdown("<h1 style='text-align:center;color:#8b2aa6'>💜 Smart PCOS Diagnosis System</h1>")

    # States: current_user and reports_state (in-memory list)
    current_user = gr.State("")          # stores username string after login
    reports_state = gr.State([])         # stores list of report dicts (in-memory)

    # Login row (visible initially)
    with gr.Row(visible=True) as login_section:
        with gr.Column():
            gr.Markdown("### 🔐 Login (demo user/pass123 or admin/admin123)")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Markdown()

    # Form section (visible after login)
    with gr.Column(visible=False) as form_section:
        # top-row: show current user + logout button
        with gr.Row():
            user_label = gr.Markdown(value="")   # will be updated with current user info
            logout_btn = gr.Button("Logout")

        with gr.Tabs():
            with gr.TabItem("PCOS Prediction"):
                gr.Markdown("### 🩺 Fill the form and get PCOS assessment")
                with gr.Row():
                    col1 = gr.Column()
                    col2 = gr.Column()
                    with col1:
                        age = gr.Number(label="Age", value=None)
                        weight = gr.Number(label="Weight (kg)", value=None)
                        height = gr.Number(label="Height (cm)", value=None)
                        cycle_len = gr.Number(label="Cycle Length (days)", value=None)
                        irreg = gr.Radio(["Yes","No"], label="Irregular Periods", value="No")
                        facial = gr.Radio(["Yes","No"], label="Facial Hair", value="No")
                        acne = gr.Radio(["Yes","No"], label="Acne", value="No")
                        hair_loss = gr.Radio(["Yes","No"], label="Hair Loss", value="No")
                    with col2:
                        mood = gr.Radio(["Yes","No"], label="Mood Swings", value="No")
                        sleep = gr.Radio(["Yes","No"], label="Sleep Issues", value="No")
                        pal_choices = list(label_encoders['Physical_activity_level'].classes_) if 'Physical_activity_level' in label_encoders else ["Sedentary","Moderate","Active"]
                        act_level = gr.Dropdown(pal_choices, label="Physical Activity Level", value=pal_choices[0])
                        act_time = gr.Number(label="Physical Activity Time (minutes/day)", value=None)
                        diet_choices = list(label_encoders['Diet_type'].classes_) if 'Diet_type' in label_encoders else ["Balanced","Processed","High Fat"]
                        diet = gr.Dropdown(diet_choices, label="Diet", value=diet_choices[0])
                        stress_choices = list(label_encoders['Stress_level'].classes_) if 'Stress_level' in label_encoders else ["Low","Moderate","High"]
                        stress = gr.Dropdown(stress_choices, label="Stress Level", value=stress_choices[0])
                        water = gr.Slider(minimum=0, maximum=5, step=0.1, label="Water Intake (L/day)", value=2.0)
                model_choice = gr.Dropdown(["Random Forest","SVM"], label="Choose Model", value="Random Forest")
                submit_btn = gr.Button("Generate Report")
                result_box = gr.Textbox(label="Assessment & Recommendations", lines=12)
                with gr.Row():
                    cm_fig = gr.Plot(label="Confusion Matrices")
                    roc_fig = gr.Plot(label="ROC Comparison")
                    fi_fig = gr.Plot(label="Feature Importance")

            # Admin panel tab (visible only for admin)
            with gr.TabItem("Admin Panel") as admin_tab:
                gr.Markdown("### 🛠 Admin: Session Reports (in-memory only)")
                reports_table = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="Session Reports")
                clear_btn = gr.Button("Clear Session Reports (admin only)")

    # Hook up login button: set visibility, current_user, and fill admin table immediately
    login_btn.click(fn=login_handler,
                    inputs=[username, password, current_user, reports_state],
                    outputs=[login_section, form_section, admin_tab, login_msg, current_user, reports_state, reports_table])

    # Update user_label with current_user when form_section visible
    def make_user_label(user):
        if not user:
            return ""
        return f"**Logged in as:** `{user}`"
    current_user.change(fn=make_user_label, inputs=[current_user], outputs=[user_label])

    # Hook up logout: bring back login screen, keep reports_state intact and clear admin table view
    logout_btn.click(fn=logout_handler, inputs=[reports_state], outputs=[login_section, form_section, admin_tab, login_msg, current_user, reports_state, reports_table])

    # Submit button click: generate report and append to session list; update admin table if admin
    submit_btn.click(fn=ui_predict_and_report,
                     inputs=[age, weight, height, cycle_len, irreg, facial, acne, hair_loss,
                             mood, sleep, act_level, act_time, diet, stress, water, model_choice, current_user, reports_state],
                     outputs=[result_box, cm_fig, roc_fig, fi_fig, reports_table, reports_state])

    # Admin clear session reports (admin only)
    def clear_reports(current_user, reports_state):
        if current_user != "admin":
            return reports_state, pd.DataFrame()
        return [], pd.DataFrame()
    clear_btn.click(fn=clear_reports, inputs=[current_user, reports_state], outputs=[reports_state, reports_table])

print("Demo credentials: user / pass123  (or admin / admin123)")
demo.launch(share=False)
