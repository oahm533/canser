import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import joblib
import os

# --- تدريب النموذج إذا لم يكن محفوظاً مسبقًا ---
if not (os.path.exists('scaler.joblib') and os.path.exists('pca.joblib') and os.path.exists('model.joblib')):
    st.info("⏳ جاري تدريب النموذج لأول مرة...")

    df = pd.read_csv(r'C:\Users\omar\Desktop\reg_cancer\data\cancer.csv')  # غيّر المسار إذا لزم

    X = df.drop(columns=['target'])
    y = df['target']

    # إزالة القيم الشاذة
    df_temp = X.copy()
    df_temp['target'] = y
    loc = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    df_temp['loc'] = loc.fit_predict(df_temp)
    df_temp = df_temp[df_temp['loc'] == 1].drop(columns=['loc'])
    X = df_temp.drop(columns=['target'])
    y = df_temp['target']

    # التحجيم وPCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_scaled)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # اختيار أفضل CV
    svc = SVC(probability=True)
    best_cv = None
    best_score = 0
    for cv_val in range(5, 20):
        scores = cross_val_score(svc, X_train, y_train, cv=cv_val)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_cv = cv_val

    st.write(f"🔧 أفضل قيمة لـ CV هي: **{best_cv}** بدقة متوسطة: **{best_score * 100:.2f}%**")

    # Grid Search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=best_cv)
    grid_search.fit(X_train, y_train)

    # دقة النموذج
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    st.write(f"📊 دقة النموذج على بيانات الاختبار: **{test_score * 100:.2f}%**")

    # حفظ المكونات
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(pca, 'pca.joblib')
    joblib.dump(grid_search, 'model.joblib')
    with open("test_score.txt", "w") as f:
        f.write(str(test_score))

# --- تحميل النموذج المحفوظ ---
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
model = joblib.load('model.joblib')

features = list(scaler.feature_names_in_)

# --- واجهة Streamlit ---
st.title("🔬 Breast Cancer Prediction (All-in-One)")
st.sidebar.image(r"C:\Users\omar\Desktop\reg_cancer\imge/cancer3.jpg")
st.image(r"C:\Users\omar\Desktop\reg_cancer\imge/cancer2.jpg")            

with open("test_score.txt", "r") as f:
    test_score = float(f.read())
st.write(f"📊 دقة النموذج على بيانات الاختبار: **{test_score * 100:.2f}%**")

st.markdown("### أدخل القيم لتوقع احتمالية الإصابة:")

# إدخال المستخدم
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(
        feature,
        value=0.0,
        step=0.1,
        key=feature
    )

input_df = pd.DataFrame([user_input], columns=features)

if st.sidebar.button("🔍 Predict Cancer Probability"):
    if input_df.isnull().values.any():
        st.error("⚠️ يوجد قيم ناقصة.")
    elif (input_df == 0.0).all(axis=1).values[0]:
        st.warning("⚠️ الرجاء إدخال القيم أولًا.")
    else:
        scaled_input = scaler.transform(input_df)
        pca_input = pca.transform(scaled_input)
        prediction = model.predict(pca_input)[0]
        prob = model.predict_proba(pca_input)[0][1]

        st.success(f"🔬 الاحتمالية: {prob * 100:.2f}%")
        if prediction == 1:
            st.success("🟢 التشخيص: حميد (Benign)")
        else:
            st.error("🔴 التشخيص: خبيث (Malignant)")

# cd C:\Users\omar\Desktop\reg_cancer
# python save_model.py

# streamlit run "C:\Users\omar\Desktop\reg_cancer\cancer.py"
