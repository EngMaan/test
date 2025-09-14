import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from xgboost import  XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA



df = pd.read_csv("conveyor_pdm_multilifecycle_kpa_with_belt_RUL.csv")


target_column = "Fault_Label"  
skipped_clms = ["Fault_Label","Fault_Type","Timestamp","RUL_hours"]

X = df.drop(columns= skipped_clms, axis=1)
y = df[target_column]
'''
# ==============================
# 2. Encode Target Labels (categorical -> integers)
# ==============================
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# ==============================
# 3. Train-Test Split
# ==============================
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# num_classes = len(np.unique(y))
# model = XGBClassifier(
#             n_estimators=1,
#             max_depth=5,
#             learning_rate=0.2,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             reg_lambda=1.0,
#             objective="multi:softprob" if num_classes > 2 else "binary:logistic",
#             eval_metric="mlogloss" if num_classes > 2 else "logloss",
#             tree_method="hist"
#         )
# model.fit(X_train, y_train)
model =XGBRegressor(
    n_estimators=100,         # عدد الأشجار
    max_depth=5,              # عمق كل شجرة
    learning_rate=0.2,        # معدل التعلم
    subsample=0.9,            # نسبة العينة من البيانات لكل شجرة
    colsample_bytree=0.9,     # نسبة الأعمدة المختارة لكل شجرة
    reg_lambda=1.0,           # معامل L2 regularization
    objective="reg:squarederror",  # الهدف: انحدار بمربع الخطأ
    eval_metric="rmse"        # مقياس التقييم: الجذر التربيعي لمتوسط مربع الخطأ
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (global feature importance)

# Example: explain the first prediction
# sample_idx = 0  # change index to check another row
# shap.plots.waterfall(shap.Explanation(
#     values=shap_values[sample_idx],
#     base_values=explainer.expected_value,
#     data=X_test.iloc[sample_idx],
#     feature_names=X_test.columns
# ))

y_pred = model.predict(X_test)
feature_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsSHAP": feature_importance
}).sort_values(by="MeanAbsSHAP", ascending=False)

print(importance_df)
# fault_indices = y_pred == 2
# X_fault_only = X_test[fault_indices]

# تطبيع البيانات قبل الـ KMeans
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_fault_only)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# kmeans = KMeans(n_clusters=3)
# fault_clusters = kmeans.fit_predict(X_pca)

# تدريب KMeans لتصنيف أنواع الأعطال (مثلاً 3 أنواع)
# kmeans = KMeans(n_clusters=2, random_state=42)
# fault_clusters = kmeans.fit_predict(X_scaled)
# for k in range(2, 10):
# kmeans = KMeans(n_clusters=3, random_state=42)
# fault_clusters = kmeans.fit_predict(X_scaled)
# score = silhouette_score(X_scaled, fault_clusters)
# print(f"k={3}, Silhouette Score={score}")
# score = davies_bouldin_score(X_scaled, fault_clusters)
# print("Davies-Bouldin Index:", score)
# score = calinski_harabasz_score(X_scaled, fault_clusters)
# print("Calinski-Harabasz Index:", score)





# # إظهار النتائج
# X_fault_only["fault_type_cluster"] = fault_clusters
# print(X_fault_only.head())

# shap.summary_plot(shap_values, X_test, feature_names=X.columns, max_display=len(X.columns))
# plt.show(block=False)   
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(cm.shape[0]))
# disp.plot(cmap=plt.cm.Blues)  # optional: change colormap
# plt.title("Confusion Matrix")
# # predicted_labels = label_encoder.inverse_transform(y_pred)
# # print("Predicted labels:", predicted_labels[:10])
# plt.show()