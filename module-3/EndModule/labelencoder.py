import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu từ file cleveland.csv
df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

# Gộp nhóm giá trị target thành 0 (không bị bệnh) và 1 (có bị bệnh)
df['target'] = df['target'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

# Điền các giá trị thiếu trong cột 'thal' và 'ca' bằng giá trị trung bình
df['thal'] = df['thal'].fillna(df['thal'].mean())
df['ca'] = df['ca'].fillna(df['ca'].mean())

# Chia dữ liệu thành X (đặc trưng) và y (nhãn)
X = df.iloc[:, :-1].values  # Các cột đặc trưng
y = df.iloc[:, -1].values   # Cột nhãn (target)

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình với KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Huấn luyện mô hình với SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Huấn luyện mô hình với Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Huấn luyện mô hình với Decision Tree
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Huấn luyện mô hình với AdaBoost
ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

# Huấn luyện mô hình với Gradient Boosting
gb_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Huấn luyện mô hình với XGBoost
xgboost_model = XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=100)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_test)

# Huấn luyện mô hình Stacking
estimators = [
    ('dtc', tree_model),
    ('rfc', RandomForestClassifier(random_state=42)),
    ('knn', knn),
    ('xgb', xgboost_model),
    ('gc', gb_model),
    ('svc', svm_model),
    ('ad', ada_model)
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)

# Hàm để tính và hiển thị độ chính xác của mô hình
def print_accuracy(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.round((cm[0][0] + cm[1][1]) / len(y_test), 2)
    print(f'Accuracy for {model_name} = {accuracy}')

# Hiển thị độ chính xác cho từng mô hình
print_accuracy(y_test, y_pred_knn, 'KNeighborsClassifier')
print_accuracy(y_test, y_pred_svm, 'SVM')
print_accuracy(y_test, y_pred_nb, 'Naive Bayes')
print_accuracy(y_test, y_pred_tree, 'Decision Tree')
print_accuracy(y_test, y_pred_ada, 'AdaBoost')
print_accuracy(y_test, y_pred_gb, 'GradientBoosting')
print_accuracy(y_test, y_pred_xgb, 'XGBoost')
print_accuracy(y_test, y_pred_stacking, 'Stacking')

