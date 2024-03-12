import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Загрузка и подготовка данных
train_data = pd.read_csv('train_df.csv')
test_data = pd.read_csv('test_df.csv')
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

scaler = StandardScaler()
features = ['feature_{}'.format(i) for i in range(79)]  #список признаков, мб переделать грамотнее
train_data[features] = scaler.fit_transform(train_data[features])
test_data[features] = scaler.transform(test_data[features])

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
model.fit(X_train, y_train)
# Оценка модели
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


probas = model.predict_proba(X_test)[:, 1]
sorted_indices = np.argsort(probas)[::-1]

relevances = y_test.iloc[sorted_indices]

# Вычисление NDCG для всех доков
def ndcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    dcg = np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))
    idcg = np.sum((2 ** np.sort(r)[::-1] - 1) / np.log2(np.arange(2, r.size + 2)))
    return dcg / idcg if idcg > 0 else 0

ndcg = ndcg_at_k(relevances, len(relevances))
print("NDCG:", ndcg)