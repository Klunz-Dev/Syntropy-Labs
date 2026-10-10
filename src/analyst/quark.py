import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

os.makedirs('models', exist_ok=True)

print('[*] The data inside the csv file must be strictly int type')
path = input('path: ').strip()

print('\n[*] There can be only one target; to set it, enter the name of the column you want to designate as the target.')
target = input('target: ').strip()

print('\n[*] You can remove unnecessary columns—such as IDs and the like—as this will help our model.')
unnecessary_columns = input('unnecessary columns: ').strip()

user_data = []
columns_name = []
columns_for_drop = [target]

if unnecessary_columns:
    columns_for_drop.extend([col.strip() for col in unnecessary_columns.split(',')])

df = pd.read_csv(path)
df = df.select_dtypes(['int', 'float'])

X = df.drop(labels=columns_for_drop, axis=1, errors='ignore')
y = df[target]

center = len(df) // 2
X_train, X_test = X.iloc[:center], X.iloc[center:]
y_train, y_test = y.iloc[:center], y.iloc[center:]

isolation_forest = IsolationForest()
linear_regression = LinearRegression()
k_means = KMeans(n_clusters=3)

isolation_forest.fit(X_train)
linear_regression.fit(X_train, y_train)
k_means.fit(X_train)

for column in X.columns:
    if column == target:
        continue

    input_data = float(input(f'{column}: ').strip())
    user_data.append(input_data)
    columns_name.append(column)

user_df = pd.DataFrame([user_data], columns=columns_name)
print(user_df)
input('> ')

isolation_forest_predict = isolation_forest.predict(X)
linear_regression_predict = linear_regression.predict(user_df)
k_means_predict = k_means.predict(X)

# ls_model_accuracy = accuracy_score(y_train, normalize=False)

df['anomaly'] = isolation_forest_predict
df['cluster'] = k_means_predict

print(df)
print(f'Предсказание на основе массива: {linear_regression_predict}')

quark_artifact = {
    'anomaly_model': isolation_forest,
    'regression_model': linear_regression,
    'clustering_model': k_means,
    'feature_columns': X.columns.tolist()
}

joblib.dump(quark_artifact, 'quark_model.joblib')