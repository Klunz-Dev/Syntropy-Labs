import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

print('[*] The data inside the csv file must be strictly int type')
path = input('path: ')

print('\n[*] There can be only one target; to set it, enter the name of the column you want to designate as the target.')
target = input('target: ')

print('\n[*] You can remove unnecessary columns—such as IDs and the like—as this will help our model.')
unnecessary_columns = input('unnecessary columns: ')

columns_for_drop = target + unnecessary_columns

if len(unnecessary_columns) == 0:
    columns_for_drop = target

df = pd.read_csv(path)


X = df.drop(labels=columns_for_drop, axis=1)
y = df['target']

center = len(df) // 2
X_train, X_test = X.iloc[:center], X.iloc[center:]
y_train, y_test = y.iloc[:center], y.iloc[center:]

df_train, df_test = df.iloc[:center], df.iloc[center:]

print(f'X train:{X_train}')
print(f'X test:{X_test}')

print(f'y train:{y_train}')
print(f'y test:{y_test}')

isolation_forest = IsolationForest()
linear_regression = LinearRegression()
k_means = KMeans()

isolation_forest.fit(df_train)
linear_regression.fit(X_train, y_train)
k_means.fit(df_train)

isolation_forest_predict = isolation_forest.predict(df_test)
linear_regression_predict = linear_regression.predict(X_test)
k_means_predict = k_means.predict(df_test)

# ls_model_accuracy = accuracy_score(y_train, normalize=False)

print(f'Аномалии в массиве данных: {isolation_forest_predict}')
print(f'Предскозание на основе массива: {linear_regression_predict}')
print(f'Разделения данных на категории в массиве: {k_means_predict}')

