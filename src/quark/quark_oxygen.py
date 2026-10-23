from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from src.data.data_manger import auto_cleaning_df

X, y = auto_cleaning_df('housing.csv', target='median_house_value')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

predict_model = LinearRegression()
cluster_model = KMeans(random_state=42, n_clusters=3)

predict_model.fit(X_train, y_train)
cluster_model.fit(X_train)

predictions = predict_model.predict(X_test)
clusters = cluster_model.predict(X_test)

r2_score_train = predict_model.score(X_train, y_train)
r2_score_test = predict_model.score(X_test, y_test)

print(f'Точность обучения: {100 * r2_score_train:.1f}%')
print(f'Точность валидации: {100 * r2_score_test:.1f}%')

print(f'Полученные кластеры: {clusters}')

print(f'Прогноз (первые 5 значений): {predictions[:5]}')

print(X, y)
