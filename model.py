from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from dataset_prepare import get_dataset
from result_presentation import show_results

models = {
    'RandomForest': RandomForestClassifier(n_estimators=10, max_depth=50, random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_dataset(test_size=0.2)

    model = models['RandomForest']
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    show_results(y_test, predictions)

# feature_importance = pd.DataFrame(
#   model.feature_importances_, index=x_columns, columns=['importance']
#   ).sort_values('importance', ascending=False)
# print(feature_importance[:60])