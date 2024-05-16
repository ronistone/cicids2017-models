from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from dataset_prepare import get_dataset
from result_presentation import show_results

from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

models = {
    'RandomForest': RandomForestClassifier(n_estimators=10, max_depth=50, random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

cols = ['#1FC17B', '#EE6352', '#AF929D', '#78FECF', '#555B6E', '#CC998D', '#429EA6',
        '#153B50', '#8367C7', '#C287E8', '#F0A6CA',
        '#521945', '#361F27', '#828489', '#9AD2CB', '#EBD494',
        '#53599A', '#80DED9', '#EF2D56', '#446DF6']


def plot_data(x, y):
    pca = PCA(n_components=3)
    res_pca = pca.fit_transform(x)

    unique_labels = np.unique(y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()

    for index, unique_label in enumerate(unique_labels):
        X_data = res_pca[y == unique_label]
        ax.scatter(X_data[:, 0], X_data[:, 1], X_data[:, 2], alpha=0.3, c=cols[index])
        # ax.scatter(X_data[:, 0], X_data[:, 1], alpha=0.3, c=cols[index])
    ax.set_xlabel('Principal Component #1')
    ax.set_ylabel('Principal Component #2')
    ax.set_zlabel('Principal Component #3')
    plt.title('PCA Results')
    plt.show()


if __name__ == '__main__':
    path = 'cicids2017/'
    x_train, x_test, y_train, y_test = get_dataset(path, test_size=0.8)

    # plot_data(x_train, y_train)

    model = models['RandomForest']
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    show_results(y_test, predictions)

# feature_importance = pd.DataFrame(
#   model.feature_importances_, index=x_columns, columns=['importance']
#   ).sort_values('importance', ascending=False)
# print(feature_importance[:60])