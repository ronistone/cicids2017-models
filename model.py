import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from dataset_prepare import get_dataset, get_dataset_autoencoders
from result_presentation import show_results

from sklearn.decomposition import PCA, KernelPCA
from sklearn.neural_network import MLPRegressor

from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


models = {
    'RandomForest': RandomForestClassifier(n_estimators=10, max_depth=50, random_state=1),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SGD': SGDClassifier(loss='hinge', penalty='elasticnet', random_state=42, max_iter=200, shuffle=True),
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


from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def tsne_scatter(features, labels, dimensions=2, save_as='graph.png'):
    if dimensions not in (2, 3):
        raise ValueError(
            'tsne_scatter can only plot in 2d or 3d. Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Attack'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Benign'
    )

    # storing it to be displayed later
    plt.legend(loc='best')
    plt.savefig(save_as);
    plt.show;


def get_autoencoder(input_dim):
    BATCH_SIZE = 256

    # https://keras.io/layers/core/
    autoencoder = tf.keras.models.Sequential([

        # deconstruct / encode
        tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
        # tf.keras.layers.Dense(64, activation='elu'),
        # tf.keras.layers.Dense(32, activation='elu'),
        # tf.keras.layers.Dense(16, activation='elu'),
        # tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        # tf.keras.layers.Dense(2, activation='elu'),

        # reconstruction / decode
        # tf.keras.layers.Dense(4, activation='elu'),
        # tf.keras.layers.Dense(8, activation='elu'),
        # tf.keras.layers.Dense(16, activation='elu'),
        # tf.keras.layers.Dense(32, activation='elu'),
        # tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dense(input_dim, activation='elu')

    ])

    # Best is input - 4 - input | f1 benign: 0.82026 | f1 attack: 0.54375

    # https://keras.io/api/models/model_training_apis/
    autoencoder.compile(optimizer="adam",
                        loss="mse",
                        metrics=["acc"])

    # print an overview of our model
    autoencoder.summary()

    # current date and time
    yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')

    # new folder for a new run
    log_subdir = f'{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(autoencoder.layers)}'

    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=15,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath='autoencoder_best_weights.keras',
        save_best_only=True,
        monitor='val_loss',
        verbose=0,
        mode='min'
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        f'logs/{log_subdir}',
        update_freq='batch'
    )

    # callbacks argument only takes a list
    cb = [early_stop, save_model, tensorboard]

    return autoencoder, cb


THRESHOLD = 1

def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad

def plot_loss(mse, y_test):
    benign = mse[y_test == 0]
    attack = mse[y_test == 1]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.hist(benign, bins=50, density=True, label="benign", alpha=.6, color="green")
    ax.hist(attack, bins=50, density=True, label="attack", alpha=.6, color="red")

    plt.title("(Normalized) Distribution of the Reconstruction Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    path = 'cicids2017/'
    # x_train, x_test, y_train, y_test = get_dataset(path, test_size=0.8)

    X_train, X_validate, X_test, y_test = get_dataset_autoencoders(path, test_size=0.8)

    # tsne_scatter(x_train, y_train, dimensions=2, save_as='tsne_initial_2d-01.png')

    auto_encoder_start = time.time()

    model, callbacks = get_autoencoder(X_train.shape[1])
    model.fit(X_train, X_train, epochs=100, batch_size=256, validation_data=(X_validate, X_validate), callbacks=callbacks)
    reconstruction = model.predict(X_test)
    autoEncoderPredictions = np.mean(np.power(X_test - reconstruction, 2), axis=1)

    plot_loss(autoEncoderPredictions, y_test)

    z_scores = mad_score(autoEncoderPredictions)
    autoEncoderPredictions = z_scores > THRESHOLD

    auto_encoder_duration = time.time() - auto_encoder_start


    # sgdStart = time.time()
    #
    # sgdModel = models['SGD']
    # sgdModel.fit(x_train, y_train)
    # predictionsSgd = sgdModel.predict(x_train)
    # x_train = np.append(x_train, predictionsSgd[:, None], axis=1)
    #
    # predictionsSgd = sgdModel.predict(x_test)
    # x_test = np.append(x_test, predictionsSgd[:, None], axis=1)
    #
    # sgdDuration = time.time() - sgdStart

    # knnModel = models['KNN']
    # knnModel.fit(x_train, y_train)
    # predictionsKnn = knnModel.predict(x_train)
    # x_train = np.append(x_train, predictionsKnn[:, None], axis=1)
    #
    # predictionsKnn = knnModel.predict(x_test)
    # x_test = np.append(x_test, predictionsKnn[:, None], axis=1)


    # forestStart = time.time()
    #
    # model = models['RandomForest']
    # model.fit(x_train, y_train)
    # predictions = model.predict(x_test)
    #
    # forestDuration = time.time() - forestStart
    #
    # knnStart = time.time()
    #
    # knnModel = models['KNN']
    # knnModel.fit(x_train, y_train)
    # predictionsKnn = knnModel.predict(x_test)
    #
    # knnDuration = time.time() - knnStart

    # sgdStart = time.time()
    #
    # sgdModel = models['SGD']
    # sgdModel.fit(x_train, y_train)
    # predictionsSgd = sgdModel.predict(x_test)
    # sgdModel.l1_ratio
    #
    # sgdDuration = time.time() - sgdStart

    # show_results(y_test, predictions, 'RandomForest')
    # print('RandomForest duration:  {:.2f} seconds'.format(forestDuration))

    show_results(y_test, autoEncoderPredictions, 'AutoEncoders')
    print('AutoEncoder duration:  {:.2f} seconds'.format(auto_encoder_duration))

    # show_results(y_test, predictionsKnn, 'KNN')
    # print('KNN duration: {:.2f} seconds'.format(knnDuration))

    # show_results(y_test, predictionsSgd, 'SGD')
    # print('SGD duration: {:.2f} seconds'.format(sgdDuration))
