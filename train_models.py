# import basic libraries
import numpy as np
# import deep learning libraries
from keras.layers import Input, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
# from keras.optimizers import Adagrad
# from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
# import ml libraries
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
# import utils
from ML_clf_with_dl_Class import MlClfWithDl
from util import create_class_weight
from collections import Counter
import params


def train_dl_veh_or_not_model(features_train, labels_train, train_opt, features_test=None, labels_test=None):
    """
    To train the deep learning model
    :param features_train: The features used to train the model
    :param labels_train: The labels used to train the model
    :param train_opt: training options, such as l2, epoch number and batch size
    :param features_test:
    :param labels_test:
    :return: The model
    """
    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    # one hot encode the output variable
    cat_labels_train = np_utils.to_categorical(labels_train)
    cat_labels_test = np_utils.to_categorical(labels_test)
    # build model
    input_shape = len(list(features_train))
    features_train = np.array(features_train)
    inputs = Input(shape=(input_shape, ))

    layer1 = Dense(32, activation='relu', kernel_regularizer=l2(train_opt['l2']))(inputs)
    drop1 = Dropout(0.3)(layer1)
    layer2 = Dense(16, activation='relu', kernel_regularizer=l2(train_opt['l2']))(drop1)
    drop2 = Dropout(0.3)(layer2)

    predictions = Dense(len(set(labels_train)), activation='softmax', kernel_regularizer=l2(train_opt['l2']))(drop2)
    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    # TODO
    # sgd = optimizers.SGD(lr = 0.01, clipnorm=1.)
    # model.compile(optimizer= Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)
    # tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    # mcp = ModelCheckpoint(filepath=train_opt['folder_name']+train_opt['model_name'] + '.hdf5', monitor='val_acc',
    #                       save_best_only=True)
    model.fit(features_train, np.array(cat_labels_train), verbose=2, epochs=train_opt['epochs'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr],
              validation_data=(np.array(features_test), np.array(np.array(cat_labels_test))))

    del input_shape, layer1, drop1, layer2, drop2, \
        class_weight, cat_labels_train, predictions, reduce_lr, features_train, labels_train

    return model


def train_dl_veh_type_model(features_train, labels_train, train_opt, features_test=None, labels_test=None):
    """
    To train the deep learning model
    :param features_train: The features used to train the model
    :param labels_train: The labels used to train the model
    :param train_opt: training options, such as l2, epoch number and batch size
    :param features_test:
    :param labels_test:
    :return: The model
    """
    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    # one hot encode the output variable
    cat_labels_train = np_utils.to_categorical(labels_train)
    cat_labels_test = np_utils.to_categorical(labels_test)
    # build model
    input_shape = len(list(features_train))
    features_train = np.array(features_train)
    inputs = Input(shape=(input_shape, ))

    layer1 = Dense(128, activation='relu', kernel_regularizer=l2(train_opt['l2']))(inputs)
    layer2 = Dense(96, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer1)
    layer3 = Dense(64, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer2)
    layer4 = Dense(32, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer3)
    drop1 = Dropout(0.2)(layer4)
    layer5 = Dense(16, activation='relu', kernel_regularizer=l2(train_opt['l2']))(drop1)

    predictions = Dense(len(set(labels_train)), activation='softmax', kernel_regularizer=l2(train_opt['l2']))(layer5)
    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    # TODO
    # sgd = optimizers.SGD(lr = 0.01, clipnorm=1.)
    # model.compile(optimizer= Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)
    # tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    # mcp = ModelCheckpoint(filepath=train_opt['folder_name'] + train_opt['model_name'] + '.hdf5', monitor='val_acc',
    #                       save_best_only=True)

    model.fit(features_train, np.array(cat_labels_train), verbose=2, epochs=train_opt['epochs'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr],
              validation_data=(np.array(features_test), np.array(np.array(cat_labels_test))))
    del inputs, layer1, layer2, layer4, drop1, layer5
    if train_opt['middle_output'] is True:
        middle_layers = Model(inputs=inputs, outputs=layer3)
        middle_output = middle_layers.predict(np.array(features_train))
        clf = None
        if train_opt['ml_opt'] is 'rf':
            clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight)
            # clf = RandomForestClassifier(n_estimators=30)
        elif train_opt['ml_opt'] is 'svm':
            clf = svm.SVC(kernel='rbf', C=1, class_weight=class_weight)
        elif train_opt['ml_opt'] is 'ada':
            clf = AdaBoostClassifier(n_estimators=20)
        else:
            print("Wrong ml_opt!")
            quit()
        clf.fit(middle_output, labels_train)

        res_model = MlClfWithDl(clf, middle_layers, train_opt['ml_opt'])

        del class_weight, cat_labels_train, predictions, reduce_lr, clf

        return res_model

    del class_weight, cat_labels_train, predictions, reduce_lr

    return model


def train_ml_model(features_train, labels_train, ml_opt):
    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    clf = None
    if ml_opt is 'rf':
        clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight)
        # clf = RandomForestClassifier(n_estimators=30)
    elif ml_opt is 'svm':
        clf = svm.SVC(kernel='rbf', C=1, class_weight=class_weight)
        # clf = svm.SVC(kernel='poly', C=1, class_weight=class_weight)
    elif ml_opt is 'ada':
        clf = AdaBoostClassifier(n_estimators=20)
    else:
        print("Wrong ml_opt!")
        quit()
    clf.fit(features_train, labels_train)

    del class_weight

    return clf


def train_rf_model_loop(features_train, labels_train, dep, n_esti, max_features):
    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    clf = RandomForestClassifier(n_estimators=n_esti, max_features=max_features, class_weight=class_weight,
                                 max_depth=dep)

    clf.fit(features_train, labels_train)

    del class_weight

    return clf


def train_lstm_model(features_train, labels_train, train_opt):
    """
        To train the deep learning model
        :param features_train: The features used to train the model
        :param labels_train: The labels used to train the model
        :param train_opt: training options, such as l2, epoch number and batch size
        :return: The model
        """
    # create class weight for imbalanced classes
    class_weight = create_class_weight(Counter(labels_train))

    # one hot encode the output variable
    cat_labels_train = np_utils.to_categorical(labels_train)

    features_train = np.reshape(np.array(features_train),
                                (len(features_train), params.window_size,
                                 int(len(list(features_train)) / params.window_size)))

    # build model
    inputs = Input(shape=(features_train.shape[1], features_train.shape[2]))

    layer1 = LSTM(128, kernel_regularizer=l2(train_opt['l2']))(inputs)
    layer5 = Dense(64, kernel_regularizer=l2(train_opt['l2']))(layer1)

    predictions = Dense(len(set(labels_train)), activation='softmax', kernel_regularizer=l2(train_opt['l2']))(layer5)
    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    # TODO
    # sgd = optimizers.SGD(lr = 0.01, clipnorm=1.)
    # model.compile(optimizer= Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)
    # tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    model.fit(features_train, np.array(cat_labels_train), verbose=2, epochs=train_opt['epochs'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr])

    # if train_opt['middle_output'] is True:
    #     middle_layers = Model(inputs=inputs, outputs=layer3)
    #     middle_output = middle_layers.predict(np.array(features_train))
    #     clf = None
    #     if train_opt['ml_opt'] is 'rf':
    #         clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight)
    #         # clf = RandomForestClassifier(n_estimators=30)
    #     elif train_opt['ml_opt'] is 'svm':
    #         clf = svm.SVC(kernel='rbf', C=1, class_weight=class_weight)
    #     elif train_opt['ml_opt'] is 'ada':
    #         clf = AdaBoostClassifier(n_estimators=20)
    #     else:
    #         print("Wrong ml_opt!")
    #         quit()
    #     clf.fit(middle_output, labels_train)
    #
    #     res_model = MlClfWithDl(clf, middle_layers, train_opt['ml_opt'])
    #
    #     del class_weight, cat_labels_train, predictions, reduce_lr, clf
    #
    #     return res_model

    del class_weight, cat_labels_train, predictions, reduce_lr

    return model
