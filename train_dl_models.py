# import basic libraries
import numpy as np
# import deep learning libraries
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
# import utils
from util import create_class_weight
from collections import Counter


def train_model(features_train, labels_train, train_opt):
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

    # build model
    input_shape = len(list(features_train))
    inputs = Input(shape=(input_shape,))

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
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)

    model.fit(np.array(features_train), np.array(cat_labels_train), verbose=2, epochs=train_opt['epochs'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr])

    return model
