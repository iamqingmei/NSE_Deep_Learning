import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from feature_calc import DL_FEATURES, WIN_FEATURES
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from evaluation import evaluation_four_result
import random
import datetime
from class_weight import create_class_weight
from collections import Counter
import app_dataset
import manual_dataset
# import itchat

begin_time = datetime.datetime.now()
train_opt = {'batch_size': 32, 'nb_epoch': 200, 'window_size': 6, 'DLNetwork': 'FULL', 'num_test': 0.333,
             'test_opt': 'random', 'l2': 0.00001,
             'remark': 'testing only'}
folder_name = './four-acc/evaluation_report/' + str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '/'
write = "Four Result Evaluation Report: \n\n"
# itchat.auto_login(hotReload=True)


def train_model(win_df):
    global write
    win_df = win_df[win_df.win_label != -1]
    win_df = win_df[win_df.win_label != 5]

    test_idx = random.sample(range(len(win_df)), int(len(win_df) * train_opt['num_test']))
    train_idx = [x for x in range(len(win_df)) if x not in test_idx]

    train_win_df = win_df.iloc[train_idx, :]
    test_win_df = win_df.iloc[test_idx, :]

    # Get features_train, features_test, labels_train and labels_test
    features_train = train_win_df.iloc[:, 0:-1]
    features_test = test_win_df.iloc[:, 0:-1]
    labels_train = train_win_df['win_label']
    labels_test = test_win_df['win_label']

    # create class weight for imbalanced classes
    write += "Counter(labels_train): " + "\n"
    write += str(Counter(labels_train)) + "\n"
    class_weight = create_class_weight(Counter(labels_train))
    write += "class_weight: " + str(class_weight) + "\n"
    write += "Counter(labels_test)" + "\n"
    write += str(Counter(labels_test)) + "\n"

    # one hot encode the output variable
    cat_labels_train = np_utils.to_categorical(labels_train)
    cat_labels_test = np_utils.to_categorical(labels_test)

    # build model
    input_shape = len(list(features_train))
    inputs = Input(shape=(input_shape,))

    layer1 = Dense(128, activation='relu', kernel_regularizer=l2(train_opt['l2']))(inputs)
    layer2 = Dense(96, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer1)
    layer3 = Dense(64, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer2)
    layer4 = Dense(32, activation='relu', kernel_regularizer=l2(train_opt['l2']))(layer3)
    drop1 = Dropout(0.2)(layer4)
    layer5 = Dense(16, activation='relu', kernel_regularizer=l2(train_opt['l2']))(drop1)

    predictions = Dense(len(set(labels_test)), activation='softmax', kernel_regularizer=l2(train_opt['l2']))(layer5)
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)

    model.fit(np.array(features_train), np.array(cat_labels_train), verbose=2, epochs=train_opt['nb_epoch'],
              batch_size=train_opt['batch_size'], callbacks=[reduce_lr])

    return model, features_test, cat_labels_test


def main():
    global write
    write += str(datetime.datetime.now()) + '\n'
    write += str(train_opt) + '\n'
    write += 'PT FEATURES included: ' + str(DL_FEATURES) + '\n'
    write += 'WIN FEATURES included: ' + str(WIN_FEATURES) + '\n'

    # the app_win_df here is unbalanced and has 5 labels
    app_win_df = app_dataset.get_app_win_df(train_opt['window_size'])

    # the manual_win_df here is unbalanced and has 5 labels
    manual_win_df = manual_dataset.get_manual_win_df(train_opt['window_size'])

    app_model, app_features_test, app_labels_test = train_model(app_win_df)
    manual_model, manual_features_test, manual_labels_test = train_model(manual_win_df)

    write += evaluation_four_result(folder_name, app_model, manual_model, np.array(app_features_test),
                                    np.array(app_labels_test), np.array(manual_features_test),
                                    np.array(manual_labels_test), app_win_df, manual_win_df)

    with open(folder_name + 'evaluation.txt', 'w') as f:
        f.truncate()
        f.write(write)

    end_time = datetime.datetime.now()
    print "spending time:" + str(end_time-begin_time)
    print write

    # itchat.send(write, toUserName='filehelper')

if __name__ == '__main__':
    main()
