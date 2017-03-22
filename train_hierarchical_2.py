import numpy as np
np.random.seed(97)
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from evaluation import evaluate_single_model, evaluate_overall_app_2, evaluate_overall_manual_2
from feature_calc import DL_FEATURES, WALK_STAT_FEATURES, VEHICLE_TYPE_FEATURES, \
    VEHICLE_OR_NOT_FEATURES
import datetime
from class_weight import create_class_weight
from collections import Counter
import app_dataset
import manual_dataset
# import itchat

begin_time = datetime.datetime.now()
train_opt = {'batch_size': 32, 'epochs': 200, 'window_size': 6, 'DLNetwork': 'FULL', 'num_test': 0.333,
             'test_opt': 'random', 'l2': 0.00001,
             'remark': 'testing features, without evaluate overall'}
folder_name = './hierarchical/evaluation_report/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M")) + '/'
# itchat.auto_login(hotReload=True)


def main():
    # get app_win_df
    app_win_df = app_dataset.get_app_win_df(train_opt['window_size'])
    # get manual_win_df
    manual_win_df = manual_dataset.get_manual_win_df(train_opt['window_size'])
    # set the win_label of manual_win_df as app_win_df
    # hence how manual_win_df does noe have win_label 1
    manual_win_df.win_label[(manual_win_df.win_label == 3)] = 4  # car
    manual_win_df.win_label[(manual_win_df.win_label == 2)] = 3  # bus
    manual_win_df.win_label[(manual_win_df.win_label == 1)] = 2  # mrt

    # divide app_win_df into app_win_df_test and app_win_df_train
    test_idx = np.random.choice(len(app_win_df), int(len(app_win_df) * train_opt['num_test']), replace=False)
    train_idx = [x for x in range(len(app_win_df)) if x not in test_idx.tolist()]
    app_win_df_test = app_win_df.iloc[test_idx, :]
    app_win_df_train = app_win_df.iloc[train_idx, :]
    # divide manual_win_df into manual_win_df_test and manual_win_df_train
    test_idx = np.random.choice(len(manual_win_df), int(len(manual_win_df) * train_opt['num_test']), replace=False)
    train_idx = [x for x in range(len(manual_win_df)) if x not in test_idx]
    manual_win_df_test = manual_win_df.iloc[test_idx, :]
    manual_win_df_train = manual_win_df.iloc[train_idx, :]

    # ~~~~~~~~~~~~~~ train vehicle_vs_non-vehicle model ~~~~~~~~~~~~~~
    vehicle_or_not_train_win_df = app_win_df_train.append(manual_win_df_train)
    vehicle_or_not_index = get_index(VEHICLE_OR_NOT_FEATURES)
    vehicle_or_not_index.append(-1)
    vehicle_or_not_train_win_df = vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index]

    # 0: walk or stationary
    # 2: mrt
    # 3: bus or car
    vehicle_or_not_train_win_df.win_label[(vehicle_or_not_train_win_df.win_label == 1)] = 0  # walk or stationary
    vehicle_or_not_train_win_df.win_label[(vehicle_or_not_train_win_df.win_label == 2)] = 1  # mrt->vehicle
    vehicle_or_not_train_win_df.win_label[(vehicle_or_not_train_win_df.win_label == 3)] = 1  # car->vehicle
    vehicle_or_not_train_win_df.win_label[(vehicle_or_not_train_win_df.win_label == 4)] = 1  # bus->vehicle

    vehicle_or_not_model = train_model(vehicle_or_not_train_win_df)

    # ~~~~~~~~~~~~~~ train walk VS stop ~~~~~~~~~~~~~~
    walk_stop_train_win_df = app_win_df_train.copy()
    walk_stop_index = get_index(WALK_STAT_FEATURES)
    walk_stop_index.append(-1)
    walk_stop_train_win_df = walk_stop_train_win_df.iloc[:, walk_stop_index]

    # 0: stationary
    # 1: walk
    walk_stop_train_win_df = walk_stop_train_win_df[(walk_stop_train_win_df.win_label == 0)
                                                    | (walk_stop_train_win_df.win_label == 1)]

    walk_stop_model = train_model(walk_stop_train_win_df)

    # ~~~~~~~~~~~~~~ train mrt VS car VS bus ~~~~~~~~~~~~~~
    vehicle_type_train_win_df = app_win_df_train.append(manual_win_df_train)
    vehicle_type_index = get_index(VEHICLE_TYPE_FEATURES)
    vehicle_type_index.append(-1)
    vehicle_type_train_win_df = vehicle_type_train_win_df.iloc[:, vehicle_type_index]
    # 3: bus
    # 4: car
    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df.win_label == 2)
                                                | (vehicle_type_train_win_df.win_label == 3)
                                                |(vehicle_type_train_win_df.win_label == 4)]

    vehicle_type_train_win_df.win_label[(vehicle_type_train_win_df.win_label == 2)] = 2  # mrt
    vehicle_type_train_win_df.win_label[(vehicle_type_train_win_df.win_label == 3)] = 0  # bus
    vehicle_type_train_win_df.win_label[(vehicle_type_train_win_df.win_label == 4)] = 1  # car

    vehicle_type_model = train_model(vehicle_type_train_win_df)

    # ~~~~~~~~~~~~~~~ Initiate write ~~~~~~~~~~~~~~~~
    write = "Evaluation Report of Train_Hierarchical \n"
    write += "Train option: " + str(train_opt) + "\n"
    write += "DL_FEATURES: " + str(DL_FEATURES) + "\n"
    write += "VEHICLE_OR_NOT_FEATURES" + str(VEHICLE_OR_NOT_FEATURES) + "\n"
    write += "WALK_STAT_FEATURES" + str(WALK_STAT_FEATURES) + "\n"
    write += "VEHICLE_TYPE_FEATURES" + str(VEHICLE_TYPE_FEATURES) + "\n"
    write += "vehicle_or_not_train_win_df: \n"
    write += str(Counter(vehicle_or_not_train_win_df['win_label'])) + '\n'
    write += "walk_stop_train_win_df: \n "
    write += str(Counter(walk_stop_train_win_df['win_label'])) + '\n'
    write += "vehicle_type_train_win_df: \n "
    write += str(Counter(vehicle_type_train_win_df['win_label'])) + '\n'
    # ~~~~~~~~~~~~~~ Evaluate Vehicle-or-not ~~~~~~~~~~~~~~~~~~~
    vehicle_or_not_test_win_df = app_win_df_test.append(manual_win_df_test)
    vehicle_or_not_test_win_df = vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]

    # 0: walk or stationary
    # 1: mrt or bus or cvehicle_or_not_test_win_dfar
    vehicle_or_not_test_win_df.win_label[(vehicle_or_not_test_win_df.win_label == 1)] = 0  # walk or stationary
    vehicle_or_not_test_win_df.win_label[(vehicle_or_not_test_win_df.win_label == 2)] = 1  # vehicle
    vehicle_or_not_test_win_df.win_label[(vehicle_or_not_test_win_df.win_label == 3)] = 1  # vehicle
    vehicle_or_not_test_win_df.win_label[(vehicle_or_not_test_win_df.win_label == 4)] = 1  # vehicle

    write_vehicle_or_not, acc_vehicle_or_not = evaluate_single_model(vehicle_or_not_model, folder_name, "vehicle_or_not",
                                                 np.array(vehicle_or_not_test_win_df.iloc[:, :-1]),
                                                 np.array(vehicle_or_not_test_win_df['win_label']))
    write += write_vehicle_or_not
    # ~~~~~~~~~~~~~~ Evaluate walk VS stop ~~~~~~~~~~~~~~
    walk_stop_test_win_df = app_win_df_test.copy()
    walk_stop_test_win_df = walk_stop_test_win_df.iloc[:, walk_stop_index]

    # 0: stationary
    # 1: walk
    walk_stop_test_win_df = walk_stop_test_win_df[(walk_stop_test_win_df.win_label == 0)
                                                  | (walk_stop_test_win_df.win_label == 1)]

    write_walk_stop, acc_walk_stop = evaluate_single_model(walk_stop_model, folder_name, "walkVSstop",
                                                           np.array(walk_stop_test_win_df.iloc[:, :-1]),
                                                           np.array(walk_stop_test_win_df['win_label']))

    write += write_walk_stop

    # ~~~~~~~~~~~~~~ Evaluate car VS bus ~~~~~~~~~~~~~~
    vehicle_type_test_win_df = app_win_df_test.append(manual_win_df_test)
    vehicle_type_test_win_df = vehicle_type_test_win_df.iloc[:, vehicle_type_index]

    # 0: bus
    # 1: car
    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df.win_label == 4)
                                              | (vehicle_type_test_win_df.win_label == 3)| (vehicle_type_test_win_df.win_label == 2)]

    vehicle_type_test_win_df.win_label[(vehicle_type_test_win_df.win_label == 3)] = 0  # bus
    vehicle_type_test_win_df.win_label[(vehicle_type_test_win_df.win_label == 4)] = 1  # car
    vehicle_type_test_win_df.win_label[(vehicle_type_test_win_df.win_label == 2)] = 2  # mrt

    write_vehicle_type, acc_vehicle_type = evaluate_single_model(vehicle_type_model, folder_name, "vehicle_type",
                                                       np.array(vehicle_type_test_win_df.iloc[:, :-1]),
                                                       np.array(vehicle_type_test_win_df['win_label']))

    write += write_vehicle_type

    vehicle_or_not_index.remove(-1)
    vehicle_type_index.remove(-1)
    walk_stop_index.remove(-1)
    write_app_overall, acc_app_overall = evaluate_overall_app_2(vehicle_or_not_model, walk_stop_model, vehicle_type_model,
                                                              np.array(app_win_df_test.iloc[:, :-1]),
                                                              np.array(app_win_df_test['win_label']), vehicle_or_not_index,
                                                              walk_stop_index, vehicle_type_index)

    write += write_app_overall

    write_manual_overall, acc_manual_overall = evaluate_overall_manual_2(vehicle_or_not_model, vehicle_type_model,
                                     np.array(manual_win_df_test.iloc[:, :-1]),
                                     np.array(manual_win_df_test['win_label']), vehicle_or_not_index, vehicle_type_index)

    write += write_manual_overall
    with open(folder_name + "report_acc_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f.txt" % (acc_vehicle_or_not, acc_walk_stop, acc_vehicle_type,
                                                                              acc_app_overall, acc_manual_overall),
              'w') as f:
        f.truncate()
        f.write(write)
        f.close()

    print(write)


def train_model(train_win_df):
    features_train = train_win_df.iloc[:, 0:-1]
    labels_train = train_win_df['win_label']

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
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.001)

    model.fit(np.array(features_train), np.array(cat_labels_train), verbose=2, epochs=train_opt['epochs'],
              batch_size=train_opt['batch_size'], class_weight=class_weight, callbacks=[reduce_lr])

    return model


def get_index(features):
    result = []
    for feature in features:
        idx = DL_FEATURES.index(feature)
        for i in range(train_opt['window_size']):
            result.append(idx)
            idx += len(DL_FEATURES)
    return result


if __name__ == '__main__':
    main()
