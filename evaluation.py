import datetime
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import os
import pandas as pd
from Write_Class import Write
import params


evaluation_report = Write("")


def evaluate_single_model(model, folder_name, model_name, features_test, labels_test, save_model=True):
    """
    Evaluate and Test a single model
    :param model: The model to be tested
    :param folder_name: The folder name to save the model
    :param model_name: The name of the model
    :param features_test: The features to be test
    :param labels_test: The ground truth labals of the testing dataset
    :param save_model: boolean, whether to save the model to disk
    :return: predicted labels of the tesing dataset
    """
    cat_labels_test = np_utils.to_categorical(labels_test)
    loss, acc = model.evaluate(features_test, cat_labels_test, verbose=2)
    write = "**********Evaluating " + str(model_name) + "************\n"
    write += 'Testing data size: ' + str(len(labels_test)) + '\n'
    write += str(Counter(labels_test)) + '\n'
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'

    result = model.predict(features_test)
    result_label = np.argmax(result, 1)
    gt_label = labels_test

    con_matrix = confusion_matrix(gt_label, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(gt_label, result_label)) + '\n'

    #  create folder if not exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if save_model:
        model_json = model.to_json()
        with open(folder_name + model_name + "_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(folder_name + model_name + "_model.h5")
        print("Saved model to disk")
        plot_model(model, to_file=folder_name + model_name + "_model.png", show_shapes=True)

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)
    return result_label


def evaluate_overall_app(triplet_model, walk_stop_model, car_bus_model, features_test, labels_test, triplet_idx,
                         walk_stop_idx, car_bus_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using App labelled data, with 5 labels\n"
    triplet_result = triplet_model.predict(features_test[:, triplet_idx])
    triplet_result = np.argmax(triplet_result, 1)

    car_bus_result = car_bus_model.predict(features_test[:, car_bus_idx])
    car_bus_result = np.argmax(car_bus_result, 1)

    walk_stop_result = walk_stop_model.predict(features_test[:, walk_stop_idx])
    walk_stop_result = np.argmax(walk_stop_result, 1)
    result_label = []

    for idx, t in enumerate(triplet_result):
        if t == 0:  # stationary or stop
            result_label.append(walk_stop_result[idx])
        elif t == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[idx] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    return result_label


def evaluate_overall_app_2(vehicle_or_not_model, walk_stop_model, vehicle_type_model, features_test, labels_test,
                           vehicle_or_not_idx, walk_stop_idx, vehicle_type_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using App labelled data, with 5 labels\n"
    vehicle_or_not_result = vehicle_or_not_model.predict(features_test[:, vehicle_or_not_idx])
    vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_result = vehicle_type_model.predict(features_test[:, vehicle_type_idx])
    vehicle_type_result = np.argmax(vehicle_type_result, 1)

    walk_stop_result = walk_stop_model.predict(features_test[:, walk_stop_idx])
    walk_stop_result = np.argmax(walk_stop_result, 1)
    result_label = []

    for idx, t in enumerate(vehicle_or_not_result):
        if t == 0:  # stationary or stop
            result_label.append(walk_stop_result[idx])
        elif t == 1:  # vehicle
            if vehicle_type_result[idx] == 0:
                result_label.append(3)
            elif vehicle_type_result[idx] == 1:
                result_label.append(4)
            elif vehicle_type_result[idx] == 2:
                result_label.append(2)
            else:
                print("Error in overall evaluation: wrong label!" + str(vehicle_type_result[idx]))
        else:
            print("Error in overall evaluation: wrong label! At idx %d, %d" % (idx, t))

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    return result_label


def evaluate_overall_manual(triplet_model, car_bus_model, features_test, labels_test, triplet_idx, car_bus_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    triplet_result = triplet_model.predict(features_test[:, triplet_idx])
    triplet_result = np.argmax(triplet_result, 1)

    car_bus_result = car_bus_model.predict(features_test[:, car_bus_idx])
    car_bus_result = np.argmax(car_bus_result, 1)

    result_label = []

    for idx, t in enumerate(triplet_result):
        if t == 0:  # stationary or stop
            result_label.append(0)
        elif t == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[idx] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    return result_label


def evaluate_overall_manual_2(vehicle_or_not_model, vehicle_type_model, features_test, labels_test, vehicle_or_not_idx,
                              vehicle_type_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    vehicle_or_not_result = vehicle_or_not_model.predict(features_test[:, vehicle_or_not_idx])
    vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_result = vehicle_type_model.predict(features_test[:, vehicle_type_idx])
    vehicle_type_result = np.argmax(vehicle_type_result, 1)

    result_label = []

    for idx, t in enumerate(vehicle_or_not_result):
        if t == 0:  # stationary or stop
            result_label.append(5)
        elif t == 1:  # vehicle
            if vehicle_type_result[idx] == 0:
                result_label.append(2)  # mrt
            elif vehicle_type_result[idx] == 1:
                result_label.append(3)  # bus
            elif vehicle_type_result[idx] == 2:
                result_label.append(4)  # car
            else:
                print("Error in overall evaluation: wrong label!"+vehicle_type_model[idx])
        else:  # t[1] == 2
            print("Error in overall evaluation: wrong label! at idx %d, %d" % (idx, t))

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    return result_label


def save_predicted_result_in_csv(result_labels, df, folder_name, all_features, model_name, label_type):
    """
    Save the predicted result into a csv, with ground truth label, location information and trip_id
    :param result_labels: The predicted label
    :param df: The dataframe containing all the information
    :param folder_name: The folder name to save
    :param all_features: a list of all the features included in df.
    :param model_name: The name of the model evaluated.
    :param label_type: The type of label we are using
    :return: None
    """
    # Save the predicted result into a csv

    # each win label is assigned to the last pt in the win
    lat_idx = all_features.index('WLATITUDE') + len(all_features) * (params.window_size - 1)
    lon_idx = all_features.index('WLONGITUDE') + len(all_features) * (params.window_size - 1)
    lat_lon = df.iloc[:, [lat_idx, lon_idx]]
    lat_lon.columns = ['WLATITUDE', 'WLONGITUDE']

    min_lon_sg = 103.565276
    max_lon_sg = 104
    min_lat_sg = 1.235578
    max_lat_sg = 1.479055
    lat_lon = lat_lon.assign(WLATITUDE=list(map(lambda x: (x * (max_lat_sg - min_lat_sg) + min_lat_sg)*(x != 0),
                                                list(lat_lon['WLATITUDE']))))
    lat_lon = lat_lon.assign(WLONGITUDE=list(map(lambda x: (x * (max_lon_sg - min_lon_sg) + min_lon_sg)*(x != 0),
                                                 list(lat_lon['WLONGITUDE']))))
    lat_lon = lat_lon.replace({0: -1}, regex=True)

    df_to_save = pd.DataFrame(np.array(lat_lon), columns=['WLATITUDE', 'WLONGITUDE'])
    df_to_save['pt_label'] = pd.Series(result_labels)
    df_to_save['gt_label'] = pd.Series(np.array(df[label_type]))
    df_to_save['trip_id'] = pd.Series(np.array(df['trip_id']))

    df_to_save.to_csv(folder_name + model_name + '_test_result_pt.csv')


def init_write(opt, train_opt, features, manual_win_df, app_win_df):
    """
    Initiate the evaluation report:training options, general options, features, distribution of each class in the dataset
    :param opt: general options
    :param train_opt: training options
    :param features: features used for each model
    :param manual_win_df: the window dataframe of manually labelled data
    :param app_win_df: the window dataframe of app labelled data
    :return: None
    """
    write = "Evaluation Report of Train_Hierarchical \n"
    write += str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '\n'
    write += "General option: \n" + str(opt) + "\n"
    write += "Train option: \n" + str(train_opt) + "\n"
    write += "Features: \n" + str(features) + "\n"
    write += "Manual_win_df: " + str(Counter(manual_win_df[opt['label_type']])) + '\n'
    write += "App_win_df: " + str(Counter(app_win_df[opt['label_type']])) + '\n'
    evaluation_report.add_content(write)


def save_write(folder_name):
    """
    Save the evaluation report into a folder
    :param folder_name: The name of folder to save the evaluation report
    :return: None
    """
    evaluation_report.save_write(folder_name)
