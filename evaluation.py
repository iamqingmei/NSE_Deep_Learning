import datetime
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import xgboost as xgb
from util import chunks
from collections import Counter
import os
import pandas as pd
from Write_Class import Write


evaluation_report = Write('')


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

        del model_json

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    del cat_labels_test, loss, acc, write, con_matrix, gt_label, result
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
                              vehicle_type_idx, if_smooth=True):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    vehicle_or_not_test = np.array(np.array(features_test.iloc[:, vehicle_or_not_idx]))
    vehicle_or_not_result = vehicle_or_not_model.predict(vehicle_or_not_test)
    if len(np.shape(vehicle_or_not_result)) > 1:
        vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_test = np.array(features_test.iloc[:, vehicle_type_idx])
    vehicle_type_result = vehicle_type_model.predict(vehicle_type_test)
    if len(np.shape(vehicle_type_result)) > 1:
        vehicle_type_result = np.argmax(vehicle_type_result, 1)

    result_label = []
    trip_chunks = list(chunks(features_test['trip_id'].tolist()))
    if if_smooth is True:
        write += "Smoooooooothing vehicle_or_not_result~~~~~~~~\n"
        for trip_chunk in trip_chunks:
            vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]] = \
                smooth_is_vehicle(features_test.iloc[trip_chunk[0]:trip_chunk[1]],
                                  vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]])

    # is_vehicle_smoothing()
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
    if if_smooth is True:
        write += "Smoooooooothing smooth_vehicle_type~~~~~~~~\n"
        for trip_chunk in trip_chunks:
            result_label[trip_chunk[0]:trip_chunk[1]] = \
                smooth_vehicle_type(features_test.iloc[trip_chunk[0]:trip_chunk[1]],
                                    result_label[trip_chunk[0]:trip_chunk[1]])

    write += str(Counter(labels_test)) + '\n'
    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    del write, vehicle_or_not_test, vehicle_or_not_result, vehicle_type_test, vehicle_type_result, trip_chunks, \
        con_matrix, acc

    return result_label


def evaluate_overall_lstm(vehicle_or_not_model, vehicle_type_model, features_test, labels_test, vehicle_or_not_idx,
                              vehicle_type_idx, if_smooth = True):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    vehicle_or_not_test = np.reshape(np.array(features_test.iloc[:, vehicle_or_not_idx]),
                                     (len(features_test),
                                      6,
                                      int(len(vehicle_or_not_idx)/6)))
    vehicle_or_not_result = vehicle_or_not_model.predict(vehicle_or_not_test)
    if len(np.shape(vehicle_or_not_result)) > 1:
        vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_test = np.reshape(np.array(features_test.iloc[:, vehicle_type_idx]),
                                   (len(features_test),
                                    6,
                                    int(len(vehicle_type_idx) / 6)))
    vehicle_type_result = vehicle_type_model.predict(vehicle_type_test)
    if len(np.shape(vehicle_type_result)) > 1:
        vehicle_type_result = np.argmax(vehicle_type_result, 1)

    result_label = []
    trip_chunks = list(chunks(features_test['trip_id'].tolist()))
    if if_smooth is True:
        write += "Smoooooooothing vehicle_or_not_result~~~~~~~~\n"
        for trip_chunk in trip_chunks:
            vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]] = \
                smooth_is_vehicle(features_test.iloc[trip_chunk[0]:trip_chunk[1]],
                                  vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]])

    # is_vehicle_smoothing()
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
    if if_smooth is True:
        write += "Smoooooooothing smooth_vehicle_type~~~~~~~~\n"
        for trip_chunk in trip_chunks:
            result_label[trip_chunk[0]:trip_chunk[1]] = \
                smooth_vehicle_type(features_test.iloc[trip_chunk[0]:trip_chunk[1]],
                                    result_label[trip_chunk[0]:trip_chunk[1]])

    write += str(Counter(labels_test)) + '\n'
    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    del write, vehicle_or_not_test, vehicle_or_not_result, vehicle_type_test, vehicle_type_result, trip_chunks, \
        con_matrix, acc

    return result_label


def evaluate_overall_bibibinary(vehicle_or_not_model, bus_or_not_model, mrt_or_car_model,
                                features_test, labels_test, vehicle_or_not_idx,
                                bus_or_not_idx, mrt_or_car_idx, if_smooth=True):
    write = "**********Evaluate Overall Result**********\n"
    write += "with 4 labels\n"
    vehicle_or_not_result = vehicle_or_not_model.predict(np.array(features_test.iloc[:, vehicle_or_not_idx]))
    if len(np.shape(vehicle_or_not_result)) > 1:
        vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    bus_or_not_result = bus_or_not_model.predict(np.array(features_test.iloc[:, bus_or_not_idx]))
    if len(np.shape(bus_or_not_result)) > 1:
        bus_or_not_result = np.argmax(bus_or_not_result, 1)

    mrt_or_car_result = mrt_or_car_model.predict(np.array(features_test.iloc[:, mrt_or_car_idx]))
    if len(np.shape(mrt_or_car_result)) > 1:
        mrt_or_car_result = np.argmax(mrt_or_car_result, 1)

    result_label = []
    trip_chunks = list(chunks(features_test['trip_id'].tolist()))
    if if_smooth is True:
        for trip_chunk in trip_chunks:
            vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]] = \
                smooth_is_vehicle(features_test.iloc[trip_chunk[0]:trip_chunk[1]],
                                  vehicle_or_not_result[trip_chunk[0]:trip_chunk[1]])

    for idx, t in enumerate(vehicle_or_not_result):
        if t == 0:  # stationary or stop
            result_label.append(5)
        elif t == 1:  # vehicle
            if bus_or_not_result[idx] == 0:
                if mrt_or_car_result[idx] == 0:
                    result_label.append(2)  # mrt
                elif mrt_or_car_result[idx] == 1:
                    result_label.append(4)  # car
            elif bus_or_not_result[idx] == 1:
                result_label.append(3)  # bus
            else:
                print("Error in overall evaluation: wrong label! at idx: %d" % idx)
        else:  # t[1] == 2
            print("Error in overall evaluation: wrong label! at idx %d, %d" % (idx, t))

    write += str(Counter(labels_test)) + '\n'
    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    del write, vehicle_or_not_result, bus_or_not_result, mrt_or_car_result, trip_chunks, con_matrix, acc

    return result_label


def save_predicted_result_in_csv(result_labels, df, folder_name, model_name, label_type):
    """
    Save the predicted result into a csv, with ground truth label, location information and trip_id
    :param result_labels: The predicted label
    :param df: The dataframe containing all the information
    :param folder_name: The folder name to save
    :param model_name: The name of the model evaluated.
    :param label_type: The type of label we are using
    :return: None
    """
    # Save the predicted result into a csv

    df_to_save = df[['WLATITUDE', 'WLONGITUDE']].copy()
    df_to_save.index = list(range(len(df_to_save)))
    df_to_save['pt_label'] = pd.Series(result_labels, index=df_to_save.index)
    df_to_save['gt_label'] = pd.Series(np.array(df[label_type]), index=df_to_save.index)
    df_to_save['trip_id'] = pd.Series(np.array(df['trip_id']), index=df_to_save.index)

    df_to_save.to_csv(folder_name + model_name + '_test_result_pt.csv')

    del df_to_save


def init_write(opt, train_opt, features, train_df=None, test_df=None):
    """
    Initiate the evaluation report:training options, general options, features, distribution of each class in the
    dataset
    :param opt: general options
    :param train_opt: training options
    :param features: features used for each model
    :param train_df: the window dataframe of manually labelled data
    :param test_df: the window dataframe of app labelled data
    :return: None
    """
    write = "Evaluation Report of Train_Hierarchical \n"
    write += str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '\n'
    write += "General option: \n" + str(opt) + "\n"
    write += "Train option: \n" + str(train_opt) + "\n"
    write += "Features: \n" + str(features) + "\n"
    if train_df is not None:
        write += "Train_df size: " + str(len(train_df)) + '\n'
        write += "Train_df[all_based_win_label]: " + str(Counter(train_df['all_based_win_label'])) + '\n'
        write += "Train_df[last_based_win_label]: " + str(Counter(train_df['last_based_win_label'])) + '\n'
        write += "Test_df size: " + str(len(test_df)) + '\n'
        write += "Test_df[all_based_win_label]: " + str(Counter(test_df['all_based_win_label'])) + '\n'
        write += "Test_df[last_based_win_label]: " + str(Counter(test_df['last_based_win_label'])) + '\n'
    evaluation_report.add_content(write)

    del write


def save_write(folder_name):
    """
    Save the evaluation report into a folder
    :param folder_name: The name of folder to save the evaluation report
    :return: None
    """
    evaluation_report.save_write(folder_name)
    evaluation_report.clear_content()


def evaluate_single_ml_model(clf, features_test, labels_test, target_names, folder_name):
    write = ''
    # use cross validation to test accuracy
    # write += "***** Cross Validation *****\n"
    # scores = cross_val_score(clf, features_train.tolist(), labels_train.tolist(), cv=10)
    # write += "All scores:\n"
    # write += str(scores) + '\n'
    write += str(clf) + '\n'
    "***** Training & Testing *****"
    # features_test = xgb.DMatrix(features_test)
    result_labels = clf.predict(features_test)
    result_labels = np.array(result_labels)
    con_matx = confusion_matrix(labels_test, result_labels)
    write += "Confusion matrix:\n"
    write += str(con_matx) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_labels, target_names=target_names)) + '\n'
    acc = accuracy_score(labels_test, result_labels)
    # show the importance of each feature
    # write += "Importance of each feature:\n"
    #
    # write += str(clf.feature_importances_) + '\n'

    #  create folder if not exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    evaluation_report.add_content(write)
    evaluation_report.add_accuracy(acc)

    del write, con_matx, acc
    return result_labels


def smooth_is_vehicle(df_trip, vehicle_or_not,  non_vehi_seg_min_dura=1 * 60, vehi_seg_min_dura=2.5 * 60):
    vehicle_or_not_chunks = list(chunks(vehicle_or_not))
    dt_all = df_trip['TIME_DELTA']
    vehicle_or_not_smoothed = vehicle_or_not
    # remove short non-vehicle segments between vehicle segments
    num_chunks = len(vehicle_or_not_chunks)
    for idx, chunk in enumerate(vehicle_or_not_chunks):
        if idx != 0 and idx != num_chunks - 1 and vehicle_or_not[chunk[0]] == 0:
            chunk_dura = sum(dt_all[chunk[0]:chunk[1]])
            if chunk_dura < non_vehi_seg_min_dura:
                vehicle_or_not_smoothed[chunk[0]:chunk[1]] = [1] * (chunk[1] - chunk[0])
    # remove vehicle segments which are still short after combining
    is_vehicle_chunks = list(chunks(vehicle_or_not_smoothed))
    for chunk in is_vehicle_chunks:
        if vehicle_or_not[chunk[0]] == 1:
            chunk_dura = sum(dt_all[chunk[0]:chunk[1]])
            if chunk_dura < vehi_seg_min_dura:
                vehicle_or_not_smoothed[chunk[0]:chunk[1]] = [0] * (chunk[1] - chunk[0])

    del vehicle_or_not_chunks, dt_all, num_chunks, is_vehicle_chunks
    return vehicle_or_not_smoothed


def smooth_vehicle_type(df_trip, original_result_label, vehi_seg_min_dura=1 * 60):
    trip_segments = list(chunks(original_result_label, True))
    dt_all = df_trip['TIME_DELTA']
    res = original_result_label

    num_chunks = len(trip_segments)

    for idx, chunk in enumerate(trip_segments):
        if idx == 0 or idx == num_chunks-1 or chunk[2] == 5:
            continue
        pre_label = trip_segments[idx-1][2]
        post_label = trip_segments[idx + 1][2]
        if pre_label == post_label:
            if pre_label == 5:
                continue
            else:
                cur_chunk_duration = sum(dt_all[chunk[0]:chunk[1]])
                if cur_chunk_duration < vehi_seg_min_dura:
                    res[chunk[0]:chunk[1]] = [pre_label] * (chunk[1] - chunk[0])

    del trip_segments, dt_all, num_chunks
    return res
