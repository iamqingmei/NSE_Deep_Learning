import datetime
import numpy as np
from feature_calc import DL_FEATURES, WIN_FEATURES
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import os


def evaluation_report(model, label_type, features_test, labels_test, train_length, train_opt_dict, result_vs_gt=True,
                      save_txt=True, save_model=True, remarks=None):
    loss, acc = model.evaluate(features_test, labels_test, verbose=2)
    write = str(datetime.datetime.now()) + '\n'
    write += str(train_opt_dict) + '\n'
    write += 'PT FEATURES included: ' + str(DL_FEATURES) + '\n'
    write += 'WIN FEATURES included: ' + str(WIN_FEATURES) + '\n'
    write += 'Training data size: ' + str(train_length) + '\n'
    write += 'Testing data size: ' + str(len(labels_test)) + '\n'
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'
    # write += model.summary().__str__()
    result = model.predict(features_test)
    result_label = np.argmax(result, 1)
    gt_label = np.argmax(labels_test, 1)

    z = zip(result_label, gt_label)

    if result_vs_gt:
        write += str(z)

    con_matrix = confusion_matrix(gt_label, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(gt_label, result_label)) + '\n'

    if remarks is not None:
        write += '\n' + 'Remarks:' + str(remarks)
    folder_name = './' + str(label_type) + '/evaluation_report/' + \
                  str(datetime.datetime.now().strftime("%y-%m-%d %H-%M")) + '_' + str(train_opt_dict['label_number']) +\
                  'l_acc' + str("%.2f" % round(acc, 2)) + '/'
    #  create folder if not exists
    if save_txt or save_model:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    if save_txt:
        file_name = str(train_opt_dict['label_number']) + 'labels_' + str("%.2f" % round(acc, 2)) + 'acc.txt'
        f = open(folder_name + file_name, 'w')
        f.truncate()
        f.write(write)
        f.close()

    if save_model:
        model_json = model.to_json()
        with open(folder_name + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(folder_name + "model.h5")
        print("Saved model to disk")
        plot_model(model, to_file=folder_name + 'model.png', show_shapes=True)

    return write


def evaluation_four_result(folder_name, app_model, manual_model, app_features_test, app_labels_test,
                           manual_features_test, manual_labels_test, app_win_df, manual_win_df):
    # with tf.Graph().as_default():

    write = ''
    #  create folder if not exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # save app_model
    with open(folder_name + "app_model.json", "w") as json_file:
        json_file.write(app_model.to_json())
    app_model.save_weights(folder_name + "app_model.h5")
    print("Saved app_model to disk")
    plot_model(app_model, to_file=folder_name + 'app_model.png', show_shapes=True)

    # save manual_model
    with open(folder_name + "manual_model.json", "w") as json_file:
        json_file.write(manual_model.to_json())
        manual_model.save_weights(folder_name + "manual_model.h5")
    print("Saved manual_model to disk")
    plot_model(manual_model, to_file=folder_name + 'manual_model.png', show_shapes=True)

    # evaluate app_model
    loss, acc = app_model.evaluate(app_features_test, app_labels_test, verbose=2)
    result = app_model.predict(app_features_test)
    result_label = np.argmax(result, 1)
    gt_label = np.argmax(app_labels_test, 1)
    write += "\nuse app_test to test app_model: \n"
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'
    write += str(confusion_matrix(gt_label, result_label)) + '\n'
    target_names = ['stationary', 'walking', 'mrt', 'bus', 'car']
    write += "Classification report:\n"
    write += str((classification_report(gt_label, result_label, target_names=target_names))) + '\n'

    # evaluate manual_model
    loss, acc = manual_model.evaluate(manual_features_test, manual_labels_test, verbose=2)
    result = manual_model.predict(manual_features_test)
    result_label = np.argmax(result, 1)
    gt_label = np.argmax(manual_labels_test, 1)
    write += "\nuse manual_test to test manual_model \n"
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'
    write += str(confusion_matrix(gt_label, result_label)) + '\n'
    target_names = ['stationary/walking', 'mrt', 'bus', 'car']
    write += "Classification report:\n"
    write += str((classification_report(gt_label, result_label, target_names=target_names))) + '\n'

    # evaluate app_model using manual data
    manual_features = manual_win_df.iloc[:, 0:-1]
    manual_labels = manual_win_df['win_label']

    manual_labels[manual_labels == 3.0] = 4  # car
    manual_labels[manual_labels == 2.0] = 3  # bus
    manual_labels[manual_labels == 1.0] = 2  # mrt
    manual_labels = np_utils.to_categorical(manual_labels)

    loss, acc = app_model.evaluate(np.array(manual_features), np.array(manual_labels), verbose=2)
    result = app_model.predict(np.array(manual_features))
    result_label = np.argmax(result, 1)
    for i in range(len(result_label)):
        if result_label[i] == 1:
            result_label[i] = 0
    gt_label = np.argmax(np.array(manual_labels), 1)
    count = 0
    for i in range(len(result_label)):
        if result_label[i] == gt_label[i]:
            count += 1
    acc = float(count)/len(result_label)
    write += "\nuse manual data test app_model: \n"
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'
    write += str(confusion_matrix(gt_label, result_label)) + '\n'
    target_names = ['stationary/walking', 'mrt', 'bus', 'car']
    write += "Classification report:\n"
    write += str((classification_report(gt_label, result_label, target_names=target_names))) + '\n'

    # evaluate manual_model using app data
    app_features = app_win_df.iloc[:, 0:-1]
    app_labels = app_win_df['win_label']

    app_labels[app_labels == 1.0] = 0  # walk or stationary
    app_labels[app_labels == 2.0] = 1  # mrt
    app_labels[app_labels == 3.0] = 2  # bus
    app_labels[app_labels == 4.0] = 3  # car
    app_labels = np_utils.to_categorical(app_labels)

    loss, acc = manual_model.evaluate(np.array(app_features), np.array(app_labels), verbose=2)
    result = manual_model.predict(np.array(app_features))
    result_label = np.argmax(result, 1)
    gt_label = np.argmax(np.array(app_labels), 1)
    write += "\nuse app data test manual_model: \n"
    write += 'loss: ' + str(loss) + ' acc' + str("%.2f" % round(acc, 4)) + '\n'
    write += str(confusion_matrix(gt_label, result_label)) + '\n'
    target_names = ['stationary/walking', 'mrt', 'bus', 'car']
    write += "Classification report:\n"
    write += str((classification_report(gt_label, result_label, target_names=target_names))) + '\n'

    return write


def evaluate_single_model(model, folder_name, model_name, features_test, labels_test, save_model=True):
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
        #plot_model(model, to_file=folder_name + model_name + "_model.png", show_shapes=True)

    return write, acc


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

    for t in list(enumerate(triplet_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(walk_stop_result[t[0]])
        elif t[1] == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[t[0]] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write, acc

def evaluate_overall_app_2(vehicle_or_not_model, walk_stop_model, vehicle_type_model, features_test, labels_test, vehicle_or_not_idx,
                        walk_stop_idx, vehicle_type_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using App labelled data, with 5 labels\n"
    vehicle_or_not_result = vehicle_or_not_model.predict(features_test[:, vehicle_or_not_idx])
    vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_result = vehicle_type_model.predict(features_test[:, vehicle_type_idx])
    vehicle_type_result = np.argmax(vehicle_type_result, 1)

    walk_stop_result = walk_stop_model.predict(features_test[:, walk_stop_idx])
    walk_stop_result = np.argmax(walk_stop_result, 1)
    result_label = []

    for t in list(enumerate(vehicle_or_not_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(walk_stop_result[t[0]])
        elif t[1] == 1:  # vehicle
            if vehicle_type_result[t[0]] == 0:
                result_label.append(3)
            elif vehicle_type_result[t[0]] == 1:
                result_label.append(4)
            elif vehicle_type_result[t[0]] == 2:
                result_label.append(2)
            else:
                print("Error in overall evaluation: wrong label!"+vehicle_type_result[t[0]] )
        else:
            print("Error in overall evaluation: wrong label!"+t[1])

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write, acc


def evaluate_overall_manual(triplet_model, car_bus_model, features_test, labels_test, triplet_idx, car_bus_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    triplet_result = triplet_model.predict(features_test[:, triplet_idx])
    triplet_result = np.argmax(triplet_result, 1)

    car_bus_result = car_bus_model.predict(features_test[:, car_bus_idx])
    car_bus_result = np.argmax(car_bus_result, 1)

    result_label = []

    for t in list(enumerate(triplet_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(0)
        elif t[1] == 1:  # mrt
            result_label.append(2)
        else:  # t[1] == 2
            if car_bus_result[t[0]] == 0:
                result_label.append(3)
            else:
                result_label.append(4)

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write, acc

def evaluate_overall_manual_2(vehicle_or_not_model, vehicle_type_model, features_test, labels_test, vehicle_or_not_idx, vehicle_type_idx):
    write = "**********Evaluate Overall Result**********\n"
    write += "Using manual labelled data, with 4 labels\n"
    vehicle_or_not_result = vehicle_or_not_model.predict(features_test[:, vehicle_or_not_idx])
    vehicle_or_not_result = np.argmax(vehicle_or_not_result, 1)

    vehicle_type_result = vehicle_type_model.predict(features_test[:, vehicle_type_idx])
    vehicle_type_result = np.argmax(vehicle_type_result, 1)

    result_label = []

    for t in list(enumerate(vehicle_or_not_result)):
        if t[1] == 0:  # stationary or stop
            result_label.append(0)
        elif t[1] == 1:  # vehicle
            if vehicle_type_result[t[0]] == 0:
                result_label.append(3)
            elif vehicle_type_result[t[0]] == 1:
                result_label.append(4)
            elif vehicle_type_result[t[0]] == 2:
                result_label.append(2)
            else:
                print("Error in overall evaluation: wrong label!"+vehicle_type_model[t[0]])
        else:  # t[1] == 2
            print("Error in overall evaluation: wrong label!"+t[1])

    con_matrix = confusion_matrix(labels_test, result_label)
    acc = accuracy_score(labels_test, result_label)
    write += str(con_matrix) + '\n'
    write += "Classification report:\n"
    write += str(classification_report(labels_test, result_label)) + '\n'

    return write, acc


