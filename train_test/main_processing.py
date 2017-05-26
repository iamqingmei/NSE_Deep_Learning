from train_test import evaluation, train_models
import numpy as np
import logging
from preprocessing import preprocessing
import time


def dl_veh_type(train_df, test_df, opt, vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt, features):
    # train vehicle type model seperately
    evaluation.init_write(opt, [vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt],
                          [features['VEHICLE_OR_NOT_FEATURES'], features['VEHICLE_TYPE']],
                          train_df, test_df)
    print(evaluation.evaluation_report.write)
    if vehicle_type_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_type_train_opt['random_seed'])
    vehicle_type_train_win_df = train_df.copy()
    vehicle_type_test_win_df = test_df.copy()

    vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])

    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df[opt['test_label_type']] == 4)
                                                        | (vehicle_type_test_win_df[opt['test_label_type']] == 3) | (
                                                        vehicle_type_test_win_df[opt['test_label_type']] == 2)]

    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df[opt['train_label_type']] == 2)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 3)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(vehicle_type_train_win_df, vehicle_type_test_win_df,
                                 [[2, 0], [3, 1], [4, 2]])
    # features_test, labels_test = preprocessing.balance_dataset(
    #     vehicle_type_test_win_df.iloc[:, vehicle_type_index],
    #     vehicle_type_test_win_df[opt['test_label_type']])
    logging.info("Start to train vehicle_type model")
    train_models.train_dl_veh_type_model(
        vehicle_type_train_win_df.iloc[:, vehicle_type_index],
        vehicle_type_train_win_df[opt['train_label_type']],
        vehicle_type_train_opt,
        vehicle_type_test_win_df.iloc[:, vehicle_type_index],
        vehicle_type_test_win_df[opt['test_label_type']])


def dl_train_test(train_df, test_df, opt, vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt, features):
    # Use Deep Learning to train hierarchical model
    evaluation.init_write(opt, [vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt],
                          [features['VEHICLE_OR_NOT_FEATURES'], features['VEHICLE_TYPE']],
                          train_df, test_df)
    print(evaluation.evaluation_report.write)
    # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    if vehicle_or_not_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_or_not_train_opt['random_seed'])
    vehicle_or_not_train_win_df = train_df.copy()
    vehicle_or_not_test_win_df = test_df.copy()

    vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'], features['ALL_FEATURES'])
    preprocessing.reassign_label(vehicle_or_not_train_win_df, vehicle_or_not_test_win_df,
                                 [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0]])
    logging.info("Start to train vehicle_or_not model")
    # features_test, labels_test = preprocessing.balance_dataset(
    #     vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index],
    #     vehicle_or_not_test_win_df[opt['test_label_type']])
    vehicle_or_not_model = train_models.train_dl_veh_or_not_model(
        vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index],
        vehicle_or_not_train_win_df[opt['train_label_type']],
        vehicle_or_not_train_opt,
        vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index],
        vehicle_or_not_test_win_df[opt['test_label_type']])
    logging.info("Start to test vehicle_or_not model")
    evaluation.evaluate_single_model(
        vehicle_or_not_model, opt['folder_name'], 'vehicle_or_not',
        np.array(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
        np.array(vehicle_or_not_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~ train vehicle_type_model ~~~~~~~~~~~~~~~~
    if vehicle_type_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_type_train_opt['random_seed'])
    vehicle_type_train_win_df = train_df.copy()
    vehicle_type_test_win_df = test_df.copy()

    vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])

    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df[opt['test_label_type']] == 4)
                                                        | (vehicle_type_test_win_df[opt['test_label_type']] == 3) | (
                                                        vehicle_type_test_win_df[opt['test_label_type']] == 2)]

    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df[opt['train_label_type']] == 2)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 3)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(vehicle_type_train_win_df, vehicle_type_test_win_df,
                                 [[2, 0], [3, 1], [4, 2]])
    # features_test, labels_test = preprocessing.balance_dataset(
    #     vehicle_type_test_win_df.iloc[:, vehicle_type_index],
    #     vehicle_type_test_win_df[opt['test_label_type']])
    logging.info("Start to train vehicle_type model")
    vehicle_type_model = train_models.train_dl_veh_type_model(
        vehicle_type_train_win_df.iloc[:, vehicle_type_index],
        vehicle_type_train_win_df[opt['train_label_type']],
        vehicle_type_train_opt,
        vehicle_type_test_win_df.iloc[:, vehicle_type_index],
        vehicle_type_test_win_df[opt['test_label_type']])

    logging.info("Start to test vehicle_type model")
    if vehicle_type_train_opt['middle_output'] is True:
        evaluation.evaluate_single_ml_model(
            vehicle_type_model,
            np.array(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
            np.array(vehicle_type_test_win_df[opt['test_label_type']]),
            ['mrt', 'bus', 'car'], opt['folder_name'])
    else:
        evaluation.evaluate_single_model(
            vehicle_type_model,
            opt['folder_name'],
            'vehicle_type',
            np.array(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
            np.array(vehicle_type_test_win_df[opt['test_label_type']]),
            save_model=False)
    # print(evaluation.evaluation_report.write)

    # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
    overall_test_win_df = test_df.copy()
    valid_all_based = preprocessing.remove_mix(overall_test_win_df, 'all_based_win_label')
    overall_result_label = evaluation.evaluate_overall_manual_2(
        vehicle_or_not_model, vehicle_type_model, valid_all_based,
        valid_all_based['all_based_win_label'], vehicle_or_not_index,
        vehicle_type_index, opt['smooth_overall_result'])
    overall_result_label = evaluation.evaluate_overall_manual_2(
        vehicle_or_not_model, vehicle_type_model, overall_test_win_df,
        overall_test_win_df['last_based_win_label'], vehicle_or_not_index,
        vehicle_type_index, opt['smooth_overall_result'])
    # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
    evaluation.save_predicted_result_in_csv(overall_result_label, overall_test_win_df, opt['folder_name'],
                                            'overall', opt['test_label_type'])
    evaluation.save_write(opt['folder_name'])

    del overall_test_win_df, vehicle_or_not_test_win_df, vehicle_or_not_train_win_df, vehicle_or_not_index, \
        vehicle_type_model, vehicle_type_train_win_df, vehicle_type_test_win_df, \
        vehicle_type_index


def ml_hie_train_test(ml_opt, train_df, test_df, opt, features):
    # Use ML to train hierarchical model
    evaluation.init_write(opt, None, [features['VEHICLE_OR_NOT_FEATURES'], features['VEHICLE_TYPE']],
                          train_df, test_df)

    # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    vehicle_or_not_train_win_df = train_df.copy()
    vehicle_or_not_test_win_df = test_df.copy()

    vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'], features['ALL_FEATURES'])
    preprocessing.reassign_label(vehicle_or_not_train_win_df, vehicle_or_not_test_win_df,
                                 [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0]])
    logging.info("Start to train vehicle_or_not model")
    vehicle_or_not_model = train_models.train_ml_model(
        np.array(vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index]),
        np.array(vehicle_or_not_train_win_df[opt['train_label_type']]),
        ml_opt)

    logging.info("Start to evaluate vehicle_or_not model")
    evaluation.evaluate_single_ml_model(
        vehicle_or_not_model,
        np.array(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
        np.array(vehicle_or_not_test_win_df[opt['test_label_type']]),
        ['not vehicle', 'vehicle'],
        opt['folder_name'])

    # ~~~~~~~~~~~~~~~~ train vehicle_type_model ~~~~~~~~~~~~~~~~
    vehicle_type_train_win_df = train_df.copy()
    vehicle_type_test_win_df = test_df.copy()

    vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])
    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df[opt['test_label_type']] == 4)
                                                        | (vehicle_type_test_win_df[opt['test_label_type']] == 3) | (
                                                            vehicle_type_test_win_df[opt['test_label_type']] == 2)]

    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df[opt['train_label_type']] == 2)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 3)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(vehicle_type_train_win_df, vehicle_type_test_win_df,
                                 [[2, 0], [3, 1], [4, 2]])
    logging.info("Start to train vehicle_type model")
    vehicle_type_model = train_models.train_ml_model(
        np.array(vehicle_type_train_win_df.iloc[:, vehicle_type_index]),
        np.array(vehicle_type_train_win_df[opt['train_label_type']]),
        ml_opt)
    logging.info("Start to evaluate vehicle_type model")
    evaluation.evaluate_single_ml_model(
        vehicle_type_model,
        np.array(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
        np.array(vehicle_type_test_win_df[opt['test_label_type']]),
        ['mrt', 'bus', 'car'],
        opt['folder_name'])

    # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
    overall_test_win_df = test_df.copy()
    valid_all_based = preprocessing.remove_mix(overall_test_win_df, 'all_based_win_label')
    evaluation.evaluate_overall_manual_2(
        vehicle_or_not_model, vehicle_type_model, valid_all_based, valid_all_based['all_based_win_label'],
        vehicle_or_not_index, vehicle_type_index, opt['smooth_overall_result'])
    evaluation.evaluate_overall_manual_2(
        vehicle_or_not_model, vehicle_type_model, overall_test_win_df, overall_test_win_df['last_based_win_label'],
        vehicle_or_not_index, vehicle_type_index, opt['smooth_overall_result'])

    # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
    # evaluation.save_predicted_result_in_csv(overall_result_label, overall_test_win_df, opt['folder_name'],
    #                                         'overall', opt['test_label_type'])
    evaluation.save_write(opt['folder_name'])
    # print(evaluation.evaluation_report.write)
    del vehicle_or_not_test_win_df, vehicle_or_not_train_win_df, vehicle_type_train_win_df, vehicle_type_test_win_df, \
        vehicle_type_model, overall_test_win_df, vehicle_or_not_model, vehicle_type_index, vehicle_or_not_index


def ml_one_train_test(ml_opt, train_df, test_df, opt, features):
    # Use ml to train All-In-One Model

    evaluation.init_write(opt, None, [features['VEHICLE_OR_NOT_FEATURES'], features['VEHICLE_TYPE']],
                          train_df, test_df)

    # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    one_model_train_win_df = train_df.copy()
    one_model_test_win_df = test_df.copy()

    one_model_index = preprocessing.get_feature_idx(features['ONE_MODEL'], features['ALL_FEATURES'])
    preprocessing.reassign_label(one_model_train_win_df, one_model_test_win_df,
                                 [[1, 0], [2, 1], [3, 2], [4, 3], [5, 0]])
    logging.info("Start to train ml one model")
    start_time = time.time()
    one_model = train_models.train_ml_model(
        np.array(one_model_train_win_df.iloc[:, one_model_index]),
        np.array(one_model_train_win_df[opt['train_label_type']]),
        ml_opt)
    logging.info("Finished to train ml one_model model")
    print("it took", time.time() - start_time, "seconds.")
    logging.info("Start to evaluate ml one model")
    valid_all_based = preprocessing.remove_mix(one_model_test_win_df, 'all_based_win_label')
    evaluation.evaluate_single_ml_model(
        one_model,
        np.array(valid_all_based.iloc[:, one_model_index]),
        np.array(valid_all_based['all_based_win_label']),
        ['not vehicle', 'mrt', 'bus', 'car'],
        opt['folder_name'])
    evaluation.evaluate_single_ml_model(
        one_model,
        np.array(one_model_test_win_df.iloc[:, one_model_index]),
        np.array(one_model_test_win_df['last_based_win_label']),
        ['not vehicle', 'mrt', 'bus', 'car'],
        opt['folder_name'])

    evaluation.save_write(opt['folder_name'])
    # print(evaluation.evaluation_report.write)
    del one_model_test_win_df, one_model_train_win_df, one_model_index, one_model


def dl_train_test_3binary(train_df, test_df, opt, vehicle_or_not_train_opt, vehicle_type_train_opt,
                          bus_or_not_train_opt,
                          mrt_or_car_train_opt, train_test_opt, features):
    # Use dl to train bi-bi-binary model
    evaluation.init_write(opt, [vehicle_or_not_train_opt, vehicle_type_train_opt, bus_or_not_train_opt,
                                mrt_or_car_train_opt, train_test_opt],
                          [features['VEHICLE_OR_NOT_FEATURES'], features['BUS_OR_NOT'], features['MRT_OR_CAR']],
                          train_df, test_df)

    # # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    if vehicle_or_not_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_or_not_train_opt['random_seed'])
    vehicle_or_not_train_win_df = train_df.copy()
    vehicle_or_not_test_win_df = test_df.copy()

    vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'], features['ALL_FEATURES'])
    preprocessing.reassign_label(vehicle_or_not_train_win_df, vehicle_or_not_test_win_df,
                                 [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0]])
    logging.info("Start to train vehicle_or_not model")
    # features_test, labels_test = preprocessing.balance_dataset(
    #     vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index],
    #     vehicle_or_not_test_win_df[opt['test_label_type']])
    vehicle_or_not_model = train_models.train_dl_veh_or_not_model(
        vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index],
        vehicle_or_not_train_win_df[opt['train_label_type']],
        vehicle_or_not_train_opt,
        vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index],
        vehicle_or_not_test_win_df[opt['test_label_type']])
    logging.info("Start to test vehicle_or_not model")
    evaluation.evaluate_single_model(
        vehicle_or_not_model, opt['folder_name'], 'vehicle_or_not',
        np.array(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
        np.array(vehicle_or_not_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~ train bus_or_not_model ~~~~~~~~~~~~~~~~~~
    if bus_or_not_train_opt['random_seed'] is not None:
        np.random.seed(bus_or_not_train_opt['random_seed'])

    bus_or_not_train_win_df = train_df.copy()
    bus_or_not_test_win_df = test_df.copy()

    bus_or_not_index = preprocessing.get_feature_idx(features['BUS_OR_NOT'], features['ALL_FEATURES'])

    bus_or_not_test_win_df = bus_or_not_test_win_df[(bus_or_not_test_win_df[opt['test_label_type']] == 4) |
                                                    (bus_or_not_test_win_df[opt['test_label_type']] == 3) |
                                                    (bus_or_not_test_win_df[opt['test_label_type']] == 2)]

    bus_or_not_train_win_df = bus_or_not_train_win_df[(bus_or_not_train_win_df[opt['train_label_type']] == 2)
                                                      | (bus_or_not_train_win_df[opt['train_label_type']] == 3)
                                                      | (bus_or_not_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(bus_or_not_train_win_df, bus_or_not_test_win_df,
                                 [[2, 0], [3, 1], [4, 0]])
    logging.info("Start to train bus_or_not model")
    # features_test, labels_test = preprocessing.balance_dataset(
    #     bus_or_not_test_win_df.iloc[:, bus_or_not_index],
    #     bus_or_not_test_win_df[opt['test_label_type']])
    bus_or_not_model = train_models.train_dl_veh_type_model(
        bus_or_not_train_win_df.iloc[:, bus_or_not_index],
        bus_or_not_train_win_df[opt['train_label_type']],
        bus_or_not_train_opt,
        bus_or_not_test_win_df.iloc[:, bus_or_not_index],
        bus_or_not_test_win_df[opt['test_label_type']])

    logging.info("Start to test bus_or_not model")

    evaluation.evaluate_single_model(
        bus_or_not_model, opt['folder_name'], 'bus_or_not',
        np.array(bus_or_not_test_win_df.iloc[:, bus_or_not_index]),
        np.array(bus_or_not_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~ train mrt_or_car_model ~~~~~~~~~~~~~~~~
    if mrt_or_car_train_opt['random_seed'] is not None:
        np.random.seed(mrt_or_car_train_opt['random_seed'])

    mrt_or_car_train_win_df = train_df.copy()
    mrt_or_car_test_win_df = test_df.copy()

    mrt_or_car_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])

    mrt_or_car_test_win_df = mrt_or_car_test_win_df[(mrt_or_car_test_win_df[opt['test_label_type']] == 4) |
                                                    (mrt_or_car_test_win_df[opt['test_label_type']] == 2)]

    mrt_or_car_train_win_df = mrt_or_car_train_win_df[(mrt_or_car_train_win_df[opt['train_label_type']] == 2) |
                                                      (mrt_or_car_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(mrt_or_car_train_win_df, mrt_or_car_test_win_df,
                                 [[2, 0], [4, 1]])
    logging.info("Start to train mrt_or_car model")
    # features_test, labels_test = preprocessing.balance_dataset(
    #     mrt_or_car_test_win_df.iloc[:, mrt_or_car_index],
    #     mrt_or_car_test_win_df[opt['test_label_type']])
    mrt_or_car_model = train_models.train_dl_veh_type_model(
        mrt_or_car_train_win_df.iloc[:, mrt_or_car_index],
        mrt_or_car_train_win_df[opt['train_label_type']],
        mrt_or_car_train_opt,
        mrt_or_car_test_win_df.iloc[:, mrt_or_car_index],
        mrt_or_car_test_win_df[opt['test_label_type']])

    logging.info("Start to test mrt_or_car model")
    evaluation.evaluate_single_model(
        mrt_or_car_model, opt['folder_name'], 'mrt_or_car',
        np.array(mrt_or_car_test_win_df.iloc[:, mrt_or_car_index]),
        np.array(mrt_or_car_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
    overall_test_win_df = test_df.copy()
    overall_result_label = evaluation.evaluate_overall_bibibinary(
        vehicle_or_not_model, bus_or_not_model, mrt_or_car_model, overall_test_win_df,
        overall_test_win_df[opt['test_label_type']],
        vehicle_or_not_index, bus_or_not_index, mrt_or_car_index,
        opt['smooth_overall_result'])
    # # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
    evaluation.save_predicted_result_in_csv(overall_result_label, overall_test_win_df, opt['folder_name'],
                                            'overall', opt['test_label_type'])
    evaluation.save_write(opt['folder_name'])


def dl_one_model_train_test(train_df, test_df, opt, one_model_train_opt, train_test_opt, features):
    # Use dl to train All-In-One model
    evaluation.init_write(opt, [one_model_train_opt, train_test_opt], features['ONE_MODEL'],
                          train_df, test_df)

    # # ~~~~~~~~~~~~~~ train one_model model ~~~~~~~~~~~~~~~~
    if one_model_train_opt['random_seed'] is not None:
        np.random.seed(one_model_train_opt['random_seed'])

    one_model_train_win_df = train_df.copy()
    one_model_test_win_df = test_df.copy()

    one_model_index = preprocessing.get_feature_idx(features['ONE_MODEL'], features['ALL_FEATURES'])
    preprocessing.reassign_label(one_model_train_win_df, one_model_test_win_df,
                                 [[1, 0], [2, 1], [3, 2], [4, 3], [5, 0]])
    logging.info("Start to train one_model model")
    start_time = time.time()
    # features_test, labels_test = preprocessing.balance_dataset(
    #     one_model_test_win_df.iloc[:, one_model_index],
    #     one_model_test_win_df[opt['test_label_type']])

    one_model_model = train_models.train_dl_veh_type_model(
        one_model_train_win_df.iloc[:, one_model_index],
        one_model_train_win_df[opt['train_label_type']],
        one_model_train_opt,
        one_model_test_win_df.iloc[:, one_model_index],
        one_model_test_win_df[opt['test_label_type']])
    logging.info("Finished to train one_model model")
    print("it took", time.time() - start_time, "seconds.")
    logging.info("Start to test one_model model")
    if one_model_train_opt['DLNetwork'] == 'LSTM':
        features_test = np.reshape(np.array(one_model_test_win_df.iloc[:, one_model_index]),
                                   (len(one_model_test_win_df.iloc[:, one_model_index]),
                                    6,
                                    len(features['ONE_MODEL'])))
        evaluation.evaluate_single_model(
            one_model_model, opt['folder_name'], 'one_model',
            features_test,
            np.array(one_model_test_win_df[opt['test_label_type']]), save_model=False)
    else:
        valid_all_based = preprocessing.remove_mix(one_model_test_win_df, 'all_based_win_label')
        evaluation.evaluate_single_model(
            one_model_model, opt['folder_name'], 'one_model',
            np.array(valid_all_based.iloc[:, one_model_index]),
            np.array(valid_all_based['all_based_win_label']), save_model=False)
        evaluation.evaluate_single_model(
            one_model_model, opt['folder_name'], 'one_model',
            np.array(one_model_test_win_df.iloc[:, one_model_index]),
            np.array(one_model_test_win_df['last_based_win_label']), save_model=False)

    evaluation.save_write(opt['folder_name'])


def lstm_train_test(train_df, test_df, opt, vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt, features):
    # Use lstm net to train hierarchical model
    evaluation.init_write(opt, [vehicle_or_not_train_opt, vehicle_type_train_opt, train_test_opt],
                          [features['VEHICLE_OR_NOT_FEATURES'], features['VEHICLE_TYPE']],
                          train_df, test_df)
    print(evaluation.evaluation_report.write)
    # # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    if vehicle_or_not_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_or_not_train_opt['random_seed'])
    vehicle_or_not_train_win_df = train_df.copy()
    vehicle_or_not_test_win_df = test_df.copy()

    vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'], features['ALL_FEATURES'])
    preprocessing.reassign_label(vehicle_or_not_train_win_df, vehicle_or_not_test_win_df,
                                 [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0]])
    logging.info("Start to train vehicle_or_not model")
    vehicle_or_not_model = train_models.train_lstm_model(vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index],
                                                         vehicle_or_not_train_win_df[opt['train_label_type']],
                                                         vehicle_or_not_train_opt)
    logging.info("Start to test vehicle_or_not model")
    features_test = np.reshape(np.array(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
                               (len(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
                                6,
                                len(features['VEHICLE_OR_NOT_FEATURES'])))
    evaluation.evaluate_single_model(
        vehicle_or_not_model, opt['folder_name'], 'vehicle_or_not',
        features_test,
        np.array(vehicle_or_not_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~ train vehicle_type_model ~~~~~~~~~~~~~~~~
    if vehicle_type_train_opt['random_seed'] is not None:
        np.random.seed(vehicle_type_train_opt['random_seed'])
    vehicle_type_train_win_df = train_df.copy()
    vehicle_type_test_win_df = test_df.copy()

    vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])

    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df[opt['test_label_type']] == 4)
                                                        | (vehicle_type_test_win_df[opt['test_label_type']] == 3) | (
                                                        vehicle_type_test_win_df[opt['test_label_type']] == 2)]

    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df[opt['train_label_type']] == 2)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 3)
                                                          | (vehicle_type_train_win_df[opt['train_label_type']] == 4)]

    preprocessing.reassign_label(vehicle_type_train_win_df, vehicle_type_test_win_df,
                                 [[2, 0], [3, 1], [4, 2]])
    logging.info("Start to train vehicle_type model")
    vehicle_type_model = train_models.train_lstm_model(vehicle_type_train_win_df.iloc[:, vehicle_type_index],
                                                       vehicle_type_train_win_df[opt['train_label_type']],
                                                       vehicle_type_train_opt)

    logging.info("Start to test vehicle_type model")
    features_test = np.reshape(np.array(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
                               (len(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
                                6,
                                len(features['VEHICLE_TYPE'])))
    evaluation.evaluate_single_model(
        vehicle_type_model, opt['folder_name'], 'vehicle_type',
        features_test,
        np.array(vehicle_type_test_win_df[opt['test_label_type']]), save_model=False)

    # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
    overall_test_win_df = test_df.copy()
    overall_result_label = evaluation.evaluate_overall_lstm(
        vehicle_or_not_model, vehicle_type_model, overall_test_win_df,
        overall_test_win_df[opt['test_label_type']], vehicle_or_not_index,
        vehicle_type_index, opt['smooth_overall_result'])
    # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
    evaluation.save_predicted_result_in_csv(overall_result_label, overall_test_win_df, opt['folder_name'],
                                            'overall', opt['test_label_type'])
    evaluation.save_write(opt['folder_name'])

    del overall_test_win_df, vehicle_or_not_test_win_df, vehicle_or_not_train_win_df, vehicle_or_not_index, \
        vehicle_type_model, vehicle_or_not_model, vehicle_type_train_win_df, vehicle_type_test_win_df, \
        vehicle_type_index
