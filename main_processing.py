import numpy as np
np.random.seed(0)
import preprocessing
import train_dl_models
import datetime
import evaluation
import logging
import sys

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)

opt = {'label_type': 'last_based_win_label',
       'test_select_type': 'window',
       'test_ratio': float(1/3),
       'remark': 'random seed is 0',
       'folder_name': './evaluation_report/' + str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '/',
       'get_win_df_from_csv:': True}

train_opt = {'batch_size': 32, 'epochs': 200, 'DLNetwork': 'FULL', 'l2': 0.00001}

features = {'ALL_FEATURES': ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'METRO_DIST', 'BUS_DIST', 'NUM_AP',
                             'STD_VELOCITY_10MIN', 'STOP_BUSSTOP_10MIN','TIME_DELTA', 'VELOCITY', 'WLATITUDE',
                             'WLONGITUDE'],
            'VEHICLE_OR_NOT_FEATURES': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'VELOCITY'],
            'VEHICLE_TYPE': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'VELOCITY']}


def main():
    if opt['get_win_df_from_csv:']:
        logging.info("Fetch win_df directly from csv")
        app_win_df, manual_win_df = preprocessing.get_win_df_from_csv()
    else:
        logging.info("Creating All App Trips")
        app_pt_trip_list = preprocessing.get_app_trip_df_list(features['ALL_FEATURES'])

        logging.info("Generating the window dataframes from all app trips")
        app_win_df = preprocessing.get_win_df(app_pt_trip_list, features['ALL_FEATURES'])
        for x in app_pt_trip_list:
            x.save_trip_object()
            del x

        logging.info("Creating All Manual Trips")
        manual_pt_trip_list = preprocessing.get_manual_trip_df_list(features['ALL_FEATURES'])

        logging.info("Generating the window dataframe from manual app trips")
        manual_win_df = preprocessing.get_win_df(manual_pt_trip_list, features['ALL_FEATURES'])

        for x in manual_pt_trip_list:
            x.save_trip_object()
            del x
        app_win_df.to_csv('./data/app/app_win_df.csv')
        manual_win_df.to_csv('./data/manual/manual_win_df.csv')

    # logging.info("Generating app_train_win_df, app_test_win_df")
    # app_train_win_df, app_test_win_df = \
    #     preprocessing.random_test_train(opt['test_select_type'], app_win_df, opt['test_ratio'], opt['label_type'])
    logging.info("Generating manual_train_win_df, manual_test_win_df")
    manual_train_win_df, manual_test_win_df = \
        preprocessing.random_test_train(opt['test_select_type'], manual_win_df, opt['test_ratio'], opt['label_type'])

    evaluation.init_write(opt, train_opt, features, manual_win_df, app_win_df)
    # ~~~~~~~~~~~~~~ train vehicle/Non-vehicle model ~~~~~~~~~~~~~~~~
    vehicle_or_not_train_win_df = manual_train_win_df
    vehicle_or_not_test_win_df = manual_test_win_df
    vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'], features['ALL_FEATURES'])
    preprocessing.reassign_label(vehicle_or_not_train_win_df, vehicle_or_not_test_win_df,
                                 [[1, 0], [2, 1], [3, 1], [4, 1], [5, 0]], opt['label_type'])
    logging.info("Start to train vehicle_or_not model")
    vehicle_or_not_model = train_dl_models.train_model(vehicle_or_not_train_win_df.iloc[:, vehicle_or_not_index],
                                                       vehicle_or_not_train_win_df[opt['label_type']], train_opt)
    logging.info("Start to test vehicle_or_not model")
    vehicle_or_not_result_label = evaluation.evaluate_single_model(
        vehicle_or_not_model, opt['folder_name'], 'vehicle_or_not',
        np.array(vehicle_or_not_test_win_df.iloc[:, vehicle_or_not_index]),
        np.array(vehicle_or_not_test_win_df[opt['label_type']]))
    # ~~~~~~~~~~~~~~~~ train vehicle_type_model ~~~~~~~~~~~~~~~~
    vehicle_type_train_win_df = manual_train_win_df
    vehicle_type_test_win_df = manual_test_win_df
    vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'], features['ALL_FEATURES'])

    vehicle_type_test_win_df = vehicle_type_test_win_df[(vehicle_type_test_win_df[opt['label_type']] == 4)
                                                        | (vehicle_type_test_win_df[opt['label_type']] == 3) | (
                                                        vehicle_type_test_win_df[opt['label_type']] == 2)]

    vehicle_type_train_win_df = vehicle_type_train_win_df[(vehicle_type_train_win_df[opt['label_type']] == 2)
                                                          | (vehicle_type_train_win_df[opt['label_type']] == 3)
                                                          | (vehicle_type_train_win_df[opt['label_type']] == 4)]

    preprocessing.reassign_label(vehicle_type_train_win_df, vehicle_type_test_win_df,
                                 [[2, 0], [3, 1], [4, 2]], opt['label_type'])
    logging.info("Start to train vehicle_type model")
    vehicle_type_model = train_dl_models.train_model(vehicle_type_train_win_df.iloc[:, vehicle_type_index],
                                                     vehicle_type_train_win_df[opt['label_type']],
                                                     train_opt)
    logging.info("Start to test vehicle_type model")
    vehicle_type_result_label = evaluation.evaluate_single_model(
        vehicle_type_model, opt['folder_name'], 'vehicle_type',
        np.array(vehicle_type_test_win_df.iloc[:, vehicle_type_index]),
        np.array(vehicle_type_test_win_df[opt['label_type']]))
    # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
    overall_result_label = evaluation.evaluate_overall_manual_2(vehicle_or_not_model, vehicle_type_model,
                                                                np.array(manual_test_win_df),
                                                                np.array(manual_test_win_df[opt['label_type']]),
                                                                vehicle_or_not_index,
                                                                vehicle_type_index)
    # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
    evaluation.save_predicted_result_in_csv(vehicle_or_not_result_label, vehicle_or_not_test_win_df, opt['folder_name'],
                                            features['ALL_FEATURES'], 'vehicle_or_not', opt['label_type'])
    evaluation.save_predicted_result_in_csv(vehicle_type_result_label, vehicle_type_test_win_df, opt['folder_name'],
                                            features['ALL_FEATURES'], 'vehicle_type', opt['label_type'])
    evaluation.save_predicted_result_in_csv(overall_result_label, manual_test_win_df, opt['folder_name'],
                                            features['ALL_FEATURES'], 'overall', opt['label_type'])
    evaluation.save_write(opt['folder_name'])

if __name__ == '__main__':
    main()
