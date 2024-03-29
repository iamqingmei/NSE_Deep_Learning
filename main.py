from train_test.main_processing import *
# from utils import memory_control
import datetime
import sys
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)

r = 7

# to select which dataset to train and test
train_test_opt = 'manual_train_app_test'

# 'google_train_google_test'
# 'manual_train_manual_test'
# 'google_train_manual_test'
# 'manual_train_app_test'
# 'google_train_app_test'
# 'manual_train_google_test'
# 'google_train_app_mix_test'
# 'google_plus_manual_train_app_test'
# 'app_train_app_test'

opt = {'test_label_type': 'last_based_win_label',
       'train_label_type': 'all_based_win_label',
       'test_select_type': 'nid',
       'test_ratio': float(1 / 3),
       'random_seed': r,
       'folder_name': './evaluation_report/' + train_test_opt + '/' +
                      str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '/',
       'save_predicted_result': False,
       'get_win_df_from_csv:': True,
       'smooth_overall_result': False,
       'remark': ''}

##################### Training Option ########################
vehicle_or_not_train_opt = {'batch_size': 32, 'epochs': 100, 'l2': 0.001, 'random_seed': r,
                            'middle_output': False, 'ml_opt': 'rf', 'folder_name': opt['folder_name'],
                            'model_name': 'vehicle_or_not'}
vehicle_type_train_opt = {'batch_size': 32, 'epochs': 100, 'l2': 0.001, 'random_seed': r,
                          'middle_output': False, 'ml_opt': 'ada', 'folder_name': opt['folder_name'],
                          'model_name': 'vehicle_type'}

bus_or_not_train_opt = {'batch_size': 512, 'epochs': 24, 'DLNetwork': 'FULL', 'l2': 0.001, 'random_seed': r,
                        'middle_output': False, 'ml_opt': 'ada', 'folder_name': opt['folder_name'],
                        'model_name': 'bus_or_not'}

mrt_or_car_train_opt = {'batch_size': 512, 'epochs': 44, 'DLNetwork': 'FULL', 'l2': 0.001, 'random_seed': r,
                        'middle_output': False, 'ml_opt': 'ada', 'folder_name': opt['folder_name'],
                        'model_name': 'mrt_or_car'}

one_model_train_opt = {'batch_size': 32, 'epochs': 100, 'DLNetwork': 'FULL', 'l2': 0.001, 'random_seed': r,
                       'middle_output': False, 'ml_opt': 'rf', 'folder_name': opt['folder_name'],
                       'model_name': 'one_model'}

features = {'ALL_FEATURES': ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'METRO_DIST', 'BUS_DIST', 'NUM_AP',
                             'STD_VELOCITY_10MIN', 'STOP_BUSSTOP_10MIN', 'TIME_DELTA', 'VELOCITY', 'PRESSURE',
                             'TEMPERATURE', 'NOISE'],
            'VEHICLE_OR_NOT_FEATURES': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'MOV_AVE_VELOCITY'],
            'VEHICLE_TYPE': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'MOV_AVE_VELOCITY',
                             'BUS_DIST', 'METRO_DIST', 'STD_VELOCITY_10MIN', 'STOP_BUSSTOP_10MIN'],
            'BUS_OR_NOT': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'MOV_AVE_VELOCITY',
                           'METRO_DIST', 'BUS_DIST'],
            'MRT_OR_CAR': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'MOV_AVE_VELOCITY'],
            'ONE_MODEL': ['TIME_DELTA', 'NUM_AP', 'STDACC', 'MEANMAG', 'MAXGYR', 'MOV_AVE_VELOCITY',
                          'BUS_DIST', 'METRO_DIST', 'STD_VELOCITY_10MIN', 'STOP_BUSSTOP_10MIN']}

##################### End Training Option ########################

if opt['random_seed'] is not None:
    # set random seed
    # this random seed only affects the self train self test
    # (divide the train&test samples from the same dataset)
    np.random.seed(opt['random_seed'])

if opt['get_win_df_from_csv:']:
    logging.info("Fetch win_df directly from csv")
    app_win_df, manual_win_df, google_win_df = preprocessing.get_win_df_from_csv()
else:
    logging.info("Creating All App Trips")
    app_pt_trip_list = preprocessing.get_app_trip_df_list(features['ALL_FEATURES'])

    logging.info("Generating the window dataframes from all app trips")
    app_win_df = preprocessing.get_win_df(app_pt_trip_list, features['ALL_FEATURES'])
    for x in app_pt_trip_list:
        # x.save_trip_object()
        del x

    logging.info("Creating All Manual Trips")
    manual_pt_trip_list = preprocessing.get_manual_trip_df_list(features['ALL_FEATURES'])

    logging.info("Generating the window dataframe from manual trips")
    manual_win_df = preprocessing.get_win_df(manual_pt_trip_list, features['ALL_FEATURES'])

    for x in manual_pt_trip_list:
        # x.save_trip_object()
        del x

    logging.info("Creating All google Trips")
    google_pt_trip_list = preprocessing.get_google_trip_df_list(features['ALL_FEATURES'])

    logging.info("Generating the window dataframe from google trips")
    google_win_df = preprocessing.get_win_df(google_pt_trip_list, features['ALL_FEATURES'])
    for x in google_pt_trip_list:
        del x
    #
    logging.info("Saving win_df")
    app_win_df.to_csv('./data/app/app_win_df.csv')
    manual_win_df.to_csv('./data/manual/manual_win_df.csv')
    google_win_df.to_csv('./data/google/google_win_df.csv')


def generate_train_test_df(train_test_df_opt):
    logging.info("Generating train_df, test_df")
    train_df = []
    test_df = []

    if train_test_df_opt == 'google_train_google_test':
        train_df, test_df = \
            preprocessing.random_test_train(opt['test_select_type'], google_win_df, opt['test_ratio'])
    elif train_test_df_opt == 'manual_train_manual_test':
        train_df, test_df = \
            preprocessing.random_test_train(opt['test_select_type'], manual_win_df, opt['test_ratio'])
    elif train_test_df_opt == 'google_train_manual_test':
        test_df = manual_win_df
        nid_list = manual_win_df['nid'].tolist()
        nid_list = list(set(nid_list))
        train_df = google_win_df[~(google_win_df['nid'].isin(nid_list))]
    elif train_test_df_opt == 'manual_train_app_test':
        train_df = manual_win_df
        test_df = app_win_df
        preprocessing.reassign_label(test_df, test_df, [[0, 5], [1, 5]])
    elif train_test_df_opt == 'google_train_app_test':
        test_df = app_win_df
        preprocessing.reassign_label(test_df, test_df, [[0, 5], [1, 5]])
        nid_list = test_df['nid'].tolist()
        nid_list = list(set(nid_list))
        train_df = google_win_df[~(google_win_df['nid'].isin(nid_list))]
    elif train_test_df_opt == 'manual_train_google_test':
        train_df = manual_win_df
        nid_list = train_df['nid'].tolist()
        nid_list = list(set(nid_list))
        test_df = google_win_df[~(google_win_df['nid'].isin(nid_list))]
    elif train_test_df_opt == 'google_train_app_mix_test':
        test_df = app_win_df[app_win_df['all_based_win_label'] == -1]
        preprocessing.reassign_label(test_df, test_df, [[0, 5], [1, 5]])
        nid_list = test_df['nid'].tolist()
        nid_list = list(set(nid_list))
        train_df = google_win_df[~(google_win_df['nid'].isin(nid_list))]
    elif train_test_df_opt == 'app_train_app_test':
        train_df, test_df = \
            preprocessing.random_test_train(opt['test_select_type'], app_win_df, opt['test_ratio'])
        preprocessing.reassign_label(train_df, test_df, [[0, 5], [1, 5]])
    elif train_test_df_opt == 'google_plus_manual_train_app_test':
        test_df = app_win_df
        preprocessing.reassign_label(test_df, test_df, [[0, 5], [1, 5]])
        train_df = google_win_df.append(manual_win_df)
    else:
        print("PLEASE enter the correct train_test_opt")
        quit()

    logging.info("Remove the mix/invalid data")
    train_df = preprocessing.remove_mix(train_df, opt['train_label_type'])
    test_df = preprocessing.remove_mix(test_df, opt['test_label_type'])

    train_df.index = list(range(len(train_df)))
    test_df.index = list(range(len(test_df)))
    return train_df, test_df


def main():
    o = 'manual_train_app_test'
    train_df, test_df = generate_train_test_df(o)
    opt['folder_name'] = './evaluation_report_testing/' + str(o) + '/' + \
                         str(datetime.datetime.now().strftime("%y-%m-%d %H:%M")) + '/'
    one_model_train_opt['epochs'] = 19
    one_model_train_opt['batch_size'] = 32
    one_model_train_opt['random_seed'] = 7
    opt['random_seed'] = 7

    dl_one_model_train_test(train_df, test_df, opt, one_model_train_opt, o, features)
    dl_train_test(train_df, test_df,
                  opt, vehicle_or_not_train_opt, vehicle_type_train_opt, o, features)
    ml_one_train_test('rbf', train_df, test_df, opt, features)
    ml_one_train_test('poly', train_df, test_df, opt, features)
    ml_one_train_test('sigmoid', train_df, test_df, opt, features)
    ml_one_train_test('rf', train_df, test_df, opt, features)
    ml_one_train_test('ada', train_df, test_df, opt, features)


if __name__ == '__main__':
    main()
