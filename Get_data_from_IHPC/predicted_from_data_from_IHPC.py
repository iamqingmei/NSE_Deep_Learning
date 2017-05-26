import numpy as np
import pandas as pd

import logging
import psycopg2
import os
import sys

import params
from Get_data_from_IHPC import tripParse
from data_retrieval.dbConnPSQL import save_extra_PSQL_2016, save_tripsummary_PSQL_2016
from utils.util import apply_DBSCAN
from utils.load_model import load_model
from preprocessing.Trip_Class import Trip
from train_test import evaluation
from utils.util import chunks_real
from preprocessing import preprocessing
from data_retrieval.feature_calc import calc_extra_features, calc_geo_time_features, clean_geo_data, cal_busmrt_dist


def predict_from_raw_df(features, opt, test_date, test_nid):
    save_to_db = False
    predict_by_model = True
    folder = './data/data_frame_from_IHPC/processed_df/'
    file_name = test_nid + '_' + test_date + '.csv'
    try:
        all_files = os.listdir(folder)
    except FileNotFoundError:
        if not os.path.exists(folder):
            os.makedirs(folder)
        all_files = os.listdir(folder)

    if file_name in all_files:
        data_frame_full = pd.DataFrame.from_csv(folder + file_name)
    else:
        raw_df, labelled_df_list = get_raw_data_from_csv(test_nid, test_date)

        logging.info("Assigning Labels")
        data_frame_full = assign_app_label(raw_df, labelled_df_list)
        if data_frame_full is None:
            logging.warning("No Dataframes with assigned labels, nid: %s date: %s" % (test_nid, test_date))
            return

        logging.info("Feature Calculation and Normalization")
        normalized_df = all_feature_calculation_normalization(data_frame_full, features['ALL_FEATURES'], test_date)
        logging.info("Converting to window dataframe")
        data_frame_full = preprocessing.get_win_df([Trip(normalized_df, 1, test_nid)], features['ALL_FEATURES'])
        del normalized_df

        if not os.path.exists(folder):
            os.makedirs(folder)
        data_frame_full.to_csv(folder + file_name)

    if predict_by_model is True:
        data_frame_full = preprocessing.remove_mix(data_frame_full, opt['test_label_type'])
        preprocessing.reassign_label(data_frame_full, data_frame_full,
                                     [[0, 5], [1, 5]])
        logging.info("Loading models")
        vehicle_type_model = load_model('./evaluation_report_testing_only/google_train_app_test/'
                                        'google_train_app_test/17-05-15 16:13/',
                                        "vehicle_type_model")
        vehicle_or_not_model = \
            load_model('./evaluation_report_testing_only/google_train_app_test/'
                       'google_train_app_test/17-05-15 16:13/', "vehicle_or_not_model")

        logging.info("Getting the result of vehicle_or_nor model")
        vehicle_or_not_test_df = data_frame_full.copy()

        vehicle_or_not_index = preprocessing.get_feature_idx(features['VEHICLE_OR_NOT_FEATURES'],
                                                             features['ALL_FEATURES'])
        preprocessing.reassign_label(vehicle_or_not_test_df, vehicle_or_not_test_df, [[1, 0], [2, 1], [3, 1], [4, 1],
                                                                                      [5, 0]])

        logging.info("Start to evaluate vehicle_or_not model")
        evaluation.evaluate_single_model(
            vehicle_or_not_model,
            opt['folder_name'],
            'model_name',
            np.array(vehicle_or_not_test_df.iloc[:, vehicle_or_not_index]),
            np.array(vehicle_or_not_test_df[opt['test_label_type']]),
            save_model=False,
            num_classes=2)

        logging.info("Getting the result of vehicle_type model")
        vehicle_type_test_df = data_frame_full.copy()
        vehicle_type_index = preprocessing.get_feature_idx(features['VEHICLE_TYPE'],
                                                           features['ALL_FEATURES'])
        vehicle_type_test_df = vehicle_type_test_df[(vehicle_type_test_df[opt['test_label_type']] == 4) |
                                                    (vehicle_type_test_df[opt['test_label_type']] == 3) |
                                                    (vehicle_type_test_df[opt['test_label_type']] == 2)]
        preprocessing.reassign_label(vehicle_type_test_df, vehicle_type_test_df,
                                     [[2, 0], [3, 1], [4, 2]])

        logging.info("Start to evaluate vehicle_type model")
        evaluation.evaluate_single_model(
            vehicle_type_model,
            opt['folder_name'],
            'model_name',
            np.array(vehicle_type_test_df.iloc[:, vehicle_type_index]),
            np.array(vehicle_type_test_df[opt['test_label_type']]),
            save_model=False,
            num_classes=3)

        # ~~~~~~~~~~~~~~~~~ get overall result ~~~~~~~~~~~~~~~~~~~
        overall_test_win_df = data_frame_full.copy()
        overall_result_label = evaluation.evaluate_overall_manual_2(vehicle_or_not_model, vehicle_type_model,
                                                                    overall_test_win_df,
                                                                    overall_test_win_df[opt['test_label_type']],
                                                                    vehicle_or_not_index,
                                                                    vehicle_type_index, opt['smooth_overall_result'])
        # ~~~~~~~~~~~~~~~~~ Save predicted result into csv for visualization ~~~~~~~~~~
        evaluation.save_predicted_result_in_csv(overall_result_label, overall_test_win_df, opt['folder_name'],
                                                'overall', opt['test_label_type'])
        evaluation.save_write(opt['folder_name'])
    if save_to_db:
        poi_latlon_heu = tripParse.detectPOI_geov(data_frame_full,
                                                  params.stopped_thresh,
                                                  params.poi_min_dwell_time,
                                                  params.loc_round_decimals)

        # Combine the detected POIs using DBSCAN
        pois_latlon_raw = np.array(poi_latlon_heu)
        logging.info("raw POIs: " + str(pois_latlon_raw))
        pois_latlon_comb = []
        if len(pois_latlon_raw) > 0:
            core_samples_mask, labels = apply_DBSCAN(pois_latlon_raw, params.poi_comb_range, params.poi_comb_samples)
            unique_labels = np.unique(labels)
            logging.info("labels when combing: " + str(labels))
            for unique_label in unique_labels:
                if not unique_label == -1:
                    cur_lat_mean, cur_lon_mean = \
                        np.mean(pois_latlon_raw[(labels == unique_label) & core_samples_mask, :], 0)
                    pois_latlon_comb.append([float(round(cur_lat_mean, params.loc_round_decimals)),
                                             float(round(cur_lon_mean, params.loc_round_decimals))])
            for idx, label in enumerate(labels):
                if label == -1:
                    pois_latlon_comb.append([float(pois_latlon_raw[idx, 0]), float(pois_latlon_raw[idx, 1])])
        logging.info("combined POIs: " + str(pois_latlon_comb))

        # idetify home & school from the POIs
        home_loc, school_loc, pois_label_temp = \
            tripParse.identify_home_school(pois_latlon_comb, data_frame_full, school_start=params.school_start,
                                           school_end=params.school_end, home_start=params.home_start,
                                           home_end=params.home_end, min_school_thresh=params.min_school_thresh,
                                           poi_cover_radius=params.poi_cover_radius)
        logging.info("Temporary labels of POIs: " + str(pois_label_temp))

        # label all points based on the home/school & POI location
        pois_dict = \
            tripParse.label_pts_by_pois(pois_latlon_comb, pois_label_temp, data_frame_full,
                                        home_cover_radius=params.home_cover_radius,
                                        sch_cover_radius=params.sch_cover_radius,
                                        poi_cover_radius=params.poi_cover_radius,
                                        poi_min_dwell_time=params.poi_min_dwell_time)
        logging.info("Chronological POIs: " + str(pois_dict['pois_latlon_chro']))
        logging.info("Chronological POI labels: " + str(pois_dict['pois_label_chro']))

        # take out trips and add triplabel to data frame
        trips_dict = {'trip_num': [], 'start_poi_loc': [], 'end_poi_loc': [], 'tot_dist(km)': [], 'tot_dura(s)': [],
                      'start_sgt': [], 'end_sgt': [], 'tot_num_trips': 0, 'nid': test_nid,
                      'analyzed_date': test_date,
                      'home_loc': home_loc, 'school_loc': school_loc, 'valid_loc_perc': [], 'num_pt': []}

        trip_labels = np.array([None] * len(data_frame_full))
        # chunks of the poi label, -1 chunks are trips
        poi_label_chunks = tripParse.chunks_real(data_frame_full['POI_LABEL'].values.tolist(), include_values=True)
        logging.info("Chronological chunks of poi labels: " + str(poi_label_chunks))
        trip_num = 0
        start_poi_num = 0
        end_poi_num = 1
        for idx, label_chunk in enumerate(poi_label_chunks):
            # go through each trip chunk and get information of trips
            if label_chunk[2] == -1:
                # if it's a trip chunk
                trip_num += 1
                trips_dict['trip_num'].append(trip_num)
                trip_labels[label_chunk[0]:label_chunk[1]] = trip_num
                if idx == 0:
                    # if there's no start poi
                    trips_dict['start_poi_loc'].append([])
                    end_poi_num -= 1
                else:
                    trips_dict['start_poi_loc'].append(pois_dict['pois_latlon_chro'][start_poi_num])
                    start_poi_num += 1
                if idx == len(poi_label_chunks) - 1:
                    # if there's no end poi
                    trips_dict['end_poi_loc'].append([])
                else:
                    trips_dict['end_poi_loc'].append(pois_dict['pois_latlon_chro'][end_poi_num])
                    end_poi_num += 1
                trips_dict['tot_dist(km)'].append(
                    round(np.nansum(data_frame_full['DISTANCE_DELTA'][label_chunk[0]:label_chunk[1]].values) / 1000,
                          params.dist_round_decimals))
                trips_dict['tot_dura(s)'].\
                    append(int(np.nansum(data_frame_full['TIME_DELTA'][label_chunk[0]:label_chunk[1]].values)))
                trips_dict['start_sgt'].append(round(data_frame_full['TIME_SGT'][label_chunk[0]], 3))
                trips_dict['end_sgt'].append(round(data_frame_full['TIME_SGT'][label_chunk[1] - 1], 3))
                cur_lat = data_frame_full['WLATITUDE'][label_chunk[0]:label_chunk[1]].values
                trips_dict['valid_loc_perc'].append(len(cur_lat[~np.isnan(cur_lat)]) * 1.0 / len(cur_lat))
                trips_dict['num_pt'].append(label_chunk[1] - label_chunk[0])
            trips_dict['tot_num_trips'] = trip_num

        data_frame_full['TRIP_LABEL'] = pd.Series(trip_labels)  # trip_labels are 1, 2, 3, 4, ...

        # con_com = """dbname='nse_mode_id' user='postgres' password='"""+dbpw_str+"""' host='localhost'"""
        con_com = """dbname='""" + params.dbname_str + """' user='""" + params.dbuser_str + """' password='""" + \
                  params.dbpw_str + """' host='""" + params.dbhost + """' port ='""" + params.dbport + """' """
        conn_psql = psycopg2.connect(con_com)
        cursor_psql = conn_psql.cursor()
        """ save extra labels and features into the PSQL DB table """
        logging.warning("Starting to save extra labels and features into the PSQL DB table")
        save_extra = save_extra_PSQL_2016(conn_psql, cursor_psql, params.tableExtra2016, data_frame_full)
        logging.warning("Extra columns saving status: " + str(save_extra))

        """ save trip dictionary into the PSQL DB table """
        if trips_dict['tot_num_trips'] > 0:
            logging.warning("Start to save trip dictionary into the PSQL DB table")
            save_trips = save_tripsummary_PSQL_2016(conn_psql, cursor_psql, params.tableTrip2016,
                                                    params.tableExtra2016, trips_dict)
            logging.warning("Trip summary saving status: " + str(save_trips))


def assign_app_label(clean_df, app_labelled_df_list):
    clean_df_processed_with_label = clean_df.copy()
    clean_df_processed_with_label.loc[:, 'pt_label'] = pd.Series([params.defalt_mixed_invalid_label] * len(clean_df),
                                                                 index=clean_df.index)

    for app_labelled_df in app_labelled_df_list:
        app_labelled_df = app_labelled_df.iloc[:, [0, -1]]
        app_labelled_df = app_labelled_df[~np.isnan(app_labelled_df['timestamp(unix)'])]
        app_labelled_df = app_labelled_df.sort_values(['timestamp(unix)'], ascending=[True])

        # chunks_real return a list like in form [[start, end, label]] :[[0, 2004, 4], [2004, 3111, 1], [3111, 3634, 3]]
        for chunk in chunks_real(app_labelled_df['User Mark'], True):
            chunk_start_time_unix = app_labelled_df.iloc[chunk[0]]['timestamp(unix)']
            chunk_end_time_unix = app_labelled_df.iloc[chunk[1] - 1]['timestamp(unix)']
            chunk_label = chunk[2]

            clean_df_processed_with_label.loc[
                ((chunk_end_time_unix > clean_df_processed_with_label.TIMESTAMP) &
                 (clean_df_processed_with_label.TIMESTAMP > chunk_start_time_unix)), 'pt_label']\
                = chunk_label

    clean_df_processed_with_label = \
        clean_df_processed_with_label[['NID', 'SGT', 'HUMIDITY', 'LIGHT', 'MODE', 'CMODE', 'NOISE', 'PRESSURE',
                                       'STEPS', 'TEMPERATURE', 'IRTEMPERATURE', 'MEANMAG', 'MEANGYR', 'STDGYR',
                                       'STDACC', 'MAXACC', 'MAXGYR', 'MAC', 'WLATITUDE', 'WLONGITUDE', 'ACCURACY',
                                       'pt_label', 'TIMESTAMP']]
    # clean_df_processed_with_label.rename(columns={'User Mark': 'pt_label'})
    success_label_count = clean_df_processed_with_label['pt_label'].tolist().count(params.defalt_mixed_invalid_label)
    if success_label_count > 0:
        logging.info("The dataframe has successfully labelled, with %d labels" %
                     (len(clean_df_processed_with_label) - success_label_count))
        return clean_df_processed_with_label
    else:
        logging.info("Assign app labels fail")
        return None


def get_raw_data_from_csv(test_nid, test_date):
    raw_df_folder = './data/data_frame_from_IHPC/raw_df/'
    label_folder = './data/labels_collected_by_ourselves/' + test_nid + '/'

    all_files_in_raw_df_folder = os.listdir(raw_df_folder)
    try:
        all_files_in_label_folder = os.listdir(label_folder)
    except FileNotFoundError:
        logging.warning("The input nid is WRONG! Cannot find label file for nid %s" % test_nid)
        sys.exit()

    raw_df = None
    for i in all_files_in_raw_df_folder:
        if ("raw_df_table_" + test_nid + "_" + test_date) in i:
            raw_df = pd.DataFrame.from_csv(raw_df_folder + i)
            break
    if raw_df is None:
        logging.warning("WRONG date or nid!")
        sys.exit()

    labelled_df_list = []
    for i in all_files_in_label_folder:
        if (test_date[:4] + test_date[5:7] + test_date[-2:]) in i:
            cur_labelled_df = pd.read_csv(label_folder + i, index_col=False)
            if len(cur_labelled_df) < 1:
                continue
            else:
                labelled_df_list.append(cur_labelled_df)

    if len(labelled_df_list) == 0:
        logging.warning("Cannot find label file for nid %s at date %s " % (test_nid, test_date))

    return raw_df, labelled_df_list


def all_feature_calculation_normalization(data_frame_full, all_features, test_date):
    clean_geo_data(data_frame_full)
    logging.info("Calculate geo and time features: ")
    success_feat_calc = calc_geo_time_features(data_frame_full, test_date, params.window_size)
    if not success_feat_calc:
        logging.warning("Feature Calculation is not successful")
        sys.exit()

    cal_busmrt_dist(data_frame_full)
    calc_extra_features(data_frame_full, params.window_size)
    per_valid_loc = float(np.sum(data_frame_full['is_localized'].values)) / len(data_frame_full)
    logging.info("Localization rate: " + str(per_valid_loc))

    v_tmp = data_frame_full['VELOCITY'].values.tolist()
    v_last = 0
    v_new = []
    for v in v_tmp:
        if not np.isnan(v):
            v_last = v
        v_new.append(v_last)
    data_frame_full['VELOCITY'] = v_new

    # normalized the point features
    normalized_df = normalize(data_frame_full[all_features])
    normalized_df['pt_label'] = data_frame_full['pt_label']
    normalized_df['WLONGITUDE'] = data_frame_full['WLONGITUDE']
    normalized_df['WLATITUDE'] = data_frame_full['WLATITUDE']

    return normalized_df
