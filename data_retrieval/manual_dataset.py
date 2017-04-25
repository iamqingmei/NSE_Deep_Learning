import numpy as np
import logging
import datetime
import psycopg2
import params
import pandas as pd
from .dbConnPSQL import getLabelledDataWithNullAppLabel, getMainDataPSQL2016, getTripDataPSQL2016, \
    get_nids_with_app_label
from .feature_calc import calc_extra_features, calc_geo_time_features, clean_geo_data, DL_FEATURES, cal_busmrt_dist
import pickle
import os


def save_manual_pt_df(window_size):
    """
    Fetch data from database and calculate feature for all the data
    After all the calculation, save the dataframe into a csv for further uses
    :param window_size: the window size for sliding window
    :return: None
    """
    # initialization
    trip_dict = {}
    trip_count = 0
    trip_dict[-1] = [0, -1]
    label_type = "manual"

    #  create folder if not exists
    if not os.path.exists('data/manual/'):
        os.makedirs('data/manual/')

    # con_com = """dbname='nse_mode_id' user='postgres' password='"""+dbpw_str+"""' host='localhost'"""
    con_com = """dbname='""" + params.dbname_str + """' user='""" + params.dbuser_str + """' password='""" + \
              params.dbpw_str + """' host='""" + params.dbhost + """' port ='""" + params.dbport + """' """
    conn_psql = psycopg2.connect(con_com)
    cursor_psql = conn_psql.cursor()

    # queary for all vehicle labelled data
    trip_df_labelled = getLabelledDataWithNullAppLabel(cursor_psql, "allweeks_tripsummary", label_type, max_num=None)
    trip_df_labelled = trip_df_labelled[trip_df_labelled.user_id == 1]
    nids_with_app_label = get_nids_with_app_label(cursor_psql, "allweeks_extra")
    # Returns trip_df_labelled which consists of a data frame of all node ID's which have been labelled
    # which are of type 'vehicle' vehicle = 4,5,6 MRT/Bus/Car
    # walk or stationary = 3
    # get all combination of nid, date, tripnum, which specify a particular trip of one node on one day
    nid_date_tripnum = trip_df_labelled[['NID', 'analyzed_date', 'trip_num']].values
    # print nid_date_tripnum
    # get the unique (nid, date) pair to obtain data of the entire day
    all_nid = nid_date_tripnum[:, 0].tolist()
    all_date = nid_date_tripnum[:, 1].tolist()
    unique_nid_date = set(zip(all_nid, all_date))
    # item in unique_nid_date_with_tripnum:
    # item[0][0] is nid, item[0][1] is date, item[1] is a list of tripnum
    unique_nid_date_with_tripnum = []
    for one_nid_date in unique_nid_date:
        # get the interested tripnum of that particular (nid, date)
        cur_nid = one_nid_date[0]
        if cur_nid in nids_with_app_label:
            continue
        else:
            cur_date = one_nid_date[1]
            cur_tripnum = nid_date_tripnum[
                (nid_date_tripnum[:, 0] == cur_nid) & (nid_date_tripnum[:, 1] == cur_date), 2].tolist()
            unique_nid_date_with_tripnum.append([one_nid_date, cur_tripnum])

    # print unique_nid_date_with_tripnum

    # initialization
    features_pt = []
    labels_pt = []
    # process each [nid, date] pair, obtain data, calculate features, and process each trip
    num_iteration = len(unique_nid_date_with_tripnum)

    for item in unique_nid_date_with_tripnum[0:num_iteration]:
        cur_nid = item[0][0]
        cur_date = item[0][1]
        all_tripnum = item[1]
        logging.info(
            "***** processing trip " + str(all_tripnum) + " of " + str(cur_nid) + " on " + str(cur_date) + " *****")

        # obtain raw data and labels for the entire day
        ana_date_tuple = datetime.datetime.strptime(str(cur_date), '%Y%m%d')
        ana_date_str = ana_date_tuple.strftime('%Y-%m-%d')
        second_date_str = (ana_date_tuple + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        data_frame_raw = getMainDataPSQL2016(cursor_psql, 'allweeks_clean', cur_nid, ana_date_str, second_date_str)
        data_frame_full = getTripDataPSQL2016(cursor_psql, 'allweeks_extra', cur_nid, ana_date_str, second_date_str,
                                              data_frame_raw)
        # print data_frame_full
        if data_frame_full is None:
            logging.warning("Didn't get any data for this [nid,date] pair!")
            continue
        if len(data_frame_full) < window_size:
            logging.warning("Not Enough data for nid: " + str(cur_nid) + " on " + str(cur_date))
            continue
        # calculate all features for the entire day
        # clean data to remove invalid locations
        logging.info("Clean data:")
        clean_geo_data(data_frame_full)

        # calculate following features:
        # ['MOV_AVE_ACCELERATION']['ACCELERATION']['VELOCITY']['MOV_AVE_VELOCITY']
        # ['DISTANCE_DELTA']['TIME_DELTA']['STEPS_DELTA']
        logging.info("Calculate geo and time features: ")
        success_feat_calc = calc_geo_time_features(data_frame_full, ana_date_str, window_size)

        if not success_feat_calc:
            continue

        cal_busmrt_dist(data_frame_full)

        # calculate following features:
        # ['STDMEAN_MAG_5WIN'] ['drMEAN_MAG_5WIN'] ['STDPRES_5WIN']['is_localized']
        calc_extra_features(data_frame_full, window_size)
        # print "Columns inside data_frame_full: "+str(data_frame_full.columns.values)
        per_valid_loc = float(np.sum(data_frame_full['is_localized'].values)) / len(data_frame_full)
        logging.info("Localization rate: " + str(per_valid_loc))

        # process VELOCITY to remove np.nan, nan of MOV_AVE_VELOCITY has been removed in util.py
        v_tmp = data_frame_full['VELOCITY'].values.tolist()
        v_last = 0
        v_new = []
        for v in v_tmp:
            if not np.isnan(v):
                v_last = v
            v_new.append(v_last)
        data_frame_full['VELOCITY'] = v_new

        for tripnum in all_tripnum:

            # go through each trip of this node on this day
            # print "--- tripnum = "+str(tripnum)+" ---"
            # obtain part of the df of this particular trip
            df_cur_trip = data_frame_full.loc[data_frame_full['triplabel'] == tripnum].copy()

            if len(df_cur_trip) < window_size:
                logging.warning("*********Not Enough data for tripnum = " + str(tripnum) + "*********")
                continue

            # get labels
            label_pt_tmp = df_cur_trip['gt_mode_manual'].values

            if None in label_pt_tmp.tolist():
                logging.warning("The trip information is invalid")
                continue

            label_pt_tmp[(label_pt_tmp == 3)] = 0  # walking or stationary
            label_pt_tmp[(label_pt_tmp == 4)] = 1  # train
            label_pt_tmp[(label_pt_tmp == 5)] = 2  # bus
            label_pt_tmp[(label_pt_tmp == 6)] = 3  # car
            # for those points without label
            for i in range(len(label_pt_tmp)):
                if np.isnan(label_pt_tmp[i]):
                    # logging.info("there are nan in labels, replaced with -1")
                    label_pt_tmp[i] = -1  # means no label

            features_pt_tmp = df_cur_trip[DL_FEATURES]
            if len(features_pt_tmp) != len(label_pt_tmp):
                logging.warning("the features are not matched with labels!!!!!!!!!!!!!!!!!!!!!!")
                continue

            labels_pt += label_pt_tmp.tolist()
            features_pt += features_pt_tmp.values.tolist()

            cur_user_id = trip_df_labelled.loc[(trip_df_labelled['NID'] == cur_nid) &
                                               (trip_df_labelled['analyzed_date'] == cur_date) &
                                               (trip_df_labelled['trip_num'] == tripnum), ['user_id']].values[0][0]
            trip_dict[trip_count] = [len(features_pt), cur_user_id]
            trip_count += 1

    pd.DataFrame(features_pt, columns=DL_FEATURES).to_csv('data/manual/unnormalized_pt_features_df.csv')
    pd.DataFrame(labels_pt, columns=['pt_label']).to_csv('data/manual/unnormalized_pt_labels_df.csv')
    with open("data/manual/trip_dict.txt", "wb") as fp:
        pickle.dump(trip_dict, fp)
