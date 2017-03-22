import numpy as np
import logging
import datetime
import psycopg2
import params
import pandas as pd
from dbConnPSQL import getLabelledDataAll, getMainDataPSQL2016, getTripDataPSQL2016
from feature_calc import calc_extra_features, calc_geo_time_features, clean_geo_data, DL_FEATURES, cal_win_label, \
    cal_win_features, cal_busmrt_dist, WIN_FEATURES
from normalization import normalize, win_normalize
import pickle


def get_app_win_df(window_size):
    app_features_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_labels_df.csv')['pt_label'].tolist()
    with open("./pt_df/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    # normalized the point features
    app_features_pt = normalize(app_features_pt[DL_FEATURES])
    # app_features_pt is a Dataframe

    labels_win = cal_win_label(app_labels_pt, window_size, trip_dict)
    features_win = cal_win_features(app_features_pt, window_size, trip_dict)

    # normalize the features for window level
    if len(WIN_FEATURES) > 0:
        features_win = win_normalize(features_win)

    # check whether the features match with labels
    if len(features_win) != len(labels_win):
        logging.warning("the windows features are not matched with labels!!!!!!!!!!!!!!!!!!!!!!")

    app_win_df = pd.DataFrame(features_win)
    app_win_df['win_label'] = pd.Series(labels_win)

    # remove the window with label mix
    app_win_df = app_win_df[app_win_df.win_label != 5]
    # now the win_df is unbalanced and has 5 labels
    return app_win_df


def save_app_pt_df(window_size):
    # initialization
    trip_dict = {}
    trip_count = 0
    trip_dict[-1] = 0
    label_type = "app"

    # conCom = """dbname='nse_mode_id' user='postgres' password='"""+dbpw_str+"""' host='localhost'"""
    conCom = """dbname='""" + params.dbname_str + """' user='""" + params.dbuser_str + """' password='""" + \
             params.dbpw_str + """' host='""" + params.dbhost + """' port ='""" + params.dbport + """' """
    connPSQL = psycopg2.connect(conCom)
    cursorPSQL = connPSQL.cursor()

    # queary for all vehicle labelled data
    trip_df_labelled = getLabelledDataAll(cursorPSQL, "allweeks_tripsummary", label_type, max_num=None)
    # Returns trip_df_labelled which consists of a data frame of all node ID's which have been labelled
    # which are of type 'vehicle' vehicle = 4,5,6 MRT/Bus/Car

    # get all combination of nid, date, tripnum, which specify a particular trip of one node on one day
    nid_date_tripnum = trip_df_labelled[['NID', 'analyzed_date', 'trip_num']].values
    # print nid_date_tripnum
    # get the unique (nid, date) pair to obtain data of the entire day
    all_nid = nid_date_tripnum[:, 0].tolist()
    all_date = nid_date_tripnum[:, 1].tolist()
    unique_nid_date = set(zip(all_nid, all_date))
    # item in unique_nid_date_with_tripnum: item[0][0] is nid, item[0][1] is date, item[1] is a list of tripnum
    unique_nid_date_with_tripnum = []
    for one_nid_date in unique_nid_date:
        # get the interested tripnum of that particular (nid, date)
        cur_nid = one_nid_date[0]
        cur_date = one_nid_date[1]
        cur_tripnum = nid_date_tripnum[
            (nid_date_tripnum[:, 0] == cur_nid) & (nid_date_tripnum[:, 1] == cur_date), 2].tolist()
        unique_nid_date_with_tripnum.append([one_nid_date, cur_tripnum])

    # print unique_nid_date_with_tripnum

    # initialization
    features_win = []
    labels_win = []
    features_pt = []
    labels_pt = []
    lat_pt = []
    lon_pt = []
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
        data_frame_raw = getMainDataPSQL2016(cursorPSQL, 'allweeks_clean', cur_nid, ana_date_str, second_date_str)
        data_frame_full = getTripDataPSQL2016(cursorPSQL, 'allweeks_extra', cur_nid, ana_date_str, second_date_str,
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
        success_featCalc = calc_geo_time_features(data_frame_full, ana_date_str, window_size)

        if not success_featCalc:
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
            if label_type == 'manual':
                is_labeled_mask = np.array(
                    [~np.isnan(np.float64(item)) for item in df_cur_trip['gt_mode_manual'].values])
                label_pt_tmp = df_cur_trip['gt_mode_manual'].values[is_labeled_mask]
            elif label_type == 'app':
                is_labeled_mask = np.array([~np.isnan(np.float64(item)) for item in df_cur_trip['gt_mode_app'].values])
                label_pt_tmp = df_cur_trip['gt_mode_app'].values[is_labeled_mask]

            if len(label_pt_tmp) < window_size:
                continue
            label_pt_tmp[(label_pt_tmp == 0) | (label_pt_tmp == 1)] = 0  # stationary
            label_pt_tmp[(label_pt_tmp == 3) | (label_pt_tmp == 2)] = 1  # walking
            label_pt_tmp[(label_pt_tmp == 4)] = 2  # train
            label_pt_tmp[(label_pt_tmp == 5)] = 3  # bus
            label_pt_tmp[(label_pt_tmp == 6)] = 4  # car

            features_pt_tmp = df_cur_trip[DL_FEATURES].values[is_labeled_mask]
            lat_pt += df_cur_trip['WLATITUDE'].values[is_labeled_mask].tolist()
            lon_pt += df_cur_trip['WLONGITUDE'].values[is_labeled_mask].tolist()
            if len(features_pt_tmp) != len(label_pt_tmp):
                logging.warning("the features are not matched with labels!!!!!!!!!!!!!!!!!!!!!!")
                continue

            labels_pt += label_pt_tmp.tolist()
            features_pt += features_pt_tmp.tolist()

            trip_dict[trip_count] = len(features_pt)
            trip_count += 1

    # for i in range(10):
    #     print trip_dict[i]

    pd.DataFrame(features_pt, columns=DL_FEATURES).to_csv('./pt_df/unnormalized_pt_features_df.csv')
    pd.DataFrame(labels_pt, columns=['pt_label']).to_csv('./pt_df/unnormalized_pt_labels_df.csv')
    with open("./pt_df/trip_dict.txt", "wb") as fp:
        pickle.dump(trip_dict, fp)