#!/usr/bin/env python

""" Scripts to get time-series data from IHPC DB and save it as csv files or save it into PSQL DB
"""

import time
import logging
import pandas as pd
import sys
import datetime
import traceback
import numpy as np
import calendar
import os
import timeit
import psycopg2
import params
from Get_data_from_IHPC.dbConnIHPC import getDataIHPC
from data_retrieval.dbConnPSQL import save_data_from_IHPC_to_PSQL
from data_retrieval.feature_calc import calc_geo_time_features, calc_extra_features, clean_geo_data, rm_int_ts_outliers
import pytz

###########
# 1. Read nid and date pairs from NID_DATE_1.csv
# 2. Request the url and save the raw data into ../data/data_frame_from_IHPC/raw_df/
###########
API_key = "sutd-nse_api:dj6M9RAxynrjw9aWztzprfh5AKHssgVj4qKXiKSfHRyGKeoX92wmwmEJKpHMIB5"

# node_date_file: the file name of the csv file of device ids and dates
# device_file_folder: the folder name which contains the file of device ids
# query_date_default: the default date string used to process if there's no date list in the file
# log_level: log_level decides what level of information to be logged
node_date_file = "NID_DATE_1.csv"
# node_date_file = "nodes_with_known_labels.csv"
# node_date_file = "nodes_with_walking_labels.csv"
device_file_folder = "../data/device_date_list/"
query_date_default = "2016-02-15"
log_level = logging.INFO


def visualization_for_point(maps, point_list, color):
    """For trip visualization"""
    for point in point_list:
        if ~np.isnan(point[0]):
            maps.addpoint(point[0], point[1], color)


# def main(url, node_date_file, queried_date =
def main():
    def process(cur_nid, queried_date):
        """Get data from IHPC for device nid with given date (%Y-%m-%d)
        Options:
            1. Save locally
            2. Save to PSQL
            3. Save track maps

        Corresponding thresholds:
            save_raw_df
            save_processed_df
            save_map
            save_df_2db

        """
        logging.warning("Process device %d on %s" % (cur_nid, queried_date))

        # convert queried_date into unix timestamp 0 am in UTC time
        cur_queried_date_tuple = datetime.datetime.strptime(queried_date, "%Y-%m-%d")
        queried_unix = calendar.timegm(cur_queried_date_tuple.timetuple())

        # get the starting and end indices for querying the data, UTC timestamps
        # start_get = queried_unix-8*3600+12*3600 #12 pm (SGT time) of the analysis day
        start_get = queried_unix - 8 * 3600  # 0 am (SGT time) of the analysis day
        end_get = start_get + 24 * 3600 - 1  # 24 pm (SGT time) of the analysis day

        # retrieve unprocessed device data from the DB
        logging.info("Get data from IHPC for device %d on %s" % (cur_nid, queried_date))
        reading_start = timeit.default_timer()
        url = "https://data.nse.sg"
        df_raw = getDataIHPC(url, cur_nid, start_get, end_get)
        reading_time = timeit.default_timer() - reading_start
        # logging of reading time into err file
        logging.warning("READING TIME of " + str(cur_nid) + ": " + str(reading_time))

        if df_raw is None:
            logging.info("No data returned for device %d, skip." % cur_nid)
            return None
        elif len(df_raw) < 50:
            # if the data frame size is smaller than a certain threshold, then abandon the data
            logging.warning("Too little data returned for device %d, skip." % cur_nid)
            return None

        # show pt counts and time span
        num_pt = len(df_raw)
        time_span = (df_raw['TIMESTAMP'][num_pt - 1] - df_raw['TIMESTAMP'][0]) / 3600 % 24
        logging.warning("NID: " + str(cur_nid) + "; TIME SPAN: " + str(time_span) + "; NUM of PTS: " + str(num_pt))

        exe_start = timeit.default_timer()

        # remove interleaved timestamps and outliers of all values
        logging.info("Before outlier cleaning the dataframe has % samplles" % len(df_raw))
        df_cleaned = rm_int_ts_outliers(df_raw)
        logging.info("After outlier cleaning the dataframe has % samplles" % len(df_cleaned))

        # save the raw data frame into csv file
        if save_raw_df:
            df_table_file = raw_df_table_folder + "/raw_df_table_" + str(cur_nid) + "_" + str(queried_date) + "_" + str(
                num_pt) + "pts" + ".csv"
            try:
                df_to_save = df_cleaned[['HUMIDITY', 'LIGHT', 'MODE', 'CMODE', 'NOISE', \
                  'PRESSURE', 'STEPS', 'TEMPERATURE', 'IRTEMPERATURE', 'MEANMAG', 'MEANGYR', 'STDGYR', \
                  'STDACC', 'MAXACC', 'MAXGYR', 'MAC', 'WLATITUDE', 'WLONGITUDE', 'ACCURACY', 'TIMESTAMP']]
                df_to_save.loc[:, 'NID'] = pd.Series([cur_nid]*len(df_to_save), index=df_to_save.index)
                ts_list = list(df_cleaned['TIMESTAMP'].values)
                sgt_list = []
                for ts in ts_list:
                    # convert UTC timestamp to SGT time tuple
                    sgt_list.append(str(datetime.datetime.fromtimestamp(ts + 3600 * 8, pytz.utc)))
                df_to_save.loc[:, 'SGT'] = pd.Series(sgt_list, index=df_to_save.index)
                df_to_save.to_csv(df_table_file)
            except IOError as err:
                logging.error("Failed to save the raw data frame into csv file: %s" % err.strerror)
                sys.exit(10)

        """ save time-series data frame into DB table """
        if save_df_2db:
            tablename = "allweeks_clean"
            print("Saving data into PSQL DB")
            success_upload = save_data_from_IHPC_to_PSQL(psql_conn, psql_cursor, tablename, df_cleaned, cur_nid)
            if success_upload:
                logging.warning("Saving full data into PSQL successful!")

        if save_processed_df or save_map:

            df_processed = df_cleaned.copy()
            # clean data to reduce noise
            logging.info("Clean data")
            clean_geo_data(df_processed)
            # calculate additional features
            logging.info("Calculate features")
            success_feat_calc = \
                calc_geo_time_features(df_processed, cur_queried_date_tuple.strftime("%Y%m%d"), params.window_size)
            if not success_feat_calc:
                return None

            # calculate second-level features of IMU features
            logging.info("Process IMU features")
            calc_extra_features(df_processed, params.window_size)

            """ save time-series data frame into a csv file if enabled """
            if save_processed_df:
                df_table_file = processed_df_table_folder + "/df_table_" + str(cur_nid) + "_" + str(
                    queried_date) + "_" + str(num_pt) + "pts" + ".csv"
                try:
                    df_processed.to_csv(df_table_file)
                except IOError as err:
                    logging.error("Failed to save the data frame into csv file: %s" % err.strerror)
                    sys.exit(10)

        exe_time = timeit.default_timer() - exe_start
        logging.warning("EXECUTING TIME of " + str(cur_nid) + ": " + str(exe_time))
        return None

    # create logger
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=log_level, stream=sys.stdout)

    # saving settins
    save_raw_df = True
    save_df_2db = False
    save_processed_df = False
    save_map = False

    # remember start time for performance analysis
    start_time = time.time()

    # load list of device IDs/dates from file
    logging.info("Load device IDs")

    try:
        nid_date_df = pd.DataFrame.from_csv(device_file_folder + node_date_file)
        nid_list = list(set(nid_date_df.NID))
        # with open(device_file_folder + node_date_file, 'r') as csv_file:
        #     for line in csv_file:
        #         if line.strip():
        #             tmp_strs = line.strip().split(',')
        #             if tmp_strs[0].isdigit():
        #                 device_ids.append(int(tmp_strs[0]))
        #                 if len(tmp_strs) == 1:
        #                     # if no desired date list in the file, use the default date
        #                     queried_dates.append(query_date_default)
        #                 else:
        #                     queried_date_tuple = datetime.datetime.strptime(tmp_strs[1], "%Y%m%d")
        #                     queried_dates.append(queried_date_tuple.strftime("%Y-%m-%d"))
    except IOError as e:
        logging.error("Failed to load device IDs: %s" % e.strerror)
        sys.exit(10)
    print("queried nid: ", str(nid_list))

    df_table_folder = "../data/data_frame_from_IHPC"
    # creat the folder storing data frame if not existing
    if save_raw_df:
        raw_df_table_folder = df_table_folder + "/raw_df"
        if not os.path.exists(raw_df_table_folder):
            os.makedirs(raw_df_table_folder)

    # creat the folder storing data frame if not existing
    if save_processed_df:
        processed_df_table_folder = df_table_folder + "/processed_df"
        if not os.path.exists(processed_df_table_folder):
            os.makedirs(processed_df_table_folder)

    # create a engine connected to PSQL if saving data
    psql_conn = None
    if save_df_2db:
        con_com = """dbname=""" + params.dbname_str + """ user='postgres' password='""" + \
                  params.dbpw_str + """' host='""" + params.dbhost + """'"""
        psql_conn = psycopg2.connect(con_com)
        psql_cursor = psql_conn.cursor()

    # create the folder storing track maps if not existing
    if save_map:
        track_map_folder = "track_maps/AllDay_fromIHPC"
        if not os.path.exists(track_map_folder):
            os.makedirs(track_map_folder)

    for n in nid_list:
        cur_date_list = nid_date_df[nid_date_df['NID'] == n]['DATE']
        for date in cur_date_list:
            try:
                process(n, date)
            except:
                e = traceback.format_exc()
                logging.error("Processing nid %d failed: %s" % (n, e))
    logging.info("---Processed data for %d nodes in %.2f seconds ---" % (len(nid_list), time.time() - start_time))

    if psql_conn:
        psql_conn.close()


if __name__ == "__main__":
    main()
