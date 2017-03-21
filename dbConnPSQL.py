""" all functions needed for data transmission between PSQL """

import psycopg2
import logging
import calendar
import pandas as pd
import numpy as np
from util import chunks
from datetime import timedelta, datetime
import pytz


def insertTrainingInfo2016(conn, cur, tablename, data):
    try:
        logging.info("DB Connected to save training information")
        insertQuery = """INSERT into """ + tablename + """ (ts,classification_type,label_type,len_training,len_test,cv_accuracy,report_path,classifier_type,feature_list,classifier_path,notes)""" \
                                                       """ VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) """
        cur.execute(insertQuery, data)
        conn.commit()
        return True
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return False


def getMainDataPSQL2015(cur, tablename, nid, analysis_date, current_date):
    """ Retrieve raw hardware data for device nid for the specified time range
        from PSQL DB: nse
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        allQuery = """SELECT * from """ + tablename + """ WHERE nid=""" + str(
            nid) + """ AND (ts>='""" + analysis_date + """ 00:00:00' AND ts<'""" + current_date + """ 00:00:00') ORDER BY ts"""
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            raw_ts = rawColumns[2]
            unix_ts = []
            str_ts = []
            for ts in raw_ts:
                unix_ts.append(calendar.timegm(ts.timetuple()) - 8 * 3600)  # convert SGT to unix timestamps
                str_ts.append(str(ts))
            df = pd.DataFrame.from_items(
                [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('SGT', str_ts), ('TIMESTAMP', unix_ts),
                 ('MODE', rawColumns[6]), ('CMODE', rawColumns[7]), \
                 ('STEPS', rawColumns[10]), ('WLATITUDE', rawColumns[14]), ('WLONGITUDE', rawColumns[15]),
                 ('ACCURACY', rawColumns[16])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def getLabelledVehicleData(cur, mode, tablename, label_type='manual', max_num=None):
    """ Retrieve raw hardware data for device nid for the specified time range
        from PSQL DB: nse
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        if label_type == 'manual':
            mode2check = "manual_label_mode"
            special_condition = """and app_label_finish='f' and users_id=1 ORDER BY nid,analyzed_date,trip_num"""
        elif label_type == 'google':
            mode2check = "google_label_mode"
            special_condition = """and google_label_finish='t' and google_failed_reason is NULL and app_label_finish='f' and (manual_label_mode='' or manual_label_mode is NULL) ORDER BY nid,analyzed_date,trip_num"""

        if (mode == 'all'):  # get all modes
            allQuery = """SELECT * FROM """ + tablename + """ WHERE (""" + mode2check + """ LIKE '%4%' OR """ + mode2check + """ LIKE '%5%' OR """ + mode2check + """ LIKE '%6%') """ + special_condition
        elif (mode == 'metro'):  # get only particular modes
            allQuery = """SELECT * FROM """ + tablename + """ WHERE (""" + mode2check + """ LIKE '%4%') """ + special_condition
        elif (mode == 'bus'):  # get only particular modes
            allQuery = """SELECT * FROM """ + tablename + """ WHERE (""" + mode2check + """ LIKE '%5%') """ + special_condition
        elif (mode == 'car'):  # get only particular modes
            allQuery = """SELECT * FROM """ + tablename + """ WHERE (""" + mode2check + """ LIKE '%6%') """ + special_condition
        else:
            logging.warning('input mode is wrong!')
            return None
        if max_num is not None:
            allQuery = allQuery + """ LIMIT """ + str(max_num)
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            df = pd.DataFrame.from_items(
                [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('analyzed_date', rawColumns[2]),
                 ('tot_num_trips', rawColumns[3]), ('trip_num', rawColumns[4]), ('home_loc', rawColumns[5]), \
                 ('school_loc', rawColumns[6]), ('tot_dist_km', rawColumns[9]), ('tot_dura_s', rawColumns[10]),
                 ('google_label_mode', rawColumns[13])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def getLabelledDataAll(cur, tablename, label_type, max_num=None):
    """ Retrieve raw hardware data for device nid for the specified time range
        from PSQL DB: nse_2016
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        if label_type == 'manual':
            search_condition = "app_label_finish='f' and manual_label_mode!='' and users_id=1"
        elif label_type == 'google':
            search_condition = "google_label_finish='t' and google_failed_reason is NULL"
        elif label_type == 'app':
            search_condition = "app_label_finish='t'"
        allQuery = """SELECT * FROM """ + tablename + """ WHERE """ + search_condition + """ ORDER BY nid,analyzed_date,trip_num"""
        # allQuery = """SELECT * FROM """+tablename+""" WHERE """+search_condition+""" ORDER BY nid,analyzed_date,trip_num DESC""" # used only for testing vehicle-or-not smoothing
        if max_num is not None:
            allQuery = allQuery + """ LIMIT """ + str(max_num)
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            df = pd.DataFrame.from_items(
                [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('analyzed_date', rawColumns[2]),
                 ('tot_num_trips', rawColumns[3]), ('trip_num', rawColumns[4]), ('home_loc', rawColumns[5]), \
                 ('school_loc', rawColumns[6]), ('tot_dist_km', rawColumns[9]), ('tot_dura_s', rawColumns[10]),
                 ('google_label_mode', rawColumns[13])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


# def getRawData(cur, nid, ids):
#     """ Retrieve raw hardware data for device nid for the specified time range
#         from PSQL DB: nse
#     """
#     try:
#         logging.info("PSQL DB Connected to fetch data")
#         allQuery = """SELECT * FROM marchpilot WHERE id=ANY(%s) AND nid="""+str(nid) 
#         cur.execute(allQuery,(ids.tolist(),))
#         dataAll = cur.fetchall()
#         if len(dataAll)>0:
#             rawColumns = zip(*dataAll)
#             raw_ts = rawColumns[2]
#             unix_ts = []
#             str_ts = []
#             for ts in raw_ts:
#                 unix_ts.append(calendar.timegm(ts.timetuple())-8*3600) # convert SGT to unix timestamps
#                 str_ts.append(str(ts))
#             df = pd.DataFrame.from_items([('ID',rawColumns[0]),('NID',rawColumns[1]),('SGT',str_ts),('TIMESTAMP',unix_ts),\
#                 ('HUMIDITY',rawColumns[4]),('LIGHT',rawColumns[5]),('MODE',rawColumns[6]),('CMODE',rawColumns[7]), \
#                 ('NOISE',rawColumns[8]),('PRESSURE',rawColumns[9]),('STEPS',rawColumns[10]),('TEMPERATURE',rawColumns[11]),\
#                 ('IRTEMPERATURE',rawColumns[12]),('MEANMAG',rawColumns[13]),('MEANGYR',rawColumns[14]),('STDGYR',rawColumns[15]),\
#                 ('STDACC',rawColumns[16]),('MAXACC',rawColumns[17]),('MAXGYR',rawColumns[19]),\
#                 ('MAC',rawColumns[20]),('WLATITUDE',rawColumns[21]),('WLONGITUDE',rawColumns[22]),('ACCURACY',rawColumns[23])])
#             return df
#         else:
#             logging.warning('No data from the DB!')
#             return None
#     except psycopg2.DatabaseError, e:
#         logging.error(e)
#         return None

def getDeviceID(cur, tablename, analysis_date, current_date):
    """ Get a list of device IDs from database
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        allQuery = """SELECT nid from """ + tablename + """ WHERE sgt>='""" + analysis_date + """ 00:00:00' AND sgt<'""" + current_date + """ 00:00:00' """
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            logging.info('Succefully get device IDs from the DB!')
            return list(rawColumns[0])
        else:
            logging.warning('No device IDs from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def getMainDataPSQL2016(cur, tablename, nid, analysis_date, current_date):
    """ Retrieve raw hardware data for device nid for the specified time range
        from PSQL DB: nse_2016
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        allQuery = """SELECT * from """ + tablename + """ WHERE nid=""" + str(
            nid) + """ AND (sgt>='""" + analysis_date + """ 00:00:00' AND sgt<'""" + current_date + """ 00:00:00') ORDER BY sgt"""
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            raw_ts = rawColumns[2]
            unix_ts = []
            str_ts = []
            for ts in raw_ts:
                unix_ts.append(calendar.timegm(ts.timetuple()) - 8 * 3600)  # convert SGT to unix timestamps
                str_ts.append(str(ts))
            df = pd.DataFrame.from_items(
                [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('SGT', str_ts), ('TIMESTAMP', unix_ts), \
                 ('HUMIDITY', rawColumns[4]), ('LIGHT', rawColumns[5]), ('MODE', rawColumns[6]),
                 ('CMODE', rawColumns[7]), \
                 ('NOISE', rawColumns[8]), ('PRESSURE', rawColumns[9]), ('STEPS', rawColumns[10]),
                 ('TEMPERATURE', rawColumns[11]), \
                 ('IRTEMPERATURE', rawColumns[12]), ('MEANMAG', rawColumns[13]), ('MEANGYR', rawColumns[14]),
                 ('STDGYR', rawColumns[15]), \
                 ('STDACC', rawColumns[16]), ('MAXACC', rawColumns[17]), ('MAXGYR', rawColumns[19]), \
                 ('MAC', rawColumns[20]), ('WLATITUDE', rawColumns[21]), ('WLONGITUDE', rawColumns[22]),
                 ('ACCURACY', rawColumns[23])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def getTripDataPSQL2016(cur, tablename, nid, analysis_date, current_date, df):
    """ Retrieve extra pt-level data for device nid for the specified time range
        from PSQL DB: nse_2016
        And attach to the given df
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        allQuery = """SELECT * from """ + tablename + """ WHERE nid=""" + str(
            nid) + """ AND (sgt>='""" + analysis_date + """ 00:00:00' AND sgt<'""" + current_date + """ 00:00:00') ORDER BY sgt"""
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            df2 = pd.DataFrame.from_items(
                [('ID_extra', rawColumns[0]), ('lat_clean', rawColumns[3]), ('lon_clean', rawColumns[4]),
                 ('triplabel', rawColumns[5]), \
                 ('poilabel', rawColumns[6]), ('ccmode', rawColumns[7]), ('gt_mode_manual', rawColumns[8]),
                 ('gt_mode_google', rawColumns[9]), ('gt_mode_app', rawColumns[10])])
            if len(df) == len(df2):
                result = pd.concat([df, df2], axis=1)
                return result
            else:
                logging.error('Length of the two df is not equal!')
                return None
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def splitZip2Save(cur, tablename, zip2save):
    """ function used to check whether a particular (nid,date) exists inside the table
        if yes, use update to save it; if no, use insert to save it.

        zip2save = zip(nids,sgts,lats,lons,trip_labels,poi_labels,ccmodes,ids)
    """
    zip2update = []
    zip2insert = []
    try:
        readQuery = """SELECT id,nid,sgt from """ + tablename + """ WHERE nid=""" + str(
            zip2save[0][0]) + """ AND (sgt>='""" + str(zip2save[0][1]) + """' AND sgt<='""" + str(
            zip2save[-1][1]) + """') ORDER BY sgt"""
        cur.execute(readQuery)
        rawData = cur.fetchall()
        if len(rawData) > 0:
            logging.warning("There's data of this node on this day existing inside the table")
            rawColumns = zip(*rawData)
            ids = rawColumns[0]
            nids = rawColumns[1]
            sgt_tuples = rawColumns[2]
            # change datetime to string
            sgts = []
            for sgt_tuple in sgt_tuples:
                sgts.append(str(sgt_tuple))
            # print ids[:5]
            # print nids[:5]
            # print sgts[:5]
            # print zip2save[:5]
            for item in zip2save:
                # go through each item inside zip2save to split to inserting and updating
                if item[0] in nids and item[1] in sgts:
                    # print "Existing sample found"
                    # current (nid,date) already exists
                    # check whether the existing id is same with the id inside the main table
                    if item[-1] != ids[sgts.index(item[1])]:
                        logging.error("Same (nid,date) but different id!")
                    # else:
                    #     logging.error("hello!!!!!~~~~~~~~~~~~~~~~~~~~~~")
                    # add this item to the updating list
                    zip2update.append(item)
                else:
                    zip2insert.append(item)
        else:
            # if nothing exists in the table for this node on this day, all should be inserted
            zip2insert = zip2save

        return zip2update, zip2insert
    except psycopg2.DatabaseError, e:
        logging.error("Failed to get existing data from DB!")
        logging.error(e)
        return None, None


def save_extra_PSQL_2016(conn, cur, tablename, data_frame):
    """ Save extra point-level information into DB
        Input:
        conn: DB connection
        cur: DB cursor
        tablename: DB table to save the point-level data
        data_frame: df of all point-level data

        Return True if successful and False otherwise.

    """
    ids = list(data_frame['ID'].values)  # convert IDs column to a list
    nids = list(data_frame['NID'].values)
    sgts = list(data_frame['SGT'].values)
    lats = list(data_frame['WLATITUDE'].values)
    lons = list(data_frame['WLONGITUDE'].values)
    ccmodes = list(data_frame['CCMODE'].values)
    poi_labels = list(data_frame['POI_LABEL'].values)
    trip_labels = list(data_frame['TRIP_LABEL'].values)

    # zip all list together
    zip2save = zip(nids, sgts, lats, lons, trip_labels, poi_labels, ccmodes, ids)

    # Separate the entire zip list to part to insert and part to update
    zip2update, zip2insert = splitZip2Save(cur, tablename, zip2save)
    logging.warning("Total length to save: " + str(len(zip2save)))
    logging.warning("Length to update: " + str(len(zip2update)))
    logging.warning("Length to insert: " + str(len(zip2insert)))

    # insert and update
    update_success = None
    insert_success = None
    if zip2update:
        # use update if already exists
        try:
            logging.warning("DB Connected to update extra columns")
            setQuery = """UPDATE """ + tablename + """ set nid=%s, sgt=%s, lat=%s, \
            lon=%s, triplabel=%s, poilabel=%s, ccmode=%s where id=%s"""
            cur.executemany(setQuery, zip2update)
            conn.commit()
            update_success = True
        except psycopg2.DatabaseError, e:
            logging.error(e)
            update_success = False
            logging.error("Updating failed!")
    if zip2insert:
        # otherwise use insert
        try:
            logging.warning("DB Connected to insert extra columns")
            insertQuery = """INSERT INTO """ + tablename + """ (nid,sgt,lat,lon,triplabel,\
                poilabel,ccmode,id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
            cur.executemany(insertQuery, zip2insert)
            conn.commit()
            insert_success = True
        except psycopg2.DatabaseError, e:
            logging.error(e)
            insert_success = False
            logging.error("Inserting failed!")
    if (zip2update is None and zip2insert is None) or (update_success is False) or (insert_success is False):
        return False
    else:
        return True


def checkNidDateExistence(cur, tablename, nid, analyzed_date):
    """ function used to check whether a particular (nid,date) pair exists 
    inside the table

    """
    existence = False
    manually_labeled = False
    app_labeled = False
    google_labeled_failed = False
    google_labeled_trusted = False
    try:
        logging.warning("Checking whether the trip info exists.")
        readQuery = """SELECT trip_num, manual_label_mode, app_label_finish, google_label_finish, google_failed_reason FROM """ + tablename + """ WHERE nid=""" + str(
            nid) + """ and analyzed_date='""" + analyzed_date + """'"""
        cur.execute(readQuery)
        rawData = cur.fetchall()
        if len(rawData) > 0:
            existence = True
            logging.warning("Previous number of trips: " + str(len(rawData)))
            rawColumns = zip(*rawData)
            manual_label_modes = rawColumns[1]
            app_label_finish_list = rawColumns[2]
            google_label_finish_list = rawColumns[3]
            google_failed_reason_list = rawColumns[4]
            for item in manual_label_modes:
                if item != '':
                    manually_labeled = True
            for item in app_label_finish_list:
                if item == True:
                    app_labeled = True
            for idx, item in enumerate(google_label_finish_list):
                if item == True:
                    google_labeled_failed = True
                    if google_failed_reason_list[idx] is None:
                        google_labeled_trusted = True
                        google_labeled_failed = False
                        break
        else:
            logging.warning("The trip info doesn't exist.")
        return existence, manually_labeled, app_labeled, google_labeled_failed, google_labeled_trusted
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None, None, None, None, None


def save_tripsummary_PSQL_2016(conn, cur, tablename_trip, tablename_extra, trips_dict):
    """ Save extra point-level information into DB
        Input:
        conn: DB connection
        cur: DB cursor
        tablename_trip: DB table to save the trip dictionary data
        trips_dict: dictionary of trip summaries

        Return True if successful and False otherwise.

    """
    nids = [trips_dict['nid']] * len(trips_dict['trip_num'])
    dates = [trips_dict['analyzed_date']] * len(trips_dict['trip_num'])
    tot_num_trips = [trips_dict['tot_num_trips']] * len(trips_dict['trip_num'])
    if not trips_dict['home_loc'] == [None, None]:
        home_loc = [trips_dict['home_loc']] * len(trips_dict['trip_num'])
    else:
        home_loc = [[]] * len(trips_dict['trip_num'])
    if not trips_dict['school_loc'] == [None, None]:
        school_loc = [trips_dict['school_loc']] * len(trips_dict['trip_num'])
    else:
        school_loc = [[]] * len(trips_dict['trip_num'])
    trip_num = trips_dict['trip_num']
    start_poi_loc = trips_dict['start_poi_loc']
    end_poi_loc = trips_dict['end_poi_loc']
    start_sgt = trips_dict['start_sgt']
    end_sgt = trips_dict['end_sgt']
    tot_dist = trips_dict['tot_dist(km)']
    tot_dura = trips_dict['tot_dura(s)']
    valid_loc_perc = trips_dict['valid_loc_perc']
    num_pt = trips_dict['num_pt']
    manual_label_modes = []
    manual_label_strs = []
    manual_label_finishs = []
    users_ids = []
    time_modified_list = []
    google_label_modes = []
    google_failed_reasons = []
    google_label_finishs = []

    # check existence by id/nid first
    existence, manually_labeled, app_labeled, google_labeled_failed, google_labeled_trusted = checkNidDateExistence(cur,
                                                                                                                    tablename_trip,
                                                                                                                    nids[
                                                                                                                        0],
                                                                                                                    dates[
                                                                                                                        0])
    if existence is None:
        logging.error("Existence checking failed!")
        return False
    try:
        if manually_labeled:
            # need to recreate the manually labeled trip-level modes for the new trips
            logging.warning("There are manual labels existing. Start recreating.")
            cur_date_tuple = datetime.strptime(dates[0], "%Y%m%d")
            one_day_after = (cur_date_tuple + timedelta(days=1)).strftime("%Y%m%d")
            # get the triplabel and pt-level label for the whole day
            allQuery = """SELECT triplabel,gt_mode_manual from """ + tablename_extra + """ WHERE nid=""" + str(
                nids[0]) + """ AND (sgt>='""" + dates[
                           0] + """ 00:00:00' AND sgt<'""" + one_day_after + """ 00:00:00') ORDER BY sgt"""
            cur.execute(allQuery)
            dataAll = cur.fetchall()
            if len(dataAll) > 0:
                rawColumns = zip(*dataAll)
                triplabels = np.array(rawColumns[0])
                gt_mode_manual_list = np.array(rawColumns[1])
                # go through each trip to get the trip-level modes
                for item in trip_num:
                    mode_str_cur_trip = ''
                    # get all labels for this trip
                    gt_mode_manual_cur_trip = gt_mode_manual_list[triplabels == item].tolist()
                    # get mode chunks of this trip
                    mode_chunks = chunks(gt_mode_manual_cur_trip, include_values=True)
                    for mode_chunk in mode_chunks:
                        if mode_chunk[2] is not None:
                            mode_str_cur_trip += str(mode_chunk[2])
                    manual_label_modes.append(mode_str_cur_trip)
                    manual_label_finishs.append('t')
                    manual_label_strs.append('recreated')
                    users_ids.append(1)
                    time_modified_list.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                logging.error("Failed to get the pt-level manual labels")

        if google_labeled_trusted:
            # need to recreate the automatically labeled trip-level modes for the new trips, if the labels are trusted
            logging.warning("There are trusted automatic labels existing. Start recreating.")
            cur_date_tuple = datetime.strptime(dates[0], "%Y%m%d")
            one_day_after = (cur_date_tuple + timedelta(days=1)).strftime("%Y%m%d")
            # get the triplabel and pt-level label for the whole day
            allQuery = """SELECT triplabel,gt_mode_google from """ + tablename_extra + """ WHERE nid=""" + str(
                nids[0]) + """ AND (sgt>='""" + dates[
                           0] + """ 00:00:00' AND sgt<'""" + one_day_after + """ 00:00:00') ORDER BY sgt"""
            cur.execute(allQuery)
            dataAll = cur.fetchall()
            if len(dataAll) > 0:
                rawColumns = zip(*dataAll)
                triplabels = np.array(rawColumns[0])
                gt_mode_google_list = np.array(rawColumns[1])
                # go through each trip to get the trip-level modes
                for item in trip_num:
                    mode_str_cur_trip = ''
                    # get all labels for this trip
                    gt_mode_google_cur_trip = gt_mode_google_list[triplabels == item].tolist()
                    # check the percentage of the labeled samples
                    None_mode_cnt = 0
                    for mode_item in gt_mode_google_cur_trip:
                        if mode_item is None:
                            None_mode_cnt += 1
                    if None_mode_cnt > 0.3 * len(gt_mode_google_cur_trip):
                        # if too many samples don't have automatic labels
                        google_label_modes.append(None)
                        google_label_finishs.append('t')
                        google_failed_reasons.append('Too few labels while recreating')
                    else:
                        # get mode chunks of this trip
                        mode_chunks = chunks(gt_mode_google_cur_trip, include_values=True)
                        for mode_chunk in mode_chunks:
                            if mode_chunk[2] is not None:
                                mode_str_cur_trip += str(mode_chunk[2])
                        google_label_modes.append(mode_str_cur_trip)
                        google_label_finishs.append('t')
                        google_failed_reasons.append(None)
            else:
                logging.error("Failed to get the pt-level manual labels")
        elif google_labeled_failed:
            logging.warning("There are failed automatic labels existing. Save as failed auto-labeling.")
            google_label_modes = [None] * len(trip_num)
            google_label_finishs = ['t'] * len(trip_num)
            google_failed_reasons = ['Failed before recreating'] * len(trip_num)
        else:
            google_label_modes = [None] * len(trip_num)
            google_label_finishs = ['f'] * len(trip_num)
            google_failed_reasons = [None] * len(trip_num)

        app_labeled_list = [app_labeled] * len(trip_num)

        if existence:
            # delete the old data if already exists because the new one might have diff number of trips
            logging.warning("Trip exists! DB Connected to delete existing trip summaries")
            deleteQuery = """DELETE FROM """ + tablename_trip + """ WHERE nid=""" + str(
                nids[0]) + """ and analyzed_date='""" + dates[0] + """'"""
            cur.execute(deleteQuery)
            conn.commit()

        if manual_label_modes:
            # insert the new data with existing manual labels
            logging.warning("DB Connected to insert new trip summaries with existing manual labels")
            insertQuery = """INSERT INTO """ + tablename_trip + """ (nid,analyzed_date, tot_num_trips,\
                trip_num,home_loc, school_loc, start_poi_loc, end_poi_loc,tot_dist_km, tot_dura_s,\
                start_sgt,end_sgt, valid_loc_perc,num_pt, manual_label_finish, manual_label_str,\
                manual_label_mode, users_id, time_modified, app_label_finish, google_label_mode,\
                google_label_finish,google_failed_reason) \
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            zip2save = zip(nids, dates, tot_num_trips, trip_num, home_loc, school_loc, \
                           start_poi_loc, end_poi_loc, tot_dist, tot_dura, start_sgt, end_sgt, valid_loc_perc, num_pt, \
                           manual_label_finishs, manual_label_strs, manual_label_modes, users_ids, time_modified_list, \
                           app_labeled_list, google_label_modes, google_label_finishs, google_failed_reasons)
            cur.executemany(insertQuery, zip2save)
            conn.commit()
            return True
        else:
            # insert the new data w/o any existing manual labels
            logging.warning("DB Connected to insert new trip summaries w/o any existing manual labels")
            insertQuery = """INSERT INTO """ + tablename_trip + """ (nid,analyzed_date,tot_num_trips,\
                trip_num,home_loc,school_loc,start_poi_loc,end_poi_loc,tot_dist_km,tot_dura_s,\
                start_sgt,end_sgt,valid_loc_perc,num_pt, app_label_finish, google_label_mode,\
                google_label_finish,google_failed_reason) \
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            zip2save = zip(nids, dates, tot_num_trips, trip_num, home_loc, school_loc, \
                           start_poi_loc, end_poi_loc, tot_dist, tot_dura, start_sgt, end_sgt, valid_loc_perc, num_pt, \
                           app_labeled_list, google_label_modes, google_label_finishs, google_failed_reasons)
            cur.executemany(insertQuery, zip2save)
            conn.commit()
            return True

    except psycopg2.DatabaseError, e:
        logging.error(e)
        return False


def checkNidSgtPeriodExistence(cur, tablename, nid, sgt_start, sgt_end):
    """ function used to check whether a particular (nid,ts period) pair exists 
    inside the table

    """
    try:
        logging.warning("Checking whether the raw data exists.")
        readQuery = """SELECT count(*) FROM """ + tablename + """ WHERE nid=""" + str(
            nid) + """ and sgt>='""" + sgt_start + """' and sgt<='""" + sgt_end + """'"""
        cur.execute(readQuery)
        rawData = cur.fetchall()
        logging.warning("Number of lines exist: " + str(rawData[0][0]))
        if rawData[0][0] > 0:
            logging.warning("Same (nid, ts period) exists.")
            return True
        else:
            logging.warning("No same data exists.")
            return False
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def save_data_from_IHPC_to_PSQL(conn, cur, tablename, data_frame, nid):
    """ Save point-level data obtained from IHPC to PSQL
        1) Convert the ts to sgt
        2) Check existence
        3) Delete if exist
        4) Insert

    """
    # add one column inside df as sgt
    ts_list = list(data_frame['TIMESTAMP'].values)
    sgt_list = []
    for ts in ts_list:
        # convert UTC timestamp to SGT time tuple
        sgt_list.append(str(datetime.fromtimestamp(ts + 3600 * 8, pytz.utc)))
    data_frame['SGT'] = sgt_list

    # try select from PSQL using the nid and sgt period to check existence
    if_exist = checkNidSgtPeriodExistence(cur, tablename, nid, sgt_list[0], sgt_list[-1])
    if if_exist is None:
        logging.warning("Existence checking failed!")
        return False

    # remove all if already exists
    if if_exist:
        # delete the old data if already exists because the new one might have diff number of lines
        try:
            logging.warning("DB Connected to delete existing data")
            deleteQuery = """DELETE FROM """ + tablename + """ WHERE nid=""" + str(nid) + """ and sgt>='""" + sgt_list[
                0] + """' and sgt<='""" + sgt_list[-1] + """'"""
            cur.execute(deleteQuery)
            conn.commit()
        except psycopg2.DatabaseError, e:
            logging.error(e)
            return False

    # form the zipped list for saving
    num_lines = len(data_frame)
    data_frame['NID'] = [nid] * num_lines
    zip2save = list(data_frame[['NID', 'SGT', 'HUMIDITY', 'LIGHT', 'MODE', 'CMODE', 'NOISE', \
                                'PRESSURE', 'STEPS', 'TEMPERATURE', 'IRTEMPERATURE', 'MEANMAG', 'MEANGYR', 'STDGYR', \
                                'STDACC', 'MAXACC', 'MAXGYR', 'MAC', 'WLATITUDE', 'WLONGITUDE', 'ACCURACY']].values)

    # insert new data in
    logging.warning("DB Connected to insert new data")
    insertQuery = """INSERT INTO """ + tablename + """ (nid,sgt,hum,light,mode,cmode,noise,\
    press,steps,temp,irtemp,meanmag,meangyr,stdgyr,stdacc,maxacc,\
    maxgyr,mac,lat,lon,acc) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cur.executemany(insertQuery, zip2save)
        conn.commit()
        return True
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return False


def save_dailysummary_PSQL_2016(conn, cur, tablename, daily_summary):
    """ Save extra point-level information into DB
        Input:
        conn: DB connection
        cur: DB cursor
        tablename: DB table to save the point-level data
        daily_summary: summary to save

        Return True if successful and False otherwise.

    """
    # check whether this (nid, date) pair has already existed
    nid = daily_summary['nid']
    analyzed_date = daily_summary['analyzed_date']
    try:
        logging.warning("Checking whether the daily summary exists.")
        readQuery = """SELECT * FROM """ + tablename + """ WHERE nid=""" + str(
            nid) + """ and analyzed_date='""" + analyzed_date + """'"""
        cur.execute(readQuery)
        rawData = cur.fetchall()
    except psycopg2.DatabaseError, e:
        logging.error("Existence checking failed.")
        logging.error(e)
        return False

    if len(rawData) > 0:
        logging.warning("The daily summary exists.")
        try:
            logging.warning("DB Connected to delete existing summary")
            deleteQuery = """DELETE FROM """ + tablename + """ WHERE nid=""" + str(
                nid) + """ and analyzed_date='""" + analyzed_date + """'"""
            cur.execute(deleteQuery)
            conn.commit()
        except psycopg2.DatabaseError, e:
            logging.error("Deleting existing summary checking failed.")
            logging.error(e)
            return False
    else:
        logging.warning("The daily summary doesn't exist.")

    # insert the daily summary
    cols = daily_summary.keys()
    cols_str = str(cols)
    cols_str = cols_str.replace('[', '')
    cols_str = cols_str.replace(']', '')
    cols_str = cols_str.replace("'", '')
    vals = [daily_summary[x] for x in cols]
    vals_str_list = ["%s"] * len(vals)
    vals_str = ", ".join(vals_str_list)
    try:
        logging.info("Inserting the daily summary.")
        # print cur.mogrify("INSERT INTO {tablename} ({cols}) VALUES ({vals_str})".format(tablename = tablename, \
        #     cols = cols_str, vals_str = vals_str), vals)
        cur.execute("INSERT INTO {tablename} ({cols}) VALUES ({vals_str})".format(tablename=tablename, \
                                                                                  cols=cols_str, vals_str=vals_str),
                    vals)
        conn.commit()
        return True
    except psycopg2.DatabaseError, e:
        logging.error("Inserting failed!")
        logging.error(e)
        return False


def getLabelledDataWithNullAppLabel(cur, tablename, label_type, max_num=None):
    """ Retrieve raw hardware data for device nid for the specified time range
        from PSQL DB: nse_2016
    """
    try:
        logging.info("PSQL DB Connected to fetch data")
        if label_type == 'manual':
            search_condition = "app_label_finish='f' and manual_label_mode!='' and manual_label_finish = 't'"
        elif label_type == 'google':
            search_condition = "google_label_finish='t' and google_label_mode is not null and google_failed_reason is " \
                               "NULL and app_label_finish = 'f' "
        else:
            logging.warning("MUST ENTER the correct label_type!!!")
            return None
        allQuery = """SELECT * FROM """ + tablename + """ WHERE """ + search_condition + """ORDER BY nid,analyzed_date,
        trip_num """
        if max_num is not None:
            allQuery = allQuery + """ LIMIT """ + str(max_num)
        cur.execute(allQuery)
        dataAll = cur.fetchall()
        if len(dataAll) > 0:
            rawColumns = zip(*dataAll)
            # df = pd.DataFrame.from_items(
            #     [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('analyzed_date', rawColumns[2]), \
                 # ('tot_num_trips', rawColumns[3]), ('trip_num', rawColumns[4]), ('home_loc', rawColumns[5]), \
                 # ('school_loc', rawColumns[6]), ('tot_dist_km', rawColumns[9]), ('tot_dura_s', rawColumns[10]), \
                 # ('google_label_mode', rawColumns[13])])
            df = pd.DataFrame.from_items(
                [('ID', rawColumns[0]), ('NID', rawColumns[1]), ('analyzed_date', rawColumns[2]), \
                ('tot_num_trips', rawColumns[3]), ('trip_num', rawColumns[4]), ('user_id', rawColumns[14])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def get_nids_with_app_label(cur, table_name):
    try:
        search_condition = "gt_mode_app is not null"
        all_query = """SELECT DISTINCT nid FROM """ + table_name + """ WHERE """ + search_condition
        cur.execute(all_query)
        data_all = cur.fetchall()
        if len(data_all) > 0:
            return set(data_all)
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None


def get_all_app_pt(cur):
    try:
        query = "select * from allweeks_clean cl, (select nid, sgt, gt_mode_app from allweeks_extra where " \
                "gt_mode_app is not null) as ex where " \
                "cl.nid = ex.nid and cl.sgt = ex.sgt order by cl.nid, cl.sgt"
        cur.execute(query)
        data = cur.fetchall()
        if len(data) > 0:
            raw_columns = zip(*data)
            raw_ts = raw_columns[2]
            unix_ts = []
            str_ts = []
            for ts in raw_ts:
                unix_ts.append(calendar.timegm(ts.timetuple()) - 8 * 3600)  # convert SGT to unix timestamps
                str_ts.append(str(ts))
            df = pd.DataFrame.from_items(
                [('ID', raw_columns[0]), ('NID', raw_columns[1]), ('SGT', str_ts), ('TIMESTAMP', unix_ts), \
                 ('HUMIDITY', raw_columns[4]), ('LIGHT', raw_columns[5]), ('MODE', raw_columns[6]),
                 ('CMODE', raw_columns[7]), \
                 ('NOISE', raw_columns[8]), ('PRESSURE', raw_columns[9]), ('STEPS', raw_columns[10]),
                 ('TEMPERATURE', raw_columns[11]), \
                 ('IRTEMPERATURE', raw_columns[12]), ('MEANMAG', raw_columns[13]), ('MEANGYR', raw_columns[14]),
                 ('STDGYR', raw_columns[15]), \
                 ('STDACC', raw_columns[16]), ('MAXACC', raw_columns[17]), ('MAXGYR', raw_columns[19]), \
                 ('MAC', raw_columns[20]), ('WLATITUDE', raw_columns[21]), ('WLONGITUDE', raw_columns[22]),
                 ('ACCURACY', raw_columns[23]), ('GT_MODE_APP', raw_columns[26])])
            return df
        else:
            logging.warning('No data from the DB!')
            return None
    except psycopg2.DatabaseError, e:
        logging.error(e)
        return None
