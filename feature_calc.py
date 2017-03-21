""" all functions needed for calculating features for mode ID """

from itertools import izip
import numpy as np
import pandas as pd
from util import chunks_real, chunks, great_circle_dist, get_hour_SGT, moving_std, moving_dr, moving_ave_velocity
from util import apply_DBSCAN
import TransitHeuristic
import logging
import random
from collections import Counter


# ['ID', 'NID', 'SGT', 'TIMESTAMP', 'HUMIDITY', 'LIGHT', 'MODE', 'CMODE', 'NOISE', 'PRESSURE', 'STEPS', 'TEMPERATURE',
#  'IRTEMPERATURE', 'MEANMAG', 'MEANGYR', 'STDGYR', 'STDACC', 'MAXACC', 'MAXGYR', 'MAC', 'WLATITUDE', 'WLONGITUDE',
#  'ACCURACY', 'ID_extra', 'lat_clean', 'lon_clean', 'triplabel', 'poilabel', 'ccmode', 'gt_mode_manual',
#  'gt_mode_google', 'gt_mode_app', 'ANALYZED_DATE', 'TIME_SGT', 'TIME_DELTA', 'STEPS_DELTA', 'DISTANCE_DELTA',
#  'VELOCITY', 'ACCELERATION', 'MOV_AVE_VELOCITY', 'MOV_AVE_ACCELERATION', 'BUS_DIST', 'METRO_DIST', 'STDMEAN_MAG_WIN',
#  'drMEAN_MAG_WIN', 'STDPRES_WIN', 'GT_MODE_APP', 'NUM_AP', 'is_localized']

# todo try other features.

# todo previous
# todo
# ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'PRESSURE', 'STDPRES_WIN', 'NUM_AP', 'WLATITUDE',
# 'WLONGITUDE', 'is_localized', 'METRO_DIST', 'BUS_DIST', 'STEPS', 'NOISE', 'TIME_DELTA', 'TEMPERATURE',
# 'IRTEMPERATURE', 'HUMIDITY', 'STD_VELOCITY_10MIN', 'MAX_VELOCITY_10MIN', 'VELOCITY', 'TIMESTAMP',
# 'STOP_10MIN', 'STOP_BUSSTOP_10MIN', 'FAST_10MIN']


# ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'PRESSURE', 'STDPRES_WIN', 'NUM_AP', 'METRO_DIST', 'BUS_DIST', 'NOISE']

DL_FEATURES = ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'PRESSURE', 'STDPRES_WIN', 'NUM_AP', 'METRO_DIST',
               'BUS_DIST', 'NOISE', 'STD_VELOCITY_10MIN', 'TEMPERATURE']

WIN_FEATURES = []

WALK_STAT_FEATURES = ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'PRESSURE', 'STDPRES_WIN', 'NUM_AP', 'NOISE',
                      'TIME_DELTA', 'STD_VELOCITY_10MIN']

TRIPLET_FEATURES = ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR', 'PRESSURE', 'STDPRES_WIN', 'METRO_DIST',
                    'STD_VELOCITY_10MIN']

BUS_CAR_FEATURES = ['MOV_AVE_VELOCITY', 'STDACC', 'MEANMAG', 'MAXGYR','PRESSURE', 'STDPRES_WIN', 'BUS_DIST',
                    'METRO_DIST', 'NOISE', 'TEMPERATURE', 'STOP_BUSSTOP_10MIN']
BUS_VELOCITY_THRESHOLD = 5.5
NEAR_BUS_STOP_THRESHOLD = 10

# TODO num_ap
# TODO choose the test sample before balancing
# TODO label_number


def calc_geo_time_features(data_frame, queried_date_str, window_size, high_velocity_thresh=40):
    """Calculate additional features and attributes from the raw hardware
    data. New attributes are added as new columns in the data frame in
    place.

    high_velocity_thresh : maximum threshold for velocities in m/s,
                           higher values are rejected. Default 40m/s
                           (= 144 km/h)

    queried_date_str: date when the data is collected, in format "%Y%m%d"
    """

    # add analyzed date into the data frame
    data_frame['ANALYZED_DATE'] = pd.Series([queried_date_str] * len(data_frame))
    # calculate the SGT time of the day, in hours
    time_SGT = map(lambda x: get_hour_SGT(x), data_frame['TIMESTAMP'].values)
    data_frame['TIME_SGT'] = pd.Series(time_SGT)

    # calculate time delta since the last measurement, in seconds
    consec_timestamps = izip(data_frame[['TIMESTAMP']].values[:-1], data_frame[['TIMESTAMP']].values[1:])
    delta_timestamps = map(lambda x: x[1][0] - x[0][0], consec_timestamps)
    if data_frame['TIME_SGT'][0] < 1.5:
        # add dt to 24 am for the first measurement when first point is within 24 am to 1.5am
        delta_timestamps = [int(data_frame['TIME_SGT'][0] * 3600)] + delta_timestamps
    else:
        # add a zero value for the first measurement when first point is not from 24 am to 1.5am
        delta_timestamps = [0] + delta_timestamps
    data_frame['TIME_DELTA'] = pd.Series(delta_timestamps)
    # check if there's negative delta_t
    ts_array = np.array(delta_timestamps)
    if any(ts_array < 0):
        logging.error("There's negative delta_t from DB!!! Length is: " + str(sum(ts_array < 0)))
        return False

    # calculate steps delta since the last measurement
    consec_steps = izip(data_frame[['STEPS']].values[:-1], data_frame[['STEPS']].values[1:])
    delta_steps = map(lambda x: x[1][0] - x[0][0], consec_steps)
    # filter out negative delta_steps
    delta_steps = [dstep if dstep >= 0 else 0 for dstep in delta_steps]
    # add a zero value for the first measurement where no delta is available
    data_frame['STEPS_DELTA'] = pd.Series([0] + delta_steps)

    # select rows in data frame that have valid locations
    df_validloc = data_frame.loc[~np.isnan(data_frame['WLATITUDE']) & ~np.isnan(data_frame['WLONGITUDE'])]
    # calculate distance delta from pairs of valid lat/lon locations that follow each other
    valid_latlon = df_validloc[['WLATITUDE', 'WLONGITUDE']].values
    dist_delta = map(lambda loc_pair: great_circle_dist(loc_pair[0], loc_pair[1], unit="meters"),
                     izip(valid_latlon[:-1], valid_latlon[1:]))
    # calculate time delta from pairs of valid timestamps
    valid_times = df_validloc['TIMESTAMP'].values
    time_delta = valid_times[1:] - valid_times[:-1]
    # calculate velocity, m/s
    velocity = dist_delta / time_delta

    # create new columns for delta distance, time delta and velocity, initialzied with NaN
    data_frame['DISTANCE_DELTA'] = pd.Series(dist_delta, df_validloc.index[1:])
    data_frame['VELOCITY'] = pd.Series(velocity, df_validloc.index[1:])  # velocity in m/s
    data_frame['ACCELERATION'] = data_frame['VELOCITY'] / data_frame['TIME_DELTA']  # acceleration in m/s^2

    # assign the velocity of those nan-loc points with the latter first valid velocity
    validloc_label = np.isnan(data_frame['WLATITUDE'].values)  # True for points with nan loc
    validloc_label_chunks = chunks(validloc_label, include_values=True)
    for label_chunk in validloc_label_chunks:
        # find True chunks (no loc) and assign the velocity
        if label_chunk[2] and label_chunk[1] != len(data_frame):
            data_frame.loc[data_frame.index[0] + label_chunk[0]:data_frame.index[0] + label_chunk[1] - 1, 'VELOCITY'] = \
                data_frame['VELOCITY'][label_chunk[1]]

    # replace very high velocity values which are due to wifi
    # localizations errors with NaN in VELOCITY column
    idx_too_high = np.where(data_frame['VELOCITY'].values > high_velocity_thresh)[0].tolist()
    idx_too_high = [item + data_frame.index[0] for item in idx_too_high]
    idx_bef_too_high = (np.array(idx_too_high) - 1).tolist()
    data_frame.loc[idx_too_high, ['WLATITUDE', 'WLONGITUDE', 'DISTANCE_DELTA', 'VELOCITY']] = np.nan
    data_frame.loc[idx_bef_too_high, ['WLATITUDE', 'WLONGITUDE', 'DISTANCE_DELTA', 'VELOCITY']] = np.nan

    # calculate the moving average of velocity, m/s
    LARGE_TIME_JUMP = 60  # seconds
    velocity_all = data_frame['VELOCITY'].values
    moving_ave_velocity_all = moving_ave_velocity(velocity_all, np.array(delta_timestamps), LARGE_TIME_JUMP,
                                                  window_size)
    moving_ave_acc_all = moving_ave_velocity(data_frame['ACCELERATION'].values, np.array(delta_timestamps),
                                             LARGE_TIME_JUMP, window_size)
    data_frame['MOV_AVE_VELOCITY'] = pd.Series(moving_ave_velocity_all)  # velocity in m/s
    data_frame['MOV_AVE_ACCELERATION'] = pd.Series(moving_ave_acc_all)  # acceleration in m/s^2

    return True


def pt_during_selected_time(df, cur_pt_idx, time):
    """
    return all the points in the last n second in the form of data frame
    :param df: the whole data set
    :param cur_pt_idx: the current idx of point in df
    :param time: In the last n second
    :return: all the points in the last n second
    """
    result_idx = 0
    cur_time = df.iloc[cur_pt_idx]['TIMESTAMP']

    stop_count = 0
    stop_bus_stop_count = 0
    fast_count = 0
    for i in range(cur_pt_idx-1, -1, -1):
        if cur_time - df.iloc[i]['TIMESTAMP'] > time:
            result_idx = i+1
            break
        if df.iloc[i]['VELOCITY'] < BUS_VELOCITY_THRESHOLD:
            stop_count += 1
            if df.iloc[i]['BUS_DIST'] < NEAR_BUS_STOP_THRESHOLD:
                stop_bus_stop_count += 1
        else:
            fast_count += 1
    return df.iloc[result_idx:cur_pt_idx], stop_count, stop_bus_stop_count, fast_count


def calc_extra_features(data_frame, window_size):
    """Calculate additional features from IMU features
    """

    # calculate the moving std, diff and dr of MEANMAG
    mean_mag_all = data_frame['MEANMAG'].values
    std_mean_mag_all = moving_std(mean_mag_all, window_size)
    data_frame['STDMEAN_MAG_WIN'] = pd.Series(std_mean_mag_all)
    dr_mean_mag_all = moving_dr(mean_mag_all, window_size)
    data_frame['drMEAN_MAG_WIN'] = pd.Series(dr_mean_mag_all)
    # add the moving std of pressure
    pressure_all = data_frame['PRESSURE'].values
    # 5 windows
    std_pres_all = moving_std(pressure_all, window_size)
    data_frame['STDPRES_WIN'] = pd.Series(std_pres_all)

    # add one column for labels
    # select rows in data frame that are during deep night
    if 'GT_MODE_APP' not in data_frame:
        data_frame['GT_MODE_APP'] = pd.Series([])

    # add number of access points
    if 'MAC' in data_frame:
        mac_list = data_frame['MAC'].values.tolist()
        num_ap = []
        for cur_mac in mac_list:
            # mac_rssi in the format ['mac:xxxx RSSI:-xx']
            count = 0
            for mac_rssi in cur_mac.split(','):
                if ('MAC' in mac_rssi) and ('RSSI' in mac_rssi):
                    loc = mac_rssi.index('RSSI:')
                    try:
                        rssi = int(mac_rssi[loc + 5:loc + 8])
                        if rssi > -60:
                            count += 1
                    except Exception as inst:
                        print inst
                        print "mac_rssi: " + str(mac_rssi)
                        print "loc: " + str(loc)
                        continue
                else:
                    continue
            num_ap.append(count)
        data_frame['NUM_AP'] = pd.Series(num_ap)

    # add valid localization labels
    # has valid location information, is_localized = 1
    # nan location info, is_localized = 0
    data_frame['is_localized'] = pd.Series([0] * len(data_frame))
    data_frame.loc[~np.isnan(data_frame['WLATITUDE']), ['is_localized']] = 1


def clean_geo_data(data_frame, valid_lat_low=1.0,
                   valid_lat_up=2.0, valid_lon_low=103.0, valid_lon_up=105.0,
                   location_accuracy_thresh=300):
    """Clean data frame by replacing entries with impossible values with
    'null values' np.nan. The method does not remove rows to keep the
    original data intact. Each predictor that is using the fetures is
    responsible for checking that the features are valid. Changes are
    made in-place. There is no return value.

    valid_lat_low : float value to signal a possible minimum latitude. Default 1.0
    valid_lat_up : float value to signal a possible maximum latitude. Default 2.0
    valid_lon_low : float value to signal a possible minimum longitude. Default 103.0
    valid_lon_up : float value to signal a possible maximum longitude. Default 105.0
    location_accuracy_thresh : upper threshold on the location
                               accuracy in meters beyond which we
                               treat the location as
                               missing. Default 1000
    """

    def invalid_location(acc):
        """Select rows with invalid accuracy. acc is a data frame column,
        returns a data frame of boolean values."""
        return (acc < 0) | (acc > location_accuracy_thresh)

    # replace invalid lat/lon values with NaN
    data_frame.loc[(data_frame['WLATITUDE'] < valid_lat_low) | (data_frame['WLATITUDE'] > valid_lat_up),
                   ['WLATITUDE', 'WLONGITUDE']] = np.nan
    data_frame.loc[(data_frame['WLONGITUDE'] < valid_lon_low) | (data_frame['WLONGITUDE'] > valid_lon_up),
                   ['WLATITUDE', 'WLONGITUDE']] = np.nan

    # replace locations with poor accuracy or negative accuracy values
    # (signal for invalid point) with NaN and set velocity as invalid
    if 'ACCURACY' in data_frame.columns:
        data_frame.loc[invalid_location(data_frame['ACCURACY']),
                       ['WLATITUDE', 'WLONGITUDE']] = np.nan


def rm_int_ts_outliers(data_frame):
    """ rm_int_ts_outliers() helpes to remove interleaved ts and outliers of sensor readings from raw df
        it refers to Sandra's cleaning code

    """
    # sort the df by ts
    df_cleaned = data_frame.sort_values('TIMESTAMP', ascending=True)

    # go through rows and record rows to remove based on step diff and other sensor readings
    num_samples = len(df_cleaned)
    # check step diff and find interleaved ts
    row_to_remove = []
    i_row = 1
    while i_row < num_samples:
        if df_cleaned.iloc[i_row]['STEPS'] - df_cleaned.iloc[i_row - 1]['STEPS'] < 0:
            startsteps = df_cleaned.iloc[i_row - 1]['STEPS']
            j = i_row
            while not (not (j < num_samples) or not (df_cleaned.iloc[j]['STEPS'] - startsteps < 0)):
                row_to_remove.append(j)
                j += 1
            i_row = j
        i_row += 1
    logging.warning("Number of rows to remove due to interleaved ts: " + str(len(row_to_remove)))
    row_to_stay = list(set(range(0, num_samples)) - set(row_to_remove))
    print "length of row_to_stay: " + str(len(row_to_stay))

    # remove those rows
    df_cleaned = df_cleaned.iloc[row_to_stay]

    # check outliers
    len_before = len(df_cleaned)
    df_cleaned = df_cleaned.loc[(df_cleaned['HUMIDITY'] >= 0) & (df_cleaned['HUMIDITY'] <= 100) &
                                (df_cleaned['LIGHT'] >= 0) & (df_cleaned['LIGHT'] <= 2550) &
                                (df_cleaned['MODE'] >= 0) & (df_cleaned['MODE'] <= 6) &
                                (df_cleaned['CMODE'] >= -1) & (df_cleaned['CMODE'] <= 6) &
                                (df_cleaned['NOISE'] >= 0) & (df_cleaned['NOISE'] <= 150) &
                                (df_cleaned['PRESSURE'] >= 0) & (df_cleaned['PRESSURE'] <= 200000) &
                                (df_cleaned['STEPS'] >= 0) & (df_cleaned['STEPS'] <= 1000000) &
                                (df_cleaned['TEMPERATURE'] >= 0) & (df_cleaned['TEMPERATURE'] <= 80) &
                                (df_cleaned['IRTEMPERATURE'] >= 0) & (df_cleaned['IRTEMPERATURE'] <= 80)]
    len_after = len(df_cleaned)
    logging.warning("Number of rows to remove due to outliers: " + str(len_before - len_after))
    return df_cleaned


def is_vehicle_smoothing(df_trip, non_vehi_seg_min_dura=1 * 60, vehi_seg_min_dura=2.5 * 60, max_vehi_pt_dt=10 * 60):
    """ function used to smooth the pt-level non-vehicle/vehicle prediction 
        Input:
        df_trip: data frame of the current processing trip
        non_vehi_seg_min_dura: minimum duration for non-vehicle segments, below which the seg will be changed to vehicle
        vehi_seg_min_dura: minimum duration for vehicle segments, below which the seg will be changed to non-vehicle
        max_vehi_pt_dt: maximum dt of a vehicle point
        Output:
        add one more column ['is_vehicle_smoothed']
    """
    is_vehicle = df_trip['is_vehicle'].values  # 0 for non-vehilce, 1 for vehicle
    dt_all = df_trip['TIME_DELTA'].values
    lat_all = df_trip['WLATITUDE'].values
    logging.debug("raw is_vehicle prediction: ")
    logging.debug(is_vehicle)
    logging.debug("delta t: ")
    logging.debug(dt_all)
    is_vehicle_smoothed = is_vehicle

    # make sure that large-dt points are non-vehicle
    is_vehicle_smoothed[dt_all > max_vehi_pt_dt] = 0

    # remove short non-vehicle segments between vehicle segments
    is_vehicle_chunks = chunks_real(is_vehicle_smoothed, include_values=True)
    num_chunks = len(is_vehicle_chunks)
    logging.debug("Number of original chunks: " + str(num_chunks))
    for idx, chunk in enumerate(is_vehicle_chunks):
        if idx != 0 and idx != num_chunks - 1 and chunk[2] == 0:
            chunk_dura = sum(dt_all[chunk[0]:chunk[1]])
            if chunk_dura < non_vehi_seg_min_dura:
                is_vehicle_smoothed[chunk[0]:chunk[1]] = [1] * (chunk[1] - chunk[0])
    logging.debug("is_vehicle after removing short non-vehicle segments:")
    logging.debug(is_vehicle_smoothed)
    # remove vehicle segments which are still short after combining
    is_vehicle_chunks = chunks_real(is_vehicle_smoothed, include_values=True)
    num_chunks = len(is_vehicle_chunks)
    logging.debug("Number of chunks after removing short non-vehicle segments: " + str(num_chunks))
    for chunk in is_vehicle_chunks:
        if chunk[2] == 1:
            chunk_dura = sum(dt_all[chunk[0]:chunk[1]])
            if chunk_dura < vehi_seg_min_dura:
                is_vehicle_smoothed[chunk[0]:chunk[1]] = [0] * (chunk[1] - chunk[0])
    logging.debug("is_vehicle after smoothing:")
    logging.debug(is_vehicle_smoothed)
    is_vehicle_chunks = chunks_real(is_vehicle_smoothed, include_values=True)
    num_chunks = len(is_vehicle_chunks)
    logging.debug("Number of chunks after smoothing: " + str(num_chunks))
    # make sure there's start/end location for the vehicle segment
    logging.debug(is_vehicle_smoothed)
    logging.debug(lat_all)
    logging.debug(dt_all)
    for chunk in is_vehicle_chunks:
        if chunk[2] == 1 and all(np.isnan(lat_all[chunk[0]:chunk[1]])):
            logging.debug("All points in this vehicle segment are unlocalized")
            # if the segment is vehicle, and all location is nan
            idx_valid_loc_before = np.where(~np.isnan(lat_all[0:chunk[0]]))[0]
            if len(idx_valid_loc_before) > 0:
                idx_valid_loc_front = idx_valid_loc_before[-1]  # the closest valid loc points before
                is_vehicle_smoothed[idx_valid_loc_front:chunk[0]] = [1] * (chunk[0] - idx_valid_loc_front)

            idx_valid_loc_after = np.where(~np.isnan(lat_all[chunk[1]:-1]))[0]
            if len(idx_valid_loc_after) > 0:
                idx_valid_loc_behind = idx_valid_loc_after[0]  # the closest valid loc points after
                is_vehicle_smoothed[chunk[1]:idx_valid_loc_behind] = [1] * (idx_valid_loc_behind - chunk[1])
    logging.debug(is_vehicle_smoothed)
    df_trip.loc[df_trip.index, 'is_vehicle_smoothed'] = is_vehicle_smoothed
    return None


def waiting_time_calc(starting_ts, df_all_day, backward_dura=20 * 60, forward_dura=5 * 60, waiting_range=50,
                      waiting_pts=5):
    """ function used to calculate the waiting time in the beginning of each vehicle segment
        Input:
        starting_ts: timestamp when the segment starts
        df_all_day: data frame of the entire day for chasing backwards
        backward_dura: the duration to look backwards from the starting point
        forward_dura: the duration to look forwards from the starting point
    """
    # find the points within the backwards and forwards time range, and with valid location
    df_possible_waiting = df_all_day.loc[(df_all_day['TIMESTAMP'] >= starting_ts - backward_dura) & (
        df_all_day['TIMESTAMP'] <= starting_ts + forward_dura)
                                         & ~np.isnan(df_all_day['WLATITUDE'])]
    if len(df_possible_waiting) == 0:
        return 0
    # apply DBSCAN clustering to the points
    core_samples_mask, cluster_labels = apply_DBSCAN(df_possible_waiting[['WLATITUDE', 'WLONGITUDE']].values.tolist(),
                                                     waiting_range, waiting_pts)
    logging.debug(cluster_labels)
    cluster_labels = np.array(cluster_labels)
    uniq_clusters = np.unique(cluster_labels)
    logging.debug(uniq_clusters)
    waiting_times = []
    for cluster in uniq_clusters:
        if cluster > -1:
            waiting_times.append(np.sum(df_possible_waiting.loc[cluster_labels == cluster, 'TIME_DELTA']))
    logging.debug(waiting_times)
    if len(waiting_times) > 0:
        return int(max(waiting_times))
    else:
        return 0


def vehicle_seg_feature_calc(df_vehi_seg, df_all_day, train_dist=150, bus_dist=100, default_dist=9999):
    """ function used to falculate features for the vehicle segment
    Input:
    df_vehi_seg: data frame of this vehicle segment
    df_all_day: data frame of the entire day
    train_dist: distance from train station in m
    bus_dist: distance from bus station in m
    default_dist: default distance from staions if no close staions are found
    Output:
    featureDict: distionary which contains all segment-level features
    """

    # initialize heuristics
    train_heuristic = TransitHeuristic.TrainPredictor()
    bus_heuristic = TransitHeuristic.BusMapPredictor()

    featureDict = {}
    # First velocity
    featureDict['first_vel'] = df_vehi_seg['MOV_AVE_VELOCITY'].iloc[0]
    # Duration
    featureDict['dur'] = df_vehi_seg['TIME_DELTA'].sum()
    # 85th percentile of velocity
    featureDict['85th'] = np.percentile(df_vehi_seg['MOV_AVE_VELOCITY'].as_matrix(), 85)
    # Distance
    featureDict['dist'] = df_vehi_seg['DISTANCE_DELTA'].sum()
    if featureDict['dist'] != 0:
        featureDict['mean_vel'] = featureDict['dist'] / featureDict['dur']
    else:
        # all unlocalized points contibutes to 0 distance
        featureDict['mean_vel'] = np.nanmean(df_vehi_seg['MOV_AVE_VELOCITY'].as_matrix())
        featureDict['dist'] = featureDict['mean_vel'] * featureDict['dur']

    # 95th percentil of acceleration
    acc_mat = df_vehi_seg['MOV_AVE_ACCELERATION'].as_matrix()
    featureDict['95th'] = np.percentile(acc_mat[~np.isnan(acc_mat)], 95)

    df_vehi_seg_validloc = df_vehi_seg.loc[~np.isnan(df_vehi_seg['WLATITUDE'])]
    if len(df_vehi_seg_validloc) > 0:
        first_valid_lat = df_vehi_seg_validloc['WLATITUDE'].iloc[0]
        first_valid_lon = df_vehi_seg_validloc['WLONGITUDE'].iloc[0]
        last_valid_lat = df_vehi_seg_validloc['WLATITUDE'].iloc[-1]
        last_valid_lon = df_vehi_seg_validloc['WLONGITUDE'].iloc[-1]
        # Distance between first point to nearest bus
        closeBusStops_begin = TransitHeuristic.find_nearest_station(first_valid_lat, first_valid_lon, \
                                                                    bus_heuristic.bus_location_tree, bus_dist)
        if len(closeBusStops_begin) > 0:
            dist2all = [busstop[1] for busstop in closeBusStops_begin]
            featureDict['first_bus'] = min(dist2all)
        else:  # just chooses the first tuple - TODO iterate for find the lowest distance
            featureDict['first_bus'] = default_dist  # np.NaN# closeBusStops_begin[0][1]
        # Distance between last point to nearest bus
        closeBusStops_end = TransitHeuristic.find_nearest_station(last_valid_lat, last_valid_lon, \
                                                                  bus_heuristic.bus_location_tree, bus_dist)
        if len(closeBusStops_end) > 0:
            dist2all = [busstop[1] for busstop in closeBusStops_end]
            featureDict['last_bus'] = min(dist2all)
        else:  # just chooses the first tuple - TODO iterate for find the lowest distance
            featureDict['last_bus'] = default_dist  # np.NaN#closeBusStops_end[0][1]
        # Distance between first point to nearest MRT
        closeTrainStation_begin = TransitHeuristic.find_nearest_station(first_valid_lat, first_valid_lon, \
                                                                        train_heuristic.train_location_tree, train_dist)
        if len(closeTrainStation_begin) > 0:
            dist2all = [mrtstation[1] for mrtstation in closeTrainStation_begin]
            featureDict['first_train'] = min(dist2all)
            # featureDict['first_train']  = closeTrainStation_begin[0][1]
        else:  # just chooses the first tuple - TODO iterate for find the lowest distance
            featureDict['first_train'] = default_dist  # np.NaN #closeTrainStation_begin[0][1]
        # Distance between last point to nearest MRT
        closeTrainStation_end = TransitHeuristic.find_nearest_station(last_valid_lat, last_valid_lon, \
                                                                      train_heuristic.train_location_tree, train_dist)
        if len(closeTrainStation_end) > 0:
            dist2all = [mrtstation[1] for mrtstation in closeTrainStation_end]
            featureDict['last_train'] = min(dist2all)
        else:  # just chooses the first tuple - TODO iterate for find the lowest distance
            featureDict['last_train'] = default_dist  # np.NaN# closeTrainStation_end[0][1]
    else:
        featureDict['first_bus'] = default_dist
        featureDict['last_bus'] = default_dist
        featureDict['first_train'] = default_dist
        featureDict['last_train'] = default_dist

    dist_to_bus = []
    dist_to_train = []
    lats = df_vehi_seg['WLATITUDE'].as_matrix()  # TODO remove iterate over all rows! Strange behaviour using iterrows()
    lons = df_vehi_seg['WLONGITUDE'].as_matrix()
    for i in range(0, len(lats)):
        if (not np.isnan(lats[i]) and not np.isnan(lons[i])):
            closeBusStops_iter = TransitHeuristic.find_nearest_station(lats[i], lons[i], \
                                                                       bus_heuristic.bus_location_tree, bus_dist * 10)

            closeTrainStops_iter = TransitHeuristic.find_nearest_station(lats[i], lons[i], \
                                                                         train_heuristic.train_location_tree,
                                                                         train_dist * 10)

            if len(closeBusStops_iter) > 0:
                dist2all = [busstop[1] for busstop in closeBusStops_iter]
                dist_to_bus.append(min(dist2all))

            if len(closeTrainStops_iter) > 0:
                dist2all = [trainstop[1] for trainstop in closeTrainStops_iter]
                dist_to_train.append(min(dist2all))

    if len(dist_to_bus) > 0:
        # Average distance to nearest bus stop
        featureDict['mean_dist_to_bus'] = np.mean(dist_to_bus)
        # Variance in distance to nearest bus stop
        featureDict['var_dist_to_bus'] = np.var(dist_to_bus)
    else:
        # Average distance to nearest bus stop
        featureDict['mean_dist_to_bus'] = default_dist
        # Variance in distance to nearest bus stop
        featureDict['var_dist_to_bus'] = default_dist

    if len(dist_to_train) > 0:
        # Average distance to nearest MRT
        featureDict['mean_dist_to_train'] = np.mean(dist_to_train)
        # Variance in distance to nearest MRT
        featureDict['var_dist_to_train'] = np.var(dist_to_train)
    else:
        # Average distance to nearest MRT
        featureDict['mean_dist_to_train'] = default_dist
        # Variance in distance to nearest MRT
        featureDict['var_dist_to_train'] = default_dist

    # STD of mean_mag over the entire segment
    featureDict['std_mean_mag'] = np.nanstd(df_vehi_seg['MEANMAG'].values)

    # Percentage of distance gaps (point clusters without location) over the segment (Number of NaN points)
    featureDict['nan_percentage'] = float(len(df_vehi_seg.loc[np.isnan(df_vehi_seg['WLATITUDE'])])) / len(df_vehi_seg)

    # Waiting time before the vehicle segment starts
    starting_ts = df_vehi_seg['TIMESTAMP'].values[0]
    featureDict['waiting_time'] = waiting_time_calc(starting_ts, df_all_day)

    return featureDict


def calculate_stairs(data_frame):
    """Calculate number of stairs climbed by looking at the pressure values. Derivatives of pressure changes are used
    to find pressure plateaus. Pressure changes between two consecutive plateaus are converted to an elevation change,
    which in turn is used to estimate the number of floors climbed. Number of stairs is calculated using an assumed number 
    of stairs per floor. 
    """
    # Smoothing for pressure
    nsmooth = 3
    # Time gap threshold for determining unique pressure plateaus
    climb_time_thresh = 2
    # Stable pressure point threshold for determining pressure plateau
    press_thresh = 3
    # Initialize stairs to 0
    stairs = 0

    dpdt = smooth((np.diff(smooth(data_frame['PRESSURE'], nsmooth)) / np.diff(data_frame['TIMESTAMP'])), nsmooth / 2)

    stab_idx_array = np.zeros(len(dpdt))
    dp_limit_high = 1
    dp_limit_low = -1

    stab_idx = np.where((dpdt > dp_limit_low) & (dpdt < dp_limit_high))
    new_stab_idx = stab_idx[0] + 1
    stab_idx_array[new_stab_idx] = 20

    # Find start and end indices of stable regions
    stable_press = np.concatenate(([0], np.equal(stab_idx_array, 20).view(np.int8), [0]))
    stable_start_end = find_array_start_end(stable_press, press_thresh)

    for item in stable_start_end:
        item[1] -= 1

    if len(stable_start_end) > 0:

        mean_press_array = []
        mean_temp_array = []
        mean_time_array = []

        for item in stable_start_end:
            mean_press_array.append(np.mean(data_frame['PRESSURE'][item[0:1]]))
            mean_temp_array.append(np.mean(data_frame['TEMPERATURE'][item[0:1]]) + 273.15)

        unstable_start_end = np.concatenate(
            ([0], stable_start_end.reshape(len(stable_start_end) * 2), [len(stab_idx_array) - 1])).reshape(-1, 2)

        for item in unstable_start_end[1:-1]:
            mean_time_array.append(data_frame['TIMESTAMP'][item[1]] - data_frame['TIMESTAMP'][item[0]])

        del_alt_array = np.zeros(len(mean_press_array) - 1)

        L = -0.0065
        g = 9.81
        M = 0.0289644
        Rs = 8.3144598
        R = Rs / M

        # Altitude is calculated from pressure and temperature values. Ref:
        # http://ubicomp.org/ubicomp2014/proceedings/ubicomp_adjunct/workshops/AwareCast/p459-liu.pdf

        for i in range(1, len(mean_press_array)):
            if (mean_press_array[i] < mean_press_array[i - 1]) and (
                            stable_start_end[i][0] - stable_start_end[i - 1][1] >= climb_time_thresh):
                del_alt_array[i - 1] = ((mean_temp_array[i] + mean_temp_array[i - 1]) / 2 / L) * (
                    ((mean_press_array[i] / mean_press_array[i - 1])) ** (-L * R / g) - 1)

        # We assume 0.17m as height of stair
        mPerStair = 0.17

        stairs_array = del_alt_array / mPerStair

        # Assume max rate of ascent as 1.2 stairs/s
        stairs_per_sec_thresh = 1.2

        final_stairs_array = []

        for i in range(0, len(stairs_array)):
            if len(mean_time_array) > 0:
                if stairs_array[i] / mean_time_array[i] <= stairs_per_sec_thresh:
                    final_stairs_array.append(stairs_array[i])

        stairs_upper_bound = 1000

        # All stairs are summed to give the total
        stairs = int(sum(final_stairs_array))
        if stairs < 0:
            stairs = 0
            logging.warning("NEGATIVE STAIRS")
        if stairs > stairs_upper_bound:
            stairs = stairs_upper_bound
            logging.warning("STAIRS EXCEEDED UPPER BOUND")

    # Original method of stair detection - just count number of steps in unstable regions

    # stairs = 0
    # unstable_start_end = np.concatenate(([0], stable_start_end.reshape(len(stable_start_end) * 2),
    # [len(stab_idx_array) - 1])).reshape(-1, 2)
    # for item in unstable_start_end:
    #   stairs += data_frame['STEPS'][item[1]] - data_frame['STEPS'][item[0]]

    return stairs


def calculate_aircon_time(data_frame):
    """Calculate the amount of time spent in an air conditioned environment
    by looking at humidity and temperature values.
    """

    start_ac_thresh = 3
    end_ac_thresh = 3
    start_ac_slope_thresh = -6
    end_ac_slope_thresh = 10
    temp_thresh = 29
    temp_num_thresh = 3

    troughs = (np.diff(np.sign(np.diff(data_frame['HUMIDITY']))) > 0).nonzero()[0] + 1
    peaks = (np.diff(np.sign(np.diff(data_frame['HUMIDITY']))) < 0).nonzero()[0] + 1
    near_trough = []
    near_peak = []

    for item in peaks:
        trough_idx = np.where(troughs > item)[0]
        if len(trough_idx) > 0:
            near_trough.append(troughs[trough_idx][0])

    for item in troughs:
        peak_idx = np.where(peaks > item)[0]
        if len(peak_idx) > 0:
            near_peak.append(peaks[peak_idx][0])

    start_slopes = np.zeros(len(data_frame['HUMIDITY']))

    for i in range(0, len(near_trough)):
        start = peaks[i]
        end = near_trough[i]
        start_slopes[start:end + 1] = [data_frame['HUMIDITY'][end] - data_frame['HUMIDITY'][start]] * (end - start + 1)

    end_slopes = np.zeros(len(data_frame['HUMIDITY']))

    for i in range(0, len(near_peak)):
        start = troughs[i]
        end = near_peak[i]
        end_slopes[start:end + 1] = [data_frame['HUMIDITY'][end] - data_frame['HUMIDITY'][start]] * (end - start + 1)

    low_temp_arr = np.concatenate(([0], (data_frame['TEMPERATURE'] <= temp_thresh).view(np.int8), [0]))
    temp_idx = find_array_start_end(low_temp_arr, temp_num_thresh)

    large_slope_arr_start = (start_slopes <= start_ac_slope_thresh).view(np.int8)
    start_idx = find_array_start_end(large_slope_arr_start, start_ac_thresh)

    large_slope_arr_end = (end_slopes >= end_ac_slope_thresh).view(np.int8)
    end_idx = find_array_start_end(large_slope_arr_end, end_ac_thresh)

    starts = []
    ends = []

    if len(start_idx) > 0:
        starts += list(start_idx[:, 0])
    if len(end_idx) > 0:
        ends += list(end_idx[:, 0])

    starts = np.sort(np.array(starts))
    ends = np.sort(np.array(ends))

    aircon_start_end = []

    for i in range(0, len(starts)):
        start_end_pair = []
        start_end_pair.append(starts[i])
        end = ends[np.where(ends > starts[i])]
        if len(end) and end[0] < len(data_frame):
            if not len(np.where(aircon_start_end == end[0])[0]):
                start_end_pair.append(int(end[0]))
                aircon_start_end.append(start_end_pair)
        else:
            start_end_pair.append(len(data_frame) - 1)
            aircon_start_end.append(start_end_pair)
            break

    all_aircon = []
    for item in aircon_start_end:
        all_aircon += range(item[0], item[1])
    for item in temp_idx:
        all_aircon += range(item[0], item[1])

    all_aircon = np.sort(all_aircon)

    aircon_start_end = []

    if len(all_aircon) > 0:
        j = 0
        start = all_aircon[j]
        end = start

        while j < len(all_aircon) - 1:
            if all_aircon[j + 1] - all_aircon[j] > 1:
                end = all_aircon[j]
            start_end_pair = []
            if start != end:
                start_end_pair.append(start)
                start_end_pair.append(end)
                aircon_start_end.append(start_end_pair)
                start = all_aircon[j + 1]
                end = start
            j += 1

        aircon_start_end.append([start, all_aircon[-1]])

    return aircon_start_end


def calculate_aircon_co2(data_frame):
    """Calculate carbon footprint caused by airconditioner usage. Time spent in an air conditioned environment
    is calculated by calculate_aircon_time(). The mean temperature during these time periods is used to determine the 
    amount of energy expended:Est. A/C power = nominal power * ( offset * e ^ ((30-room T)/scale)).
    Carbon footprint is calculated as: Cooling CO2 = ( time spent in A/C environment (h) * Est. A/C power (W) / 1000 ) *
     Air Con specific carbon intensity
    """

    aircon_intensity = 432
    offset_factor = 0.5
    scale_factor = 6
    nominal_power = 1500
    temp_correction = 0
    upper_bound = 17

    aircon_start_end = calculate_aircon_time(data_frame)
    mean_temp_array = []
    aircon_hours_array = []

    for item in aircon_start_end:
        mean_temp_array.append(np.mean(data_frame['TEMPERATURE'][item[0]:item[1]] - temp_correction))
        aircon_hours_array.append((data_frame['TIMESTAMP'][item[1]] - data_frame['TIMESTAMP'][item[0]]) / 3600.0)

    aircon_power_array = nominal_power * (offset_factor * np.exp((30 - np.array(mean_temp_array)) / scale_factor))
    aircon_energy_array = (aircon_hours_array * aircon_power_array / 1000)
    aircon_co2_array = aircon_energy_array * aircon_intensity

    total_aircon_energy = sum(aircon_energy_array)
    total_aircon_co2 = sum(aircon_co2_array)
    total_aircon_time = sum(aircon_hours_array)

    if total_aircon_energy > upper_bound:
        total_aircon_energy = upper_bound
        total_aircon_co2 = total_aircon_energy * aircon_intensity
        logging.warning("AIRCON EXCEEDED UPPER BOUND")

    return total_aircon_energy, total_aircon_co2, total_aircon_time


def smooth(y, box_pts):
    """Smoothing function for pressure"""
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def find_array_start_end(a, num_thresh):
    """Function to find start and end indices of sections of the array we are interested in."""
    absdiff = np.abs(np.diff(a))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    if len(ranges) > 0:
        ranges_mat = np.matrix(ranges)
        idx_picked = np.where(np.diff(ranges) >= num_thresh)[0]
        if len(idx_picked) > 0:
            ranges = ranges_mat[idx_picked, :]
        else:
            ranges = []
    return np.array(ranges)


def cal_win_label_special_trip_dict(labels, window_size, trip_dict, user_id=-1):
    """
    calculate the window labels for a list of labels
    :param labels: list of labels
    :param window_size: the window size
    :return: list of window labels
    """
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    result = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        if user_id != -1:
            if trip_dict[keys[key_idx + 1]][1] != user_id:
                continue
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]][0], trip_dict[keys[key_idx + 1]][0]):
            result.append(get_win_label(labels[idx - window_size + 1:idx + 1]))
    return result


def cal_win_label(labels, window_size, trip_dict):
    """
    calculate the window labels for a list of labels
    :param labels: list of labels
    :param window_size: the window size
    :return: list of window labels
    """
    if type(labels) is not np.ndarray:
        labels = np.array(labels)
    result = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]], trip_dict[keys[key_idx + 1]]):
            result.append(get_win_label(labels[idx - window_size + 1:idx + 1]))
    return result


def get_win_label(labels_in_win):
    """
    calculate the label for a whole window
    :param labels_in_win: all the point labels in the window
    :return: a label for the window
    """
    result = 5
    labels_in_win = labels_in_win.tolist()
    label_set = set(labels_in_win)
    if len(label_set) == 1:
        result = labels_in_win[0]
    return result


def cal_win_features_special_trip_dict(features, window_size, trip_dict, user_id=-1):
    """
    convert the point features to window features
    :param features: list of point features
    :param window_size: the window size
    :param trip_dict:
    :param user_id:
    :return: list of window features
    """

    if type(features) is not np.ndarray:
        features = np.array(features)
    results = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        if user_id != -1:
            if trip_dict[keys[key_idx + 1]][1] != user_id:
                continue

        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]][0], trip_dict[keys[key_idx + 1]][0]):
            results.append(get_win_feature(features[idx - window_size + 1:idx + 1]))
    return results


def cal_win_features(features, window_size, trip_dict):
    """
    convert the point features to window features
    :param features: list of point features
    :param window_size: the window size
    :return: list of window features
    """

    if type(features) is not np.ndarray:
        features = np.array(features)
    results = []
    keys = sorted(trip_dict.keys())
    for key_idx in range(len(keys) - 1):
        for idx in xrange(window_size - 1 + trip_dict[keys[key_idx]], trip_dict[keys[key_idx + 1]]):
            results.append(get_win_feature(features[idx - window_size + 1:idx + 1]))
    return results


def get_win_feature(features_list):
    """
    append all point features into one list
    :param features_list: a list of point features in a window
    :return: the features of a window in a list
    """
    results = []
    for features in features_list:
        for feature in features:
            results.append(feature)
    return results


def cal_busmrt_dist(df, train_dist=1500, bus_dist=1000, default_dist=-1):
    """
    cal_busmrt_dist function is to calculate The BUS_DIST and METRO_DIST.
    :param df: The DataFrame of data.
    :param train_dist: The threshold for METRO_DIST which cannot be larger than the threshold.
    :param bus_dist: The threshold for BUS_DIST which cannot be larger than the threshold.
    :param default_dist: If there is no valid BUS_DIST or METRO_DIST value, default_dist will be used.
    :return: None, the BUS_DIST and METRO_DIST will be added into the df.
    """
    train_heuristic = TransitHeuristic.TrainPredictor()
    bus_heuristic = TransitHeuristic.BusMapPredictor()

    bus_dist_list = []
    metro_dist_list = []

    TIME_THRESHOLD = 10 * 60
    std_velocity_all = []
    max_velocity_10min_all = []
    stops_10min_all = []
    fast_10min_all = []
    stops_near_bus_stop_10min_all = []

    df['BUS_DIST'] = pd.Series([default_dist]*len(df))
    df['METRO_DIST'] = pd.Series([default_dist]*len(df))
    for i in range(len(df)):
        closeBus = []
        closeMRT = []
        if ~np.isnan(df['WLATITUDE'].iloc[i]) and ~np.isnan(df['WLONGITUDE'].iloc[i]):
            closeBus = TransitHeuristic.find_nearest_station(df['WLATITUDE'].iloc[i], df['WLONGITUDE'].iloc[i], \
                                                             bus_heuristic.bus_location_tree, bus_dist)
            closeMRT = TransitHeuristic.find_nearest_station(df['WLATITUDE'].iloc[i], df['WLONGITUDE'].iloc[i], \
                                                             train_heuristic.train_location_tree, train_dist)
        if len(closeBus) > 0:
            dist2all = [Bus[1] for Bus in closeBus]
            df.set_value(i, 'BUS_DIST', min(dist2all))

        if len(closeMRT) > 0:
            dist2all = [MRT[1] for MRT in closeMRT]
            df.set_value(i, 'METRO_DIST', min(dist2all))

        recent_points, stop_count, stop_bus_stop_count, fast_count = pt_during_selected_time(df, i, TIME_THRESHOLD)
        stops_10min_all.append(stop_count)
        stops_near_bus_stop_10min_all.append(stop_bus_stop_count)
        fast_10min_all.append(fast_count)
        if (recent_points is None) or (len(recent_points) == 0):
            # there is no recent points
            std_velocity_all.append(0)
            max_velocity_10min_all.append(0)
        else:
            std = np.nanstd(recent_points['VELOCITY'])
            if np.isnan(std):
                std_velocity_all.append(0)
            else:
                std_velocity_all.append(std)

            m = max(recent_points['VELOCITY'])
            if np.isnan(m):
                max_velocity_10min_all.append(0)
            else:
                max_velocity_10min_all.append(m)

    df['MAX_VELOCITY_10MIN'] = pd.Series(max_velocity_10min_all)
    df['STD_VELOCITY_10MIN'] = pd.Series(std_velocity_all)
    df['STOP_10MIN'] = pd.Series(stops_10min_all)
    df['STOP_BUSSTOP_10MIN'] = pd.Series(stops_near_bus_stop_10min_all)
    df['FAST_10MIN'] = pd.Series(fast_10min_all)


def random_select_idx(labels_all_list, num_test, label_number):
    """
    random_select_idx function is to randomly select equal number of samples of each class from the whole data set
    :param labels_all_list: the list of labels of all the data
    :param num_test: the total number of selected samples
    :param label_number: the number of different class labels
    :return:
    """
    test_idx = []
    num_test_for_each_label = num_test / label_number
    if num_test_for_each_label > min(Counter(labels_all_list).values()):
        logging.warning("The num_test is too large!!!!!!!")
        return None
    for i in range(label_number):
        index_list_cur_label = []
        for cur in range(len(labels_all_list)):
            if labels_all_list[cur] == i:
                index_list_cur_label.append(cur)
        test_idx += random.sample(index_list_cur_label, num_test_for_each_label)
    test_idx = sorted(test_idx)
    return test_idx