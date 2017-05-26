import numpy as np
import params
import pickle
import os


class Trip:
    tripCount = 0

    def __init__(self, df, trip_id, nid):
        self.trip_id = trip_id
        self.trip_df = df
        self.trip_length = len(df)
        self.nid = nid
        Trip.tripCount += 1

    def get_win_info(self, features):
        features_df = self.trip_df[features]
        last_based_labels = []
        all_based_win_labels = []
        all_features = []
        all_time_delta = []
        all_lon = []
        all_lat = []
        min_time = 0.0
        max_time = 300.0
        for idx in range(params.window_size-1, self.trip_length):
            all_based_win_labels.append(get_all_based_win_label(self.trip_df.iloc[idx - params.window_size + 1:idx + 1]
                                                                ['pt_label']))
            last_based_labels.append(self.trip_df.iloc[idx]['pt_label'])

            lon = self.trip_df.iloc[idx]['WLONGITUDE']
            lat = self.trip_df.iloc[idx]['WLATITUDE']
            time_delta = self.trip_df.iloc[idx]['TIME_DELTA']

            all_lon.append(lon)
            all_lat.append(lat)
            all_time_delta.append(time_delta)

            all_features.append(list(np.array(features_df.iloc[idx - params.window_size + 1:idx + 1]).reshape(-1)))

        all_time_delta = list(map(lambda x: (int(time_delta * (max_time - min_time) + min_time)), all_time_delta))

        return all_features, last_based_labels, all_based_win_labels, \
               [self.trip_id] * (self.trip_length - params.window_size + 1), \
               [self.nid] * (self.trip_length - params.window_size + 1), all_time_delta, all_lon, all_lat

    def get_win_info_with_localization_rate(self, features):
        # 4/6 for bus classes
        # 4/6 for car classes
        features_df = self.trip_df[features]
        last_based_labels = []
        all_based_win_labels = []
        all_features = []
        all_time_delta = []
        all_lon = []
        all_lat = []
        min_time = 0.0
        max_time = 300.0
        for idx in range(params.window_size-1, self.trip_length):
            cur_all_based_win_label = get_all_based_win_label(self.trip_df.iloc[idx - params.window_size + 1:idx + 1]
                                                              ['pt_label'])
            cur_last_based_win_label = self.trip_df.iloc[idx]['pt_label']
            cur_win_localization_count = np.count_nonzero(~np.isnan(self.trip_df
                                                                    .iloc[idx - params.window_size + 1:idx + 1]
                                                                    ['WLONGITUDE']))
            if (cur_all_based_win_label == 3) or (cur_last_based_win_label == 3):
                if cur_win_localization_count < params.min_bus_localization_count:
                    continue
            elif (cur_all_based_win_label == 4) or (cur_last_based_win_label == 4):
                if cur_win_localization_count < params.min_car_localization_count:
                    continue
            elif (cur_all_based_win_label == 0) or (cur_last_based_win_label == 0):
                if cur_win_localization_count < params.min_no_veh_localization_count:
                    continue
            elif (cur_all_based_win_label == 1) or (cur_last_based_win_label == 1):
                if cur_win_localization_count < params.min_no_veh_localization_count:
                    continue
            elif (cur_all_based_win_label == 5) or (cur_last_based_win_label == 5):
                if cur_win_localization_count < params.min_no_veh_localization_count:
                    continue
            elif (cur_all_based_win_label == 2) or (cur_last_based_win_label == 2):
                if cur_win_localization_count < params.min_mrt_localization_count:
                    continue

            all_based_win_labels.append(cur_all_based_win_label)
            last_based_labels.append(cur_last_based_win_label)

            lon = self.trip_df.iloc[idx]['WLONGITUDE']
            lat = self.trip_df.iloc[idx]['WLATITUDE']
            time_delta = self.trip_df.iloc[idx]['TIME_DELTA']

            all_lon.append(lon)
            all_lat.append(lat)
            all_time_delta.append(time_delta)

            all_features.append(list(np.array(features_df.iloc[idx - params.window_size + 1:idx + 1]).reshape(-1)))

        all_time_delta = list(map(lambda x: (int(time_delta * (max_time - min_time) + min_time)), all_time_delta))
        return all_features, last_based_labels, all_based_win_labels, \
               [self.trip_id] * len(last_based_labels), \
               [self.nid] * len(last_based_labels), all_time_delta, all_lon, all_lat

    def get_pt_df(self, features):
        return self.trip_df[features]

    def save_trip_object(self):
        folder_name = './data/trip_object/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(folder_name + 'trip_' + str(self.trip_id) + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


def get_all_based_win_label(labels_in_win):
    result = params.defalt_mixed_invalid_label
    label_set = set(labels_in_win)
    if len(label_set) == 1:
        result = labels_in_win.iloc[0]
    return result
