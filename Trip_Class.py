import pandas as pd
import numpy as np
import params
import pickle
import os


class Trip:
    tripCount = 0

    def __init__(self, df, trip_id):
        self.trip_id = trip_id
        self.trip_df = df
        self.trip_length = len(df)
        Trip.tripCount += 1

    def get_win_info(self, features):
        features_df = self.trip_df[features]
        last_based_labels = self.trip_df.iloc[params.window_size-1:]['pt_label']
        all_based_win_labels = []
        all_features = []
        for idx in range(params.window_size-1, self.trip_length):
            all_based_win_labels.append(get_all_based_win_label(self.trip_df.iloc[idx - params.window_size + 1:idx + 1]
                                                                ['pt_label']))
            all_features.append(list(np.array(features_df.iloc[idx - params.window_size + 1:idx + 1]).reshape(-1)))
        return all_features, last_based_labels, all_based_win_labels, \
               [self.trip_id] * (self.trip_length - params.window_size + 1)

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



