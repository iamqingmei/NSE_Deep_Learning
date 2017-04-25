import pandas as pd
import numpy as np
import pickle
import params
from Trip_Class import Trip
from normalization import normalize
import os
import logging
from collections import Counter


def get_app_trip_df_list(all_features):
    app_features_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_labels_df.csv')['pt_label'].tolist()
    with open("./data/app/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    app_pt_df = normalize(app_features_pt[all_features])
    app_pt_df['pt_label'] = app_labels_pt
    app_pt_df['WLONGITUDE'] = app_features_pt['WLONGITUDE']
    app_pt_df['WLATITUDE'] = app_features_pt['WLATITUDE']
    # normalized the point features

    trip_keys = list(trip_dict.keys())
    trip_keys.remove(-1)
    trips_df_list = []
    for cur_trip_key in trip_keys:
        trips_df_list.append(Trip(
            app_pt_df.iloc[trip_dict[cur_trip_key - 1]:trip_dict[cur_trip_key]],
            cur_trip_key,
            app_features_pt.iloc[trip_dict[cur_trip_key - 1]]['NID']))

    del app_features_pt, app_labels_pt, app_pt_df, trip_keys

    return trips_df_list


def get_manual_trip_df_list(all_features):
    manual_features_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_features_df.csv')
    manual_labels_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_labels_df.csv')

    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 0] = 5  # walk/stationary
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 3] = 4  # car
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 2] = 3  # bus
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 1] = 2  # mrt

    with open("./data/manual/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    manual_pt_df = normalize(manual_features_pt[all_features])
    manual_pt_df['pt_label'] = manual_labels_pt['pt_label']
    manual_pt_df['WLONGITUDE'] = manual_features_pt['WLONGITUDE']
    manual_pt_df['WLATITUDE'] = manual_features_pt['WLATITUDE']
    # normalized the point features

    trip_keys = list(trip_dict.keys())
    trip_keys.remove(-1)
    trips_df_list = []
    for cur_trip_key in trip_keys:
        trips_df_list.append(Trip(
            manual_pt_df.iloc[trip_dict[cur_trip_key - 1][0]:trip_dict[cur_trip_key][0]],
            cur_trip_key + 1000,
            manual_features_pt.iloc[trip_dict[cur_trip_key - 1][0]]['NID']))

    del manual_features_pt, manual_labels_pt, manual_pt_df, trip_keys

    return trips_df_list


def get_google_trip_df_list(all_features):
    file_loc = "./data/google/"
    all_files = os.listdir(file_loc)
    features_df_list = []
    labels_df_list = []
    max_num = 0
    for file in all_files:
        if 'unnormalized_pt_features_df' in file:
            n = int(file[file.index('df_') + 3:file.index('.csv')])
            if n > max_num:
                max_num = n

    for i in range(max_num + 1):
        # go through each file and read out df
        features_tmp = pd.DataFrame.from_csv(file_loc + 'unnormalized_pt_features_df_' + str(i) + '.csv')
        labels_tmp = pd.DataFrame.from_csv(file_loc + 'unnormalized_pt_labels_df_' + str(i) + '.csv')

        features_df_list.append(features_tmp)
        labels_df_list.append(labels_tmp)

    with open("./data/google/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    # concatenate all dfs
    google_features = pd.DataFrame(pd.concat(features_df_list, ignore_index=True))
    google_labels = pd.DataFrame(pd.concat(labels_df_list, ignore_index=True))

    google_labels.pt_label[google_labels.pt_label == 0] = 5  # walk/stationary
    google_labels.pt_label[google_labels.pt_label == 3] = 4  # car
    google_labels.pt_label[google_labels.pt_label == 2] = 3  # bus
    google_labels.pt_label[google_labels.pt_label == 1] = 2  # mrt

    # normalize
    google_pt_df = normalize(google_features[all_features])
    google_pt_df['pt_label'] = google_labels['pt_label']
    google_pt_df['WLONGITUDE'] = google_features['WLONGITUDE']
    google_pt_df['WLATITUDE'] = google_features['WLATITUDE']

    trip_keys = list(trip_dict.keys())
    trip_keys.remove(-1)
    trips_list = []
    for cur_trip_key in trip_keys:
        trips_list.append(Trip(
            google_pt_df.iloc[trip_dict[cur_trip_key - 1]:trip_dict[cur_trip_key]],
            cur_trip_key + 2000,
            google_features.iloc[trip_dict[cur_trip_key - 1]]['NID']))

    del file_loc, all_files, features_df_list, labels_df_list, max_num, google_features, google_labels, google_pt_df, \
        trip_keys

    return trips_list


def random_test_train(select_type, df, ratio):
    if select_type == 'trip':
        trip_keys = list(set(df['trip_id'].tolist()))

        test_trip_idx = np.random.choice(trip_keys, int(len(trip_keys) * ratio),
                                         replace=False)
        test_trip_idx = sorted(test_trip_idx)
        train_trip_idx = [x for x in trip_keys if x not in test_trip_idx]
        test_df = df[(df['trip_id'].isin(test_trip_idx))]
        train_df = df[(df['trip_id'].isin(train_trip_idx))]

        del trip_keys, test_trip_idx, train_trip_idx
        return train_df, test_df
    elif select_type == 'window':
        test_idx = np.random.choice(len(df), int(len(df) * ratio), replace=False)
        train_idx = [x for x in range(len(df)) if x not in test_idx]
        test_df = df.iloc[test_idx, :]
        train_df = df.iloc[train_idx, :]

        del test_idx, train_idx
    elif select_type == 'nid':
        nid_list = list(set(df['nid'].tolist()))
        test_nid_idx = np.random.choice(nid_list, int(len(nid_list) * ratio), replace=False)
        train_nid_idx = [x for x in nid_list if x not in test_nid_idx]
        test_df = df[(df['nid'].isin(test_nid_idx))]
        train_df = df[(df['nid'].isin(train_nid_idx))]

        del nid_list, test_nid_idx, train_nid_idx
    else:
        print("Wrong random selection type! Please enter 'trip', 'nid' or 'window'")
        test_df = pd.DataFrame()
        train_df = pd.DataFrame()
        quit()
    return train_df, test_df


def get_feature_idx(features, all_features):
    result = []
    for feature in features:
        idx = all_features.index(feature)
        for i in range(params.window_size):
            result.append(idx)
            idx += len(all_features)

    result = sorted(result)
    return result


def reassign_label(train_df, test_df, reassign_map):
    # map [[old,new],[old,new],[old,new]]
    for i in reassign_map:
        train_df.ix[train_df['last_based_win_label'] == i[0], 'last_based_win_label'] = i[1]
        test_df.ix[test_df['last_based_win_label'] == i[0], 'last_based_win_label'] = i[1]
        train_df.ix[train_df['all_based_win_label'] == i[0], 'all_based_win_label'] = i[1]
        test_df.ix[test_df['all_based_win_label'] == i[0], 'all_based_win_label'] = i[1]


def get_win_df(pt_trip_list, all_features):
    win_features = []
    all_based = []
    last_based = []
    trip_id = []
    nid_list = []
    time_delta_list = []
    lon_list = []
    lat_list = []
    for trip in pt_trip_list:
        logging.info("Creating win_df for trip: %d" % trip.trip_id)
        # return all_features, last_based_labels, all_based_win_labels, \
        #        [self.trip_id] * (self.trip_length - params.window_size + 1), \
        #        [self.nid] * (self.trip_length - params.window_size + 1), all_time_delta, all_lon, all_lat
        # f, l, a, t, nid, time_delta, lon, lat = trip.get_win_info_with_localization_rate(all_features)
        f, l, a, t, nid, time_delta, lon, lat = trip.get_win_info_with_localization_rate(all_features)
        win_features += f
        all_based += a
        last_based += l
        trip_id += t
        nid_list += nid
        time_delta_list += time_delta
        lon_list += lon
        lat_list += lat

    resulted_win_df = pd.DataFrame(win_features, index=list(range(len(win_features))))
    resulted_win_df['last_based_win_label'] = pd.Series(last_based, index=resulted_win_df.index)
    resulted_win_df['all_based_win_label'] = pd.Series(all_based, index=resulted_win_df.index)
    resulted_win_df['trip_id'] = pd.Series(trip_id, index=resulted_win_df.index)
    resulted_win_df['nid'] = pd.Series(nid_list, index=resulted_win_df.index)
    resulted_win_df['TIME_DELTA'] = pd.Series(time_delta_list, index=resulted_win_df.index)
    resulted_win_df['WLATITUDE'] = pd.Series(lat_list, index=resulted_win_df.index)
    resulted_win_df['WLONGITUDE'] = pd.Series(lon_list, index=resulted_win_df.index)

    del win_features, all_based, last_based, trip_id, nid_list, time_delta_list, lon_list, lat_list

    return resulted_win_df


def get_win_df_from_csv():
    app_win_df = pd.DataFrame.from_csv('./data/app/app_win_df.csv')
    manual_win_df = pd.DataFrame.from_csv('./data/manual/manual_win_df.csv')
    google_win_df = pd.DataFrame.from_csv('./data/google/google_win_df.csv')
    return app_win_df, manual_win_df, google_win_df


def remove_mix(df, label_type):
    return df[df[label_type] != params.defalt_mixed_invalid_label]


def balance_dataset(features_df, labels_df):
    win_df = features_df
    win_df['win_label'] = pd.Series(labels_df, index=win_df.index)

    labels_count_dict = Counter(labels_df)
    print(labels_count_dict)

    for cur_label in labels_count_dict.keys():
        if cur_label == max(labels_count_dict.values()):
            continue
        cur_label_count = labels_count_dict[cur_label]
        # print "cur_label is: " + str(cur_label)
        # print "cur_label_count is: " + str(cur_label_count)

        copy_times = int(round(float(max(labels_count_dict.values()))/cur_label_count - 1, 0))
        # print "copy_times is :" + str(copy_times)

        if copy_times > 0:
            cur_label_df = win_df[(win_df['win_label'] == cur_label)]
            # print "cur_label_df : " + str(len(cur_label_df))
            win_df = win_df.append(pd.concat([cur_label_df]*copy_times, ignore_index=True))
            # print "added " + str(labels_count_dict[cur_label] * copy_times) + " label " + str(cur_label)
            # print "now the size of win_df is: " + str(len(win_df))
        # else:
        #     print "no need to copy"

    print("after balance: ")
    labels_count_dict = Counter(win_df['win_label'])
    print(labels_count_dict)

    return win_df.iloc[:, :-1], win_df['win_label']
