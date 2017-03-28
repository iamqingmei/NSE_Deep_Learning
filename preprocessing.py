import pandas as pd
import numpy as np
import pickle
import params
from Trip_Class import Trip
from normalization import normalize


def get_app_trip_df_list(all_features):
    app_features_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_labels_df.csv')['pt_label'].tolist()
    with open("./data/app/trip_dict.txt", "rb") as fp:  # Unpickling
        trip_dict = pickle.load(fp)

    app_pt_df = normalize(app_features_pt[all_features])
    app_pt_df['pt_label'] = app_labels_pt
    # normalized the point features

    trip_keys = list(trip_dict.keys())
    trip_keys.remove(-1)
    trips_df_list = []
    for cur_trip_key in trip_keys:
        trips_df_list.append(Trip(
            app_pt_df.iloc[trip_dict[cur_trip_key - 1]:trip_dict[cur_trip_key]],
            cur_trip_key))

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
    # normalized the point features

    trip_keys = list(trip_dict.keys())
    trip_keys.remove(-1)
    trips_df_list = []
    for cur_trip_key in trip_keys:
        trips_df_list.append(Trip(
            manual_pt_df.iloc[trip_dict[cur_trip_key - 1][0]:trip_dict[cur_trip_key][0]],
            cur_trip_key + 1000))

    return trips_df_list


def random_test_train(select_type, df, ratio, label_type, remove_mixed=True):
    if remove_mixed:
        df = df[df[label_type] != params.defalt_mixed_invalid_label]
    if select_type == 'trip':
        trip_keys = list(set(df['trip_id'].tolist()))

        test_trip_idx = np.random.choice(trip_keys, int(len(trip_keys) * ratio),
                                         replace=False)
        test_trip_idx = sorted(test_trip_idx)
        train_trip_idx = [x for x in trip_keys if x not in test_trip_idx]
        test_df = df[(df['trip_id'].isin(test_trip_idx))]
        train_df = df[(df['trip_id'].isin(train_trip_idx))]
        return train_df, test_df
    elif select_type == 'window':
        test_idx = np.random.choice(len(df), int(len(df) * ratio), replace=False)
        train_idx = [x for x in range(len(df)) if x not in test_idx]
        test_df = df.iloc[test_idx, :]
        train_df = df.iloc[train_idx, :]
    else:
        print("Wrong random selection type! Please enter 'trip' or 'window'")
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


def reassign_label(train_df, test_df, reassign_map, label_type):
    # map [[old,new],[old,new],[old,new]]
    for i in reassign_map:
        train_df.ix[train_df[label_type] == i[0], label_type] = i[1]
        test_df.ix[test_df[label_type] == i[0], label_type] = i[1]


def get_win_df(pt_trip_list, all_features):
    win_features = []
    all_based = []
    last_based = []
    trip_id = []
    for trip in pt_trip_list:
        f, a, l, t = trip.get_win_info(all_features)
        win_features += f
        all_based += a.tolist()
        last_based += l
        trip_id += t
    resulted_win_df = pd.DataFrame(win_features)
    resulted_win_df['last_based_win_label'] = pd.Series(last_based)
    resulted_win_df['all_based_win_label'] = pd.Series(all_based)
    resulted_win_df['trip_id'] = pd.Series(trip_id)

    return resulted_win_df


def get_win_df_from_csv():
    app_win_df = pd.DataFrame.from_csv('./data/app/app_win_df.csv')
    manual_win_df = pd.DataFrame.from_csv('./data/manual/manual_win_df.csv')
    return app_win_df, manual_win_df
