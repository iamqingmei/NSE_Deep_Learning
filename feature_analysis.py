import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def std_velocity_10min(df, df_type):
    if not os.path.exists('./data/std_velocity_10min/' + df_type + '/'):
        os.makedirs('./data/std_velocity_10min/' + df_type + '/')
    label = 2
    cur_label_type = 'mrt'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['STD_VELOCITY_10MIN']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['STD_VELOCITY_10MIN'].to_csv('./data/std_velocity_10min/' + df_type + '/' + cur_label_type + '_STD_VELOCITY_10MIN.csv')
    label = 3
    cur_label_type = 'bus'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['STD_VELOCITY_10MIN']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['STD_VELOCITY_10MIN'].to_csv('./data/std_velocity_10min/' + df_type + '/' + cur_label_type + '_STD_VELOCITY_10MIN.csv')
    label = 4
    cur_label_type = 'car'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['STD_VELOCITY_10MIN']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['STD_VELOCITY_10MIN'].to_csv('./data/std_velocity_10min/' + df_type + '/' + cur_label_type + '_STD_VELOCITY_10MIN.csv')

    cur_label_type = 'non_vehicle'
    cur_label_df = df[(df.pt_label == 0) | (df.pt_label == 1) | (df.pt_label == 5)]
    cur_label_df = cur_label_df[['STD_VELOCITY_10MIN']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['STD_VELOCITY_10MIN'].to_csv('./data/std_velocity_10min/' + df_type + '/' + cur_label_type + '_STD_VELOCITY_10MIN.csv')


def save_features(features_name, df, df_type):
    folder_name = './data/' + features_name + '/' + df_type + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    label = 2
    cur_label_type = 'mrt'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[[features_name]]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df[features_name].to_csv(folder_name + cur_label_type + '_' + features_name + '.csv')
    label = 3
    cur_label_type = 'bus'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[[features_name]]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df[features_name].to_csv(folder_name + cur_label_type + '_' + features_name + '.csv')
    label = 4
    cur_label_type = 'car'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[[features_name]]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df[features_name].to_csv(folder_name + cur_label_type + '_' + features_name + '.csv')
    cur_label_type = 'non_vehicle'
    cur_label_df = df[(df.pt_label == 0) | (df.pt_label == 1) | (df.pt_label == 5)]
    cur_label_df = cur_label_df[[features_name]]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df[features_name].to_csv(folder_name + cur_label_type + '_' + features_name + '.csv')


def velocity(df, df_type):
    if not os.path.exists('./data/velocity/' + df_type + '/'):
        os.makedirs('./data/velocity/' + df_type + '/')
    label = 2
    cur_label_type = 'mrt'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['VELOCITY']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['VELOCITY'].to_csv('./data/velocity/' + df_type + '/' + cur_label_type + '_velocity.csv')
    label = 3
    cur_label_type = 'bus'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['VELOCITY']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['VELOCITY'].to_csv('./data/velocity/' + df_type + '/' + cur_label_type + '_velocity.csv')
    label = 4
    cur_label_type = 'car'
    cur_label_df = df[df.pt_label == label]
    cur_label_df = cur_label_df[['VELOCITY']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['VELOCITY'].to_csv('./data/velocity/' + df_type + '/' + cur_label_type + '_velocity.csv')
    cur_label_type = 'non_vehicle'
    cur_label_df = df[(df.pt_label == 0) | (df.pt_label == 1) | (df.pt_label == 5)]
    cur_label_df = cur_label_df[['VELOCITY']]
    cur_label_df.index = list(range(len(cur_label_df)))
    cur_label_df['VELOCITY'].to_csv('./data/velocity/' + df_type + '/' + cur_label_type + '_velocity.csv')

# options
num_bins = 20
get_all_feature_df_from_csv = True
ALL_FEATURES = ['STOP_10MIN', 'STOP_BUSSTOP_10MIN', 'FAST_10MIN']

if get_all_feature_df_from_csv:
    manual_features_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_features_df.csv')
    manual_labels_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_labels_df.csv')
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 0] = 5  # walk/stationary
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 3] = 4  # car
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 2] = 3  # bus
    manual_labels_pt.pt_label[manual_labels_pt.pt_label == 1] = 2  # mrt

    manual_df = manual_features_pt.copy()
    manual_df['pt_label'] = manual_labels_pt.pt_label

    app_features_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_labels_df.csv')
    app_df = app_features_pt.copy()
    app_df['pt_label'] = app_labels_pt.pt_label

    all_features_df = pd.concat([manual_features_pt, app_features_pt])

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

    # concatenate all dfs
    google_features = pd.DataFrame(pd.concat(features_df_list, ignore_index=True))
    google_labels = pd.DataFrame(pd.concat(labels_df_list, ignore_index=True))

    google_labels.pt_label[google_labels.pt_label == 0] = 5  # walk/stationary
    google_labels.pt_label[google_labels.pt_label == 3] = 4  # car
    google_labels.pt_label[google_labels.pt_label == 2] = 3  # bus
    google_labels.pt_label[google_labels.pt_label == 1] = 2  # mrt

    google_df = google_features.copy()
    google_df['pt_label'] = google_labels.pt_label
    all_features_df = pd.concat([all_features_df, google_features])

save_features('BUS_DIST', app_df, 'app')
save_features('BUS_DIST', google_df, 'google')
save_features('BUS_DIST', manual_df, 'manual')
# cur_feature = bus_features_df['STOP_10MIN'] + bus_features_df['FAST_10MIN']
# counts, bin_edges = np.histogram(cur_feature, bins=100, range=(min(cur_feature.tolist()), max(cur_feature.tolist())))
# # Plot
# counts = counts/len(bus_features_df)

# plt.plot(bin_edges[1:], counts)

# for i in list(all_features_df):
#     cur_feature = all_features_df[i]
#     print("current feature is " + str(i))
#     print("current feature max: " + str(max(cur_feature.tolist())))
#     print("current feature min: " + str(min(cur_feature.tolist())))
#     # Plotting the graph
#     # Use the histogram function to bin the data
#     counts, bin_edges = np.histogram(cur_feature, bins=num_bins,
#                                      range=(min(cur_feature.tolist()), max(cur_feature.tolist())))
#     # Now find the cdf
#     cdf = np.cumsum(counts)
#     cdf = cdf.tolist()
#     for a in range(len(cdf)):
#         cdf[a] = float(cdf[a]) / float(len(cur_feature))
#     # Plot
#     cdf.insert(0, 0)
#     plt.plot(bin_edges[:], cdf)






