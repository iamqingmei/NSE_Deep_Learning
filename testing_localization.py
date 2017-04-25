import numpy as np
import pandas as pd
import os

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

google_pt_df = google_features.copy()
google_pt_df['pt_label'] = google_labels['pt_label']

manual_features_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_features_df.csv')
manual_labels_pt = pd.DataFrame.from_csv('./data/manual/unnormalized_pt_labels_df.csv')

manual_labels_pt.pt_label[manual_labels_pt.pt_label == 0] = 5  # walk/stationary
manual_labels_pt.pt_label[manual_labels_pt.pt_label == 3] = 4  # car
manual_labels_pt.pt_label[manual_labels_pt.pt_label == 2] = 3  # bus
manual_labels_pt.pt_label[manual_labels_pt.pt_label == 1] = 2  # mrt

manual_pt_df = manual_features_pt.copy()
manual_pt_df['pt_label'] = manual_labels_pt['pt_label']

manual_pt_df = manual_pt_df[manual_pt_df['pt_label'] != -1]
google_pt_df = google_pt_df[google_pt_df['pt_label'] != -1]
app_features_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_features_df.csv')
app_labels_pt = pd.DataFrame.from_csv('./data/app/unnormalized_pt_labels_df.csv')

app_labels_pt.pt_label[app_labels_pt.pt_label == 0] = 5  # walk/stationary
app_labels_pt.pt_label[app_labels_pt.pt_label == 1] = 5  # walk/stationary

app_pt_df = app_features_pt.copy()
app_pt_df['pt_label'] = app_labels_pt


for i in [2,3,4]:
    print(i)
    for data in [app_pt_df, manual_pt_df, google_pt_df]:
        i_dataset = data[data['pt_label'] == i]
        print(len(i_dataset))
        print(len(data))
        print(len(i_dataset[~np.isnan(i_dataset['WLONGITUDE'])]) / len(i_dataset))

for data in [app_pt_df, manual_pt_df, google_pt_df]:
    # data = data[data['pt_label'] == 5]
    # print(len(data))
    print(len(data[~np.isnan(data['WLONGITUDE'])]) / len(data))
print("hi")
