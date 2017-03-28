import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# options
num_bins = 20
get_all_feature_df_from_csv = True
ALL_FEATURES = ['STOP_10MIN', 'STOP_BUSSTOP_10MIN', 'FAST_10MIN']

if get_all_feature_df_from_csv:
    manual_features_pt = pd.DataFrame.from_csv('./manual/pt_df/unnormalized_pt_features_df.csv')
    manual_labels_pt = pd.DataFrame.from_csv('./manual/pt_df/unnormalized_pt_labels_df.csv')
    app_features_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_features_df.csv')
    app_labels_pt = pd.DataFrame.from_csv('./pt_df/unnormalized_pt_labels_df.csv')
    all_features_df = pd.concat([manual_features_pt, app_features_pt])
    bus_manual_features_df = manual_features_pt[manual_labels_pt.pt_label == 3]
    bus_app_features_df = app_features_pt[app_labels_pt.pt_label == 4]
    bus_features_df = pd.concat([bus_manual_features_df, bus_app_features_df])

cur_feature = bus_features_df['STOP_10MIN'] + bus_features_df['FAST_10MIN']
counts, bin_edges = np.histogram(cur_feature, bins=100, range=(min(cur_feature.tolist()), max(cur_feature.tolist())))
# Plot
counts = counts/len(bus_features_df)

plt.plot(bin_edges[1:], counts)

for i in range(len(ALL_FEATURES)):
    cur_feature = all_features_df[ALL_FEATURES[i]]
    print ("current feature is " + str(ALL_FEATURES[i]))
    print ("current feature max: " + str(max(cur_feature.tolist())))
    print ("current feature min: " + str(min(cur_feature.tolist())))
    # Plotting the graph
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(cur_feature, bins=num_bins, range=(min(cur_feature.tolist()), max(cur_feature.tolist())))
    # Now find the cdf
    cdf = np.cumsum(counts)
    cdf = cdf.tolist()
    for a in range(len(cdf)):
        cdf[a] = float(cdf[a])/float(len(cur_feature))
    # Plot
    cdf.insert(0, 0)
    plt.plot(bin_edges[:], cdf)


# current feature is MOV_AVE_VELOCITY
# current feature max: 33.7045651346
# current feature min: 0.0
#
# current feature is STDACC
# current feature max: 4634.0
# current feature min: -1.0

# current feature is MEANMAG
# current feature max: 30919.0
# current feature min: -1.0
# current feature is MAXGYR
# current feature max: 32767.0
# current feature min: -32613.0
# current feature is PRESSURE
# current feature max: 101884.0
# current feature min: 99488.0
# current feature is STDPRES_WIN
# current feature max: 569.30064309
# current feature min: 0.0
# current feature is WLATITUDE
# current feature max: 1.456986
# current feature min: 1.265377
# current feature is WLONGITUDE
# current feature max: 103.990426
# current feature min: 103.669782
# current feature is is_localized
# current feature max: 1.0
# current feature min: 0.0
# current feature is METRO_DIST
# current feature max: 1499.92328752
# current feature min: -1.0
# current feature is BUS_DIST
# current feature max: 966.582672968
# current feature min: -1.0
# current feature is STEPS
# current feature max: 106574
# current feature min: 96
# current feature is NOISE
# current feature max: 96
# current feature min: 27
# current feature is TIME_DELTA
# current feature max: 5141
# current feature min: 0
# current feature is TEMPERATURE
# current feature max: 58.5
# current feature min: 22.39
# current feature is IRTEMPERATURE
# current feature max: 62.12
# current feature min: 2.63
# current feature is HUMIDITY
# current feature max: 100.0
# current feature min: 6.5
# current feature is STD_VELOCITY_10MIN
# current feature max: 16.13141206
# current feature min: 0.0
# current feature is MAX_VELOCITY_10MIN
# current feature max: 39.70455285
# current feature min: 0.0
# current feature is VELOCITY
# current feature max: 39.70455285
# current feature min: 0.0
# current feature is TIMESTAMP
# current feature max: 1470820430
# current feature min: 1456699307
# current feature is STOP_10MIN
# current feature max: 37.0
# current feature min: 0.0
# current feature is STOP_BUSSTOP_10MIN
# current feature max: 36.0
# current feature min: 0.0
# current feature is FAST_10MIN
# current feature max: 36.0
# current feature min: 0.0