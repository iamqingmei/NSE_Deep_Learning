import numpy as np
import math
import pandas as pd

min_dict = {'MOV_AVE_VELOCITY': 0, 'STDACC': -1, 'MEANMAG': -1, 'MAXGYR': -20000, 'PRESSURE': 100000, 'STDPRES_WIN': 0,
            'is_localized': 0, 'NUM_AP': 0, 'LIGHT': 0, 'NOISE': 27, 'STEPS': 96, 'TIME_DELTA': 0,
            'TEMPERATURE': 22.39, 'IRTEMPERATURE': 5, 'HUMIDITY': 20, 'STD_VELOCITY_10MIN': 0, 'MAX_VELOCITY_10MIN': 0,
            'VELOCITY': 0, 'STOP_10MIN': 0, 'STOP_BUSSTOP_10MIN': 0, 'FAST_10MIN': 0}

max_dict = {'MOV_AVE_VELOCITY': 35, 'STDACC': 4700, 'MEANMAG': 5000, 'MAXGYR': 20000, 'PRESSURE': 102000,
            'STDPRES_WIN': 100, 'is_localized': 1, 'NUM_AP': 20, 'LIGHT': 2000, 'NOISE': 90, 'STEPS': 106500,
            'TIME_DELTA': 300, 'TEMPERATURE': 38, 'IRTEMPERATURE': 35, 'HUMIDITY': 80, 'STD_VELOCITY_10MIN': 8.5,
            'MAX_VELOCITY_10MIN': 39, 'VELOCITY': 35, 'STOP_10MIN': 37, 'STOP_BUSSTOP_10MIN': 36, 'FAST_10MIN': 36}

var_dict = {'METRO_DIST': 315000, 'BUS_DIST': 16600}


def normalize(features_df):
    """
    normalize the input features with min-max scaler, [0,1],
    except the bus_dist and metro_dist
    for valid bus_dist and metro_dist, we use Gaussian function to normalize it,
    invalid bus_dist and metro_dist will be put 0
    :param features_df: input feature data frames
    :return: the normalized features in np.array
    """

    scaled_features = pd.DataFrame(columns=list(features_df))
    if 'METRO_DIST' in list(features_df):
        mrt_dist_list = features_df['METRO_DIST'].tolist()

        valid_mrt_bool = list(i != -1 for i in mrt_dist_list)
        valid_mrt_dist = list(filter(lambda a: a != -1.0, mrt_dist_list))
        scaled_mrt = [0] * len(mrt_dist_list)
        if len(valid_mrt_dist) > 0:
            scaled_valid_mrt = gaussian_fun(valid_mrt_dist, 'METRO_DIST')
            # list(map(lambda x: math.exp(-x*x/(2*529.0213))*(x!=-1), mrt_dist_list))
            count = 0
            for i in range(len(scaled_mrt)):
                if valid_mrt_bool[i] is True:
                    scaled_mrt[i] = scaled_valid_mrt[count]
                    count += 1

        scaled_features['METRO_DIST'] = pd.Series(scaled_mrt)

    if 'BUS_DIST' in list(features_df):
        bus_dist_list = features_df['BUS_DIST'].tolist()

        valid_bus_bool = list(i != -1 for i in bus_dist_list)
        valid_bus_dist = list(filter(lambda a: a != -1.0, bus_dist_list))
        scaled_bus = [0] * len(bus_dist_list)
        if len(valid_bus_dist) > 0:
            scaled_valid_bus = gaussian_fun(valid_bus_dist, 'BUS_DIST')
            count = 0
            for i in range(len(scaled_bus)):
                if valid_bus_bool[i] is True:
                    scaled_bus[i] = scaled_valid_bus[count]
                    count += 1

        scaled_features['BUS_DIST'] = pd.Series(scaled_bus)

    # invalid nan lat or lon will put 0
    if 'WLATITUDE' in list(features_df):
        min_lat_sg = 1.235578
        max_lat_sg = 1.479055
        lat_list = features_df['WLATITUDE']
        lat_list = list(map(lambda x: (x - min_lat_sg) / (max_lat_sg - min_lat_sg) * (~np.isnan(x)), lat_list))

        #  remove nan in lat_list
        for i in range(len(lat_list)):
            if np.isnan(lat_list[i]):
                lat_list[i] = 0

        scaled_features['WLATITUDE'] = lat_list

    if 'WLONGITUDE' in list(features_df):
        min_lon_sg = 103.565276
        max_lon_sg = 104
        lon_list = features_df['WLONGITUDE']
        lon_list = list(map(lambda x: (x - min_lon_sg) / (max_lon_sg - min_lon_sg) * (~np.isnan(x)), lon_list))

        #  remove nan in lon_list
        for i in range(len(lon_list)):
            if np.isnan(lon_list[i]):
                lon_list[i] = 0

        scaled_features['WLONGITUDE'] = lon_list

    for col in list(features_df):
        if col in ['BUS_DIST', 'METRO_DIST', 'WLONGITUDE', 'WLATITUDE']:
            continue
        # print "processing column " + str(col)
        # print "min of this column " + str(min_dict[col])
        # print "max of this column " + str(max_dict[col])
        tmp = min_max_normalize(features_df[col].tolist(), min_dict[col], max_dict[col])
        scaled_features[col] = pd.Series(tmp)

    return scaled_features


def cal(num, maxi, var):
    return math.exp(-(math.pow(num - maxi, 2)) / (2 * var))


def gaussian_fun(valid_dist_list, feature):
    var = var_dict[feature]
    return list(cal(i, 0, var) for i in valid_dist_list)


def min_max_normalize(before_normalize, min_value, max_value):
    # invalid value will be nan
    result = []
    for x in before_normalize:
        if np.isnan(x):
            result.append(0.0)
        elif x >= max_value:
            result.append(1.0)
        elif x <= min_value:
            result.append(0.0)
        else:
            result.append(float(x - min_value) / float(max_value - min_value))
    return result
