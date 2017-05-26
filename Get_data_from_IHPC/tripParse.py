"""Module to identify home and school locations and trips between
them.

"""


import numpy as np
import pandas as pd
from utils.util import great_circle_dist,chunks,chunks_real,get_hour_SGT
import logging
import copy
from data_retrieval.feature_calc import calculate_stairs, calculate_aircon_co2


def getCO2(modes, dist_trav):
    #returns the CO2 for the trip. values are in g/km, so return gram of CO2
    # assumes distance in km
    #mode_to_emissions= {'car': 187, 'bus': 19, 'train': 13, 'walk': 0}
    mode_to_emissions= {6: 187.0, 5: 19.0, 4: 13.0, 3: 0.0}
    #car = 6, bus = 5, train = 4, all others 0 CO2
    co2 = sum( (mode_to_emissions[mode] * dist for mode, dist in zip(modes, dist_trav) if dist > 0 ) )
#        if co2<0:
#            logging.warning('negative co2!!!')
#            co2=0
    return co2


def checkLim_round(value_in, lim, round_decimal=4):
    if value_in > lim:
        value_out = lim
    elif value_in < 0:
        value_out = 0
    else:
        value_out = value_in
    # value_out = float(np.round(value_out,round_decimal))
    value_out = round(value_out,round_decimal)
    return value_out


def detectPOI_geov(data_frame, stopped_thresh, poi_min_dwell_time, loc_round_decimals):
    """ Find POI's in the data frame, return a list of POI locations
        input:
        data_frame: pandas data frame of all day data
        stopped_thresh: maximum geo-velocity below which the point is considered as stopped
        poi_min_dwell_time: minimum time above which a POI is caught
    """

    def store_poi():
        """ Identify the lat/lon location of a POI
            idx_buffer has the indices in the data frame for this POI.
            stop_time counts the total stationary time of the current place
        """
        # if the total stop time is larger than poi_min_dwell_time,
        # then a poi is detected, otherwise return
        if stop_time <= poi_min_dwell_time:
            return
        # select lat/lon locations for this POI, assignment makes a copy of the rows
        df_stop_all = data_frame.loc[idx_buffer][['WLATITUDE', 'WLONGITUDE']]
        # check if all location are nan
        lat_stop = df_stop_all['WLATITUDE'].values
        if np.all(np.isnan(lat_stop)):
            return
        # get the mean lat/lon as the POI
        poi_lat, poi_lon = np.mean(df_stop_all.values[~np.isnan(lat_stop),:], axis=0)
        # record poi lat/lon and the indices that correspond to it
        pois_latlon.append([float(round(poi_lat,loc_round_decimals)),float(round(poi_lon,loc_round_decimals))]) # POI location

    # start of detectPOI_geov()
    pois_latlon = []
    stop_time = 0
    idx_buffer = []
    for index, row in data_frame.iterrows():
        if row['MOV_AVE_VELOCITY'] < stopped_thresh:
            # if stop time is zero we are at the beginning of a new stop
            # reset the idx_buffer
            if stop_time == 0:
                idx_buffer = []
                if index!=0:
                    idx_buffer = [index-1]
            # remember that we are stopped this index
            idx_buffer.append(index)
            # add the time delta of this row to the time we are stopped here
            stop_time += row['TIME_DELTA']
        else:
            # we are moving, check if we have just concluded a POI and store the POI
            if stop_time > 0:
                store_poi()
            stop_time = 0

    # process the last stop after exiting loop
    if stop_time > 0:
        store_poi()

    return pois_latlon


def identify_home_school(pois_latlon, data_frame, school_start,
                         school_end, home_start, home_end,
                         min_school_thresh, poi_cover_radius):
    """Identify home and school locations. 
    Input:
    pois_latlon: list of [lat,lon] of POIs, format: [[lat,lon],[lat,lon],...]
    data_frame: the pandas data frame of the original data.
    school_start: the hour of the day when school starts.
    school_end: the hour of the day when school end.
    home_start: the first hour of the day when students are assumed to be
    home at night.
    home_end: the last hour of the day when students are assumed to be
    home at night.
    poi_cover_radius: the covering rage of POI, with which the points are considered as inside this POI
    Output:
    home_loc: [lat,lon]
    school_loc: [lat,lon]
    pois_label_temp: temporary labels for each POI in the given list, -2 for home, -3 for school , -4 to -n for others

    """

    if len(pois_latlon)==0:
        return [None,None],[None,None],[]

    # get out time data
    all_sgt = data_frame[['TIME_SGT']].values
    all_delta_time = data_frame[['TIME_DELTA']].values

    # initialization
    time_at_school = [] # list of integers: time during the school time of each POI
    time_at_home = [] # list of integers: time during the home time of each POI

    # go though each poi
    for poi in pois_latlon:

        """ search the surrounding points and add into the POI """
        # calculate the distance from each point to the poi
        all_latlon = data_frame[['WLATITUDE', 'WLONGITUDE']].values
        dist_to_poi = map(lambda x: great_circle_dist(x, poi, unit="meters"), all_latlon)
        # find the indices of points that are near to this poi
        mask_near_poi = np.array(dist_to_poi)<=poi_cover_radius
        poi_sgt = all_sgt[mask_near_poi]
        poi_delta_time = all_delta_time[mask_near_poi]

        # find the idx of points among current idx_poi which fit school time range
        time_at_school.append(np.sum(poi_delta_time[(poi_sgt >= school_start) & (poi_sgt < school_end)]))

        # find the idx of points among current idx_poi which fit home time range
        # logging.debug('poi: '+str(poi))
        # logging.debug("home_end: "+str(home_end))
        # logging.debug("home_start: "+str(home_start))
        # logging.debug("poi_sgt: "+str(poi_sgt))
        time_at_home.append(np.sum(poi_delta_time[(poi_sgt < home_end) | (poi_sgt >= home_start)]))

    logging.info("List of time at school: "+str(time_at_school))
    logging.info("List of time at home: "+str(time_at_home))
    # get school
    max_sch_time = max(time_at_school)
    idx_max_sch_time = np.argmax(time_at_school)
    logging.info("Selected idx as school: "+str(idx_max_sch_time))
    # get home
    max_home_time = max(time_at_home)
    idx_max_home_time = np.argmax(time_at_home)
    logging.info("Selected idx as home: "+str(idx_max_home_time))

    # if max_sch_time and max_home_time are all zero, make all loc as None
    if max_sch_time==0 and max_home_time==0:
        home_loc = [None,None]
        school_loc = [None,None]
    # if the same pois is detected as home or school
    # decide by the school/home time
    elif idx_max_sch_time==idx_max_home_time:
        if max_sch_time > max_home_time:
            school_loc = pois_latlon[idx_max_sch_time]
            # see whether there's a 2nd largest time_at_home
            time_at_home_withoutmax = copy.copy(time_at_home)
            time_at_home_withoutmax.remove(max_home_time)
            if len(time_at_home_withoutmax)>0:
                max_home_time_2 = max(time_at_home_withoutmax)
                if max_home_time_2 > 0:
                    idx_max_home_time_2 = time_at_home.index(max_home_time_2)
                    home_loc = pois_latlon[idx_max_home_time_2]
                else:
                    home_loc = [None,None]
            else:
                home_loc = [None,None]
        elif max_sch_time < max_home_time:
            home_loc = pois_latlon[idx_max_home_time]
            # see whether there's a 2nd largest time_at_school
            time_at_school_withoutmax = copy.copy(time_at_school)
            time_at_school_withoutmax.remove(max_sch_time)
            if len(time_at_school_withoutmax)>0:
                max_sch_time_2 = max(time_at_school_withoutmax)
                if max_sch_time_2 > 0:
                    idx_max_sch_time_2 = time_at_school.index(max_sch_time_2)
                    school_loc = pois_latlon[idx_max_sch_time_2]
                else:
                    school_loc = [None,None]
            else:
                school_loc = [None,None]
        else:
            home_loc = [None,None]
            school_loc = [None,None]
    else:
        # if there is instance for school time, assign school
        if max_sch_time>0:
            school_loc = pois_latlon[idx_max_sch_time]
        else:
            school_loc = [None,None]
        # if there is instance for home time, assign home
        if max_home_time>0:
            home_loc = pois_latlon[idx_max_home_time]
        else:
            home_loc = [None,None]

    # generate the temporary labels for all POIs and return
    label_other_poi = -4
    pois_label_temp = []
    for poi in pois_latlon:
        if poi == home_loc:
            pois_label_temp.append(-2)
        elif poi == school_loc:
            pois_label_temp.append(-3)
        else:
            pois_label_temp.append(label_other_poi)
            label_other_poi -= 1

    return home_loc,school_loc,pois_label_temp


def label_pts_by_pois(pois_latlon_comb,pois_label_temp,data_frame,home_cover_radius,sch_cover_radius,poi_cover_radius,poi_min_dwell_time):
    """ label all points to detected POIs
        Input:
        pois_latlon_comb: combined pois, but not chronological
        pois_label_temp: temporary labels for each POI in the given list, -2 for home, -3 for school , -4 to -n for others
        data_frame: pandas data frame of all one day data of the device
        Output:
        pois_latlon_chro: chronological pois, can be duplicated, like: [home, school, home]
        pois_label_chro: chronological poi labels, -2 for home, -3 for school, 1 to n for other POIs
        data_frame['POI_LABEL']: label of each point indicating which POI it belongs to, -1 for none, -2 for home... [same as above]
    """

    # list of lat/lon and timestamp of all points across the day
    latlon_all = data_frame[['WLATITUDE','WLONGITUDE']].values.tolist()
    ts_all = data_frame['TIMESTAMP'].values
    sgt_all = data_frame['TIME_SGT'].values
    lat_all = data_frame['WLATITUDE'].values
    delta_dist_all = data_frame['DISTANCE_DELTA'].values
    # pt-level label indicating which poi this point belongs to
    # initialize the labels with -1
    poi_label_pt = np.array([-1]*len(latlon_all)) 

    # go through each POI, label all points
    for idx,poi in enumerate(pois_latlon_comb):
        dist2poi_list = map(lambda x: great_circle_dist(x, [poi[0],poi[1]], unit="meters"), latlon_all)
        dist2poi_array = np.array(dist2poi_list)
        if pois_label_temp[idx] == -2:
            # if the poi is home
            poi_label_pt[dist2poi_array<=home_cover_radius] = -2
        elif pois_label_temp[idx] == -3:
            # if the poi is school
            poi_label_pt[dist2poi_array<=sch_cover_radius] = -3
        else:
            # if it's other pois
            poi_label_pt[dist2poi_array<=poi_cover_radius] = pois_label_temp[idx]

    # wipe out short noise between two adjacent same pois
    poi_label_chunks = chunks_real(poi_label_pt,include_values=True)
    num_chunks =  len(poi_label_chunks)
    if num_chunks>1:
        for idx,label_chunk in enumerate(poi_label_chunks):
            if label_chunk[2]==-1:
                # go through all noise chunks
                lat_cur = lat_all[label_chunk[0]:label_chunk[1]]
                if idx == 0:
                    # if noise chunk is in the beginning
                    if (ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]]<60*5) or \
                    (float(ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]])/(label_chunk[1]-label_chunk[0])>1000) \
                    or (len(lat_cur[np.isnan(lat_cur)])*1.0/len(lat_cur)>0.9):
                        # if the chunk is small, or average delta time is big, or most points have invalid location
                        poi_label_pt[label_chunk[0]:label_chunk[1]]=poi_label_chunks[idx+1][2]
                elif idx == num_chunks-1:
                    # if noise chunk is in the end
                    if (ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]]<60*5) or \
                    (float(ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]])/(label_chunk[1]-label_chunk[0])>1000) \
                    or (len(lat_cur[np.isnan(lat_cur)])*1.0/len(lat_cur)>0.9):
                        # if the chunk is small, or average delta time is big, or most points have invalid location
                        poi_label_pt[label_chunk[0]:label_chunk[1]]=poi_label_chunks[idx-1][2]
                elif (poi_label_chunks[idx-1][2]==poi_label_chunks[idx+1][2]):
                    # if the former and latter chunks have same labels and this noise chunk is short in duration
                    # or most points have invalid loc, or most of the points are sleeping, or the average velocity is small
                    # set labels of this noise chunk as the same label
                    if (ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]]<60*20) or (len(lat_cur[np.isnan(lat_cur)])*1.0/len(lat_cur)>0.9) \
                        or (np.nansum(delta_dist_all[label_chunk[0]:label_chunk[1]])/(ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]])<1.0) \
                        or (float(ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]])/(label_chunk[1]-label_chunk[0])>1000):
                        # logging.debug("noise removed")
                        poi_label_pt[label_chunk[0]:label_chunk[1]]=poi_label_chunks[idx-1][2]

    # obtain the pois_latlon_chro and pois_label_chro
    pois_latlon_comb = np.array(pois_latlon_comb)
    pois_label_temp = np.array(pois_label_temp)
    pois_latlon_chro = []
    pois_label_chro = []
    pois_start_idx = []
    pois_start_sgt = []
    pois_end_idx = []
    pois_end_sgt = []
    num_normal_poi = 1
    poi_label_chunks = chunks(poi_label_pt,include_values=True)
    for label_chunk in poi_label_chunks:
        # go through all poi chunks chronologically
        if label_chunk[2] == -1:
            # non poi chunk
            continue
        else:
            # poi chunk
            if (ts_all[label_chunk[1]-1]-ts_all[label_chunk[0]]<poi_min_dwell_time):
                # if the poi chunk is too short, remove it
                poi_label_pt[label_chunk[0]:label_chunk[1]]=-1
            else:
                if label_chunk[2]==-2 or label_chunk[2]==-3:
                    # home or school chunk
                    pois_latlon_chro.append(pois_latlon_comb[pois_label_temp==label_chunk[2],:].tolist()[0])
                    pois_label_chro.append(label_chunk[2])
                else:
                    # for normal poi, just count from 1 to n
                    pois_latlon_chro.append(pois_latlon_comb[pois_label_temp==label_chunk[2],:].tolist()[0])
                    pois_label_chro.append(num_normal_poi)
                    poi_label_pt[label_chunk[0]:label_chunk[1]] = num_normal_poi
                    num_normal_poi += 1
                pois_start_idx.append(label_chunk[0])
                pois_start_sgt.append(sgt_all[label_chunk[0]])
                pois_end_idx.append(label_chunk[1]-1)
                pois_end_sgt.append(sgt_all[label_chunk[1]-1])
                # if there are more than two points with valid location for this poi, then remove the first 
                # and last point with valid location out
                lat_cur_poi = lat_all[label_chunk[0]:label_chunk[1]]
                idx_valid_loc = np.where(~np.isnan(lat_cur_poi))[0]
                if len(idx_valid_loc)>2:
                    if label_chunk[0]!=0: # to avoid creating invalid trips in the beginning of the day
                        poi_label_pt[label_chunk[0]:label_chunk[0]+idx_valid_loc[0]+1]=-1
                    if label_chunk[1]!=len(lat_all): # to avoid creating invalid trips in the end of the day
                        poi_label_pt[label_chunk[0]+idx_valid_loc[-1]:label_chunk[1]]=-1

    data_frame['POI_LABEL'] = pd.Series(poi_label_pt)
    pois_dict = {'pois_latlon_chro':pois_latlon_chro, 'pois_label_chro': pois_label_chro, 'pois_start_idx': pois_start_idx, \
    'pois_end_idx': pois_end_idx, 'pois_start_sgt': pois_start_sgt, 'pois_end_sgt': pois_end_sgt}
    return pois_dict


def get_daily_summary(data_frame, home_loc, school_loc, pois, mode_thresh):
    """
    function used to get the information of am and pm trips
    :param data_frame
    :param home_loc
    :param school_loc
    :param mode_thresh: integer for minimum duration of one mode segment
    :return: dictionary daily_summary

    """

    # limits on return values for dist/CO2
    dist_lim = 45.3 # max travel distance, km
    co2_lim = 17010.2
    time_lim = 3.5
    max_mode = 5 # maximum number of mode segments displayed in AM/PM trip
    max_walk = 4.0 # max distance of walking, km
    steps_factor = 1.6 # adjusting factor for steps value
        
    # intialize returned dict: daily_summary
    daily_summary = {'am_mode': [],'pm_mode': [],'am_distance': [],'pm_distance': [],'am_duration': [],'pm_duration':[],\
                    'am_duration_tot': 0,'pm_duration_tot': 0,'am_distance_tot': 0,'pm_distance_tot': 0,\
                    'am_nan_pts_perc': 0,'pm_nan_pts_perc': 0,'am_num_pts_tot': 0,'pm_num_pts_tot': 0,\
                    'travel_co2': 0,'outdoor_time': 0, 'stairs': 0, 'aircon_co2': 0, 'aircon_energy': 0, 'aircon_time':0,\
                    'home_end_sgt': None, 'sch_start_sgt': None, 'sch_end_sgt': None, 'home_start_sgt':None,\
                    'hourly_steps': [], 'tot_steps': 0, \
                    'hourly_max_noise':[], 'hourly_max_noise_indoor':[], 'hourly_max_noise_outdoor':[],'max_noise_hour':-1, 'max_noise_hour_indoor':-1, 'max_noise_hour_outdoor':-1, \
                    'max_noise':0, 'min_noise':0, 'mean_noise':0, \
                    'max_temp':0, 'min_temp':0, 'mean_temp':0, \
                    'max_hum':0, 'min_hum':0, 'mean_hum':0}

    # obtain start/end indeces of home/school
    poi_label_pt_array = data_frame['POI_LABEL'].values # -2 for home, -3 for school
    poi_label_pt_list = poi_label_pt_array.tolist()
    am_pm_triplabel_array = np.array([0]*len(poi_label_pt_list)) # 0 for non am/pm points

    if home_loc!=[None,None] and school_loc!=[None,None] and -2 in poi_label_pt_list and -3 in poi_label_pt_list:
        # home and school are valid, obtain start/end indeces of home and school
        sch_idx_pt = np.where(poi_label_pt_array==-3)[0]
        sch_start_idx = sch_idx_pt[0]
        sch_end_idx = sch_idx_pt[-1]
        home_idx_pt = np.where(poi_label_pt_array==-2)[0]
        home_before_sch_idx = home_idx_pt[home_idx_pt<sch_start_idx]
        home_after_sch_idx = home_idx_pt[home_idx_pt>sch_end_idx]
        if len(home_before_sch_idx)!=0:
            home_end_idx = home_before_sch_idx[-1]
        else:
            home_end_idx = -1
        if len(home_after_sch_idx)!=0:
            home_start_idx = home_after_sch_idx[0]
        else:
            home_start_idx = -1

        pois['am_poi'] = []
        pois['pm_poi'] = []

        if home_end_idx!=-1 and sch_start_idx!=-1:
            # if there's end of home and start of school, there's AM trip
            # get the end time of home and start time of school
            daily_summary['home_end_sgt'] = data_frame['TIME_SGT'][home_end_idx]
            daily_summary['sch_start_sgt'] = data_frame['TIME_SGT'][sch_start_idx]
            # get raw indeces of am trip
            am_trip_idx_all = range(home_end_idx,sch_start_idx+1)
            am_pois_idx_all = []
            # compare the start/end indices of each POI with the start/end indices of am trip
            num_pois = len(pois['pois_start_idx'])
            for i_poi in xrange(0,num_pois):
                if pois['pois_start_idx'][i_poi]>home_end_idx and pois['pois_end_idx'][i_poi]<sch_start_idx:
                    # if one poi is within the trip, then it belongs to this trip
                    # not apply the poi_cover_radius
                    pois['am_poi'].append(i_poi)
                    am_pois_idx_all = am_pois_idx_all+range(pois['pois_start_idx'][i_poi],pois['pois_end_idx'][i_poi]+1)
            # define a new dataframe for the AM trip
            am_trip_idx_no_poi = sorted(set(am_trip_idx_all)-set(am_pois_idx_all))
            df_am = data_frame.loc[am_trip_idx_no_poi]
            am_pm_triplabel_array[am_trip_idx_no_poi] = 1 # 1 for am trip points
            isAM = True
            # get the analytic summary for am trips
            num_mode_seg_am = segFind(df_am, daily_summary, mode_thresh, isAM, dist_lim, max_mode, max_walk)
            daily_summary['am_distance_tot'] = sum(daily_summary['am_distance'])
            daily_summary['am_duration_tot'] = sum(daily_summary['am_duration'])
            get_pt_perc(df_am,daily_summary,isAM)
        else:
            logging.warning("No AM trip is detected.")

        if sch_end_idx!=-1 and home_start_idx!=-1:
            # if there's end of school and start of home, there's PM trip
            # get the end time of school and start time of home
            daily_summary['sch_end_sgt'] = data_frame['TIME_SGT'][sch_end_idx]
            daily_summary['home_start_sgt'] = data_frame['TIME_SGT'][home_start_idx]
            # get raw indeces of pm trip
            pm_trip_idx_all = range(sch_end_idx,home_start_idx+1)
            pm_pois_idx_all = []
            # compare the start/end indices of each POI with the start/end indices of am trip
            num_pois = len(pois['pois_start_idx'])
            for i_poi in xrange(0,num_pois):
                if pois['pois_start_idx'][i_poi]>sch_end_idx and pois['pois_end_idx'][i_poi]<home_start_idx:
                    # if one poi is within the trip, then it belongs to this trip
                    # not apply the poi_cover_radius
                    pois['pm_poi'].append(i_poi)
                    pm_pois_idx_all = pm_pois_idx_all+range(pois['pois_start_idx'][i_poi],pois['pois_end_idx'][i_poi]+1)
            # define a new dataframe for the PM trip
            pm_trip_idx_no_poi = sorted(set(pm_trip_idx_all)-set(pm_pois_idx_all))
            df_pm = data_frame.loc[pm_trip_idx_no_poi]
            am_pm_triplabel_array[pm_trip_idx_no_poi] = 2 # 2 for pm trip points
            isAM = False
            # get the analytic summary for pm trips
            num_mode_seg_pm = segFind(df_pm, daily_summary, mode_thresh, isAM, dist_lim, max_mode, max_walk)
            daily_summary['pm_distance_tot'] = sum(daily_summary['pm_distance'])
            daily_summary['pm_duration_tot'] = sum(daily_summary['pm_duration'])
            get_pt_perc(df_pm,daily_summary,isAM)
        else:
            logging.warning("No PM trip is detected.")
    else:
        logging.warning("Lacking valid home or school locations. No AM nor PM trip is detected.")

    # calculate outside time in hours
    outdoor = data_frame.loc[data_frame['CCMODE'].isin([0,2,4,5,6])]
    if not outdoor.empty:
        daily_summary['outdoor_time']= checkLim_round(outdoor['TIME_DELTA'].sum()/3600.0, time_lim)

    # sum CO2 for AM and PM travel
    if ((daily_summary['am_mode']==[3] and daily_summary['pm_mode']==[]) or
        (daily_summary['am_mode']==[] and daily_summary['pm_mode']==[3]) or
        (daily_summary['am_mode']==[3] and daily_summary['pm_mode']==[3])):
        daily_summary['travel_co2'] += 0.00001
    else:
        daily_summary['travel_co2'] += getCO2(daily_summary['am_mode'], daily_summary['am_distance'])
        daily_summary['travel_co2'] += getCO2(daily_summary['pm_mode'], daily_summary['pm_distance'])
        daily_summary['travel_co2'] = checkLim_round(daily_summary['travel_co2'], co2_lim)

    # add am_pm_triplabel into the data frame
    data_frame['am_pm_triplabel'] = pd.Series(am_pm_triplabel_array.tolist())

    # calculate the climbed stairs and air-conditioning eneryg/co2
    daily_summary['stairs'] = calculate_stairs(data_frame)
    total_aircon_energy, total_aircon_co2, total_aircon_time = calculate_aircon_co2(data_frame)
    daily_summary['aircon_co2'] = total_aircon_co2
    daily_summary['aircon_energy'] = total_aircon_energy
    daily_summary['aircon_time'] = total_aircon_time

    # get the SGT in hours and floor the values
    data_frame['int_hour_SGT'] = np.floor(data_frame['TIME_SGT'].values)
    # initialization
    hourly_steps = []
    hourly_max_noise = []
    hourly_max_noise_indoor = []
    hourly_max_noise_outdoor = []
    # go through each hour, from 0 to 23
    for i_hour in xrange(0,24):
        df_cur_hour = data_frame.loc[data_frame['int_hour_SGT']==i_hour]
        if len(df_cur_hour)>0:
            hourly_steps.append((df_cur_hour['STEPS'].values[-1]-df_cur_hour['STEPS'].values[0])*steps_factor)
            hourly_max_noise.append(np.max(df_cur_hour['NOISE'].values))
            outdoor_mask = np.bitwise_or(df_cur_hour['CCMODE'].values==0, df_cur_hour['CCMODE'].values==2);
            indoor_mask = np.bitwise_or(df_cur_hour['CCMODE'].values==1, df_cur_hour['CCMODE'].values==3);
            if outdoor_mask.any():
                # there are outdoor pts
                hourly_max_noise_outdoor.append(np.max(df_cur_hour['NOISE'].values[outdoor_mask]))
            else:
                hourly_max_noise_outdoor.append(-1)
            if indoor_mask.any():
                # there are indoor pts
                hourly_max_noise_indoor.append(np.max(df_cur_hour['NOISE'].values[indoor_mask]))
            else:
                hourly_max_noise_indoor.append(-1)
        else:
            hourly_steps.append(-1)
            hourly_max_noise.append(-1)
            hourly_max_noise_outdoor.append(-1)
            hourly_max_noise_indoor.append(-1)

    # add the step count every hour as a list, with a 1.6 adjusting factor
    # for getStepsHour
    daily_summary['hourly_steps'] = hourly_steps

    # add the step count of every day, with a 1.6 adjusting factor
    # for getSteps
    daily_summary['tot_steps'] = (data_frame['STEPS'].values[-1]-data_frame['STEPS'].values[0])*steps_factor

    # add the largest noise value in each hour as a list, including both indoor and outdoor
    # for getLoudNoiseNew()
    daily_summary['hourly_max_noise'] = hourly_max_noise

    # add the largest noise value in each hour as a list, outdoor (-1 for indoor)
    daily_summary['hourly_max_noise_outdoor'] = hourly_max_noise_outdoor

    # add the largest noise value in each hour as a list, indoor (-1 for outdoor)
    daily_summary['hourly_max_noise_indoor'] = hourly_max_noise_indoor

    # add the hour when largest noise (>=85) happens in this day, including both indoor and outdoor
    if max(hourly_max_noise)>=85:
        daily_summary['max_noise_hour'] = hourly_max_noise.index(max(hourly_max_noise))
    
    # add the hour when largest noise (>=85) happens in this day, outdoor
    # for getLoudNoiseAlt(), getLouadNoise()
    if max(hourly_max_noise_outdoor)>=85:
        daily_summary['max_noise_hour_outdoor'] = hourly_max_noise_outdoor.index(max(hourly_max_noise_outdoor))
    
    # add the hour when largest noise (>=85) happens in this day, indoor
    if max(hourly_max_noise_indoor)>=85:
        daily_summary['max_noise_hour_indoor'] = hourly_max_noise_indoor.index(max(hourly_max_noise_indoor))

    # add max, min, mean of noise, temperature, humidity respectively
    daily_summary['max_noise'] = np.max(data_frame['NOISE'].values)
    daily_summary['min_noise'] = np.min(data_frame['NOISE'].values)
    daily_summary['mean_noise'] = np.mean(data_frame['NOISE'].values)
    daily_summary['max_temp'] = np.max(data_frame['TEMPERATURE'].values)
    daily_summary['min_temp'] = np.min(data_frame['TEMPERATURE'].values)
    daily_summary['mean_temp'] = np.mean(data_frame['TEMPERATURE'].values)
    daily_summary['max_hum'] = np.max(data_frame['HUMIDITY'].values)
    daily_summary['min_hum'] = np.min(data_frame['HUMIDITY'].values)
    daily_summary['mean_hum'] = np.mean(data_frame['HUMIDITY'].values)


    return daily_summary


def segFind(data_frame, daily_summary, mode_thresh, isAM, dist_lim, max_mode, max_walk):
    # function used to get mode segments out from a trip data frame and obtain the mode and dist

    # definition of mode code
    MODE_WALK_IN = 3;
    MODE_WALK_OUT = 2;
    MODE_STOP_IN = 1;
    MODE_STOP_OUT = 0;

    # thresholds for calculating distance of mode segs
    ALL_WALK_TIME=15*60   # time shorter than which the distance of walk mode segment is calculated using all points
    real_to_jump_dist = 2;  # for short walking seg, limit the distance by 2 times the jump distance

    pred_modes = data_frame[['CCMODE']].values[:,0] # take out the predicted modes, copy of the data_frame column
    # change all STOP_IN, STOP_OUT, WALK_OUT to WALK_IN
    pred_modes[(pred_modes==MODE_STOP_IN) | (pred_modes==MODE_STOP_OUT) | (pred_modes==MODE_WALK_OUT)] = MODE_WALK_IN
    mode_segs = list(chunks(pred_modes,True)) # take the mode chunks
    num_valid_mode_seg = 0
    prev_mode = 0

    # print pred_modes
    # print mode_segs
    # print data_frame['DISTANCE_DELTA'].values.tolist()
    # print data_frame['TIME_DELTA'].values.tolist()

    # go through each mode chunk
    for mode_seg in mode_segs:
        time_span = np.sum(data_frame['TIME_DELTA'].values[mode_seg[0]:mode_seg[1]])

        # abandon if the total segment time is less than threshold, and shorten the list down to 5 mode segments at most
        if time_span < mode_thresh or num_valid_mode_seg > max_mode-1:
            continue
        else:
            latlon_start = [data_frame['WLATITUDE'].values[mode_seg[0]],data_frame['WLONGITUDE'].values[mode_seg[0]]]
            latlon_end = [data_frame['WLATITUDE'].values[mode_seg[1]-1],data_frame['WLONGITUDE'].values[mode_seg[1]-1]]
            jump_dist = great_circle_dist(latlon_start,latlon_end,'meters')
            num_valid_mode_seg += 1
            if isAM:
                mode_key = 'am_mode'
                dist_key = 'am_distance'
                dura_key = 'am_duration'
            else:
                mode_key = 'pm_mode'
                dist_key = 'pm_distance'
                dura_key = 'pm_duration'

            # calculate the distance of this mode segment
            if int(mode_seg[2]) == MODE_WALK_IN:
                modes_cur_seg = data_frame['CCMODE'].values[mode_seg[0]:mode_seg[1]]
                dist_cur_seg = data_frame['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]]
                if time_span<ALL_WALK_TIME:
                    # if the time span is too small, consider all 0-3 modes as walking
                    dist_seg = np.nansum(dist_cur_seg)
                    dist_seg = checkLim_round(dist_seg,jump_dist*real_to_jump_dist)
                else:
                    # else if the time span is not too small, only consider 2 and 3 modes
                    dist_seg = np.nansum(dist_cur_seg[np.where((modes_cur_seg==MODE_WALK_IN) | (modes_cur_seg==MODE_WALK_OUT))[0]])
                    dist_seg = checkLim_round(dist_seg,max_walk*1000)
            else:
                dist_seg = np.nansum(data_frame['DISTANCE_DELTA'].values[mode_seg[0]:mode_seg[1]])

            if dist_seg==0 or np.isnan(dist_seg):
                # filter out the zero or nan values of dist_seg
                continue

            if mode_seg[2]==prev_mode:
                # if the current mode is same to the previous one, combine the two distance
                prev_dist = daily_summary[dist_key][len(daily_summary[dist_key])-1]
                cur_dist = checkLim_round((prev_dist*1000+dist_seg) / 1000,dist_lim)
                daily_summary[dist_key][len(daily_summary[dist_key])-1]=float(np.round(cur_dist,4))

                prev_dura = daily_summary[dura_key][len(daily_summary[dura_key])-1]
                cur_dura = prev_dura+time_span
                daily_summary[dura_key][len(daily_summary[dura_key])-1]=int(cur_dura)
                continue

            daily_summary[mode_key].append(int(mode_seg[2])) # append the mode
            daily_summary[dist_key].append(checkLim_round(dist_seg / 1000,dist_lim))
            daily_summary[dura_key].append(int(time_span))
            prev_mode = mode_seg[2]

    return num_valid_mode_seg


def get_pt_perc(data_frame,daily_summary,isAM):
    """function used to calculate the percentage of points close to any MRT stations in a data frame
    :param data_frame: given data frame
    :param daily_summary: dictionary of trip info where to return the percentage
    :param isAM: whether it's a am trip or pm trip

    """
    if isAM == True:
        key_nan_pts = "am_nan_pts_perc"
        key_num_pts = "am_num_pts_tot"
    else:
        key_nan_pts = "pm_nan_pts_perc"
        key_num_pts = "pm_num_pts_tot"

    # list of boolean which indicates whether the point is close to MRT stations
    # value is nan if the location is nan
    num_valid_pts_tot = 0
    num_pts_nan = 0
    latlon_all = data_frame[['WLATITUDE','WLONGITUDE']].values.tolist()
    for latlon in latlon_all:
        if not np.isnan(latlon[0]):
            # if the location is not valid
            num_valid_pts_tot += 1
        else:
            num_pts_nan += 1

    round2decimal = 4
    num_pts_tot = num_pts_nan+num_valid_pts_tot
    logging.debug("Number of total points: "+str(num_pts_tot))
    logging.debug("Number of total nan points: "+str(num_pts_nan))
    logging.debug("Number of valid points: "+str(num_valid_pts_tot))
    daily_summary[key_nan_pts] = round(float(num_pts_nan)/num_pts_tot,round2decimal)
    daily_summary[key_num_pts] = num_pts_tot
    return None