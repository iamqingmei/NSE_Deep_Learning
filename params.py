"""DB related """
# dbpw_str = "postgres"
# dbname_str = "nse2016qm"
# dbuser_str = "i337029"
# dbhost = "lssinh003.sin.sap.corp"
# dbport = "5433"


dbpw_str = "123456"
dbname_str = "nse_local"
dbuser_str = "postgres"
dbhost = "localhost"
dbport = "5432"

# dbpw_str = "123456"
# dbname_str = "nse_local"
# dbuser_str = "postgres"
# dbhost = "10.34.37.174"
# dbport = "5433"

tableTrip2016 = "allweeks_tripsummary"
tableExtra2016 = "allweeks_extra"

defalt_mixed_invalid_label = -1

window_size = 6

min_bus_localization_count = 1
min_car_localization_count = 1
min_no_veh_localization_count = 1
min_mrt_localization_count = 1


BUS_VELOCITY_THRESHOLD = 5.5
NEAR_BUS_STOP_THRESHOLD = 50

stopped_thresh = 1.0
loc_round_decimals = 5
poi_comb_range = 150
poi_comb_samples = 2
dist_round_decimals = 3

# school_start is the hour of the day when school starts. Default 9am.
school_start = 9
# school_end is the hour of the day when school end. Default 1pm.
school_end = 14
# home_start is the first hour of the day when students are assumed to be
# home at night. Default 10pm.
home_start = 22
# home_end is the last hour of the day when students are assumed to be
# home at night. Default 5am.
home_end = 6
# threshold for the minimum distance between home and school. Default 300m
min_school_thresh = 300

# poi_min_dwell_time is the time in seconds above which a stopped
# location is considered a point of interest.
poi_min_dwell_time = 10 * 60
# poi_cover_range is a distance which decides whether the other location
# points are considered as belonging to the poi. Default = 150 meter
poi_cover_radius = 150
# poi_cover_range is a distance which decides whether the other location
# points are considered as belonging to the poi. Default = 150 meter
home_cover_radius = 50
# poi_cover_range is a distance which decides whether the other location
# points are considered as belonging to the poi. Default = 150 meter
sch_cover_radius = 200