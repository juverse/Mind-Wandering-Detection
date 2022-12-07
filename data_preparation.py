####### data preparation Mind Wandering ######

### dependecies ###
import pandas as pd 
import numpy as np

### read data ###
df_all = pd.read_csv(r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_Eventstatistics\Event Statistics - Single - 01.txt"
, na_values = "-")
df_all

df_initial = df_all.drop(columns=["Trial Start Raw Time [ms]", "Trial Start Time of Day [h:m:s:ms]", "Export Start Trial Time [ms]" , 
"Color", "Category Group","Event Start Raw Time [ms]","Event End Raw Time [ms]", "Annotation Name", "Annotation Description", "Annotation Tags", 
'Fixation Position X [px]', 'Fixation Position Y [px]', "Mouse Position X [px]",
"Mouse Position Y [px]", "Scroll Direction X", "Scroll Direction Y","Port Status", "Content", "Index"]) #, inplace = True)
df_initial

### cut last 10 seconds of videos ###

# 1. df split in vidoes => more efficint
# 2. function for finding the last  10 seconds from each video (it is the MW time)
#      2.1 find the interval, which we are interested in [end of video-10s, end of vide]
#      2.2 which events are in this interval: if event start time > start_vid1 und event end time < end_vid1
# 3. perform this functoin for each video
# 4. bring all df for each video together

df_split = df_initial.groupby("Stimulus") 

## finding the last 10 seconds of a video
# df = df of videos; num in milliseconds
def get_vid_last_sec(df, num):
    end_obs = df_initial.loc[1, "Export End Trial Time [ms]"]#for all same end time, so I can use the first end time
    # extract export end trial for video, get first row
    return (end_obs  - num, end_obs)

def event_end_trial_in_float(df):
    return df.loc[:, "Event End Trial Time [ms]"].astype(float) #df["Event End Trial Time [ms]"].astype(float)

## filter all rows which are between the end and the last seconds of the video
# df = df of one video
def vid_nec(df):
    # number of last seconds
    start_obs = get_vid_last_sec(df, 10000)[0]
    end_obs = get_vid_last_sec(df, 10000)[1]

    # preparation of df
    df_event_end_trial = event_end_trial_in_float(df)

    #get df last seconds: 
    df_vid_final = df.loc[(df_event_end_trial > start_obs) & (df_event_end_trial <= end_obs), :]
    return df_vid_final

## necessary data together
df_concat = pd.DataFrame()
for i in range(1,16):
    df_temp = pd.DataFrame(df_split.get_group("vid" +str(i)))
    df_concat = pd.concat([df_concat, vid_nec(df_temp)])

### prepare eye traacking data ###

## blink duration:
# 1. 150 - 250 mseconds  Stern, J. A., Skelly, J.J. 1984. The eyeblink and workload considerations. Proceedings of the
#human factors society, 28th Annual meeting Santa Monica: Human Factors Society. 

# 2. 100 - 400 mseconds 
# Schiffman, H.R., Sensation and Perception. An Integrated Approach, New York: John Wiley and Sons, Inc., 2001

# => threshold:  500 mseconds  ?
blink_duration = df_concat.loc[df_concat["Category"] == "Blink", :]
blink_threshold = 500

# df without too long blinks
df_concat_blink = df_concat.loc[(df_concat["Category"] != "Blink") |
                                ((df_concat["Category"] == "Blink") & (df_concat["Event Duration [ms]"] < blink_threshold))]

## split data in right and left eye => just take one eye for statistics
df_final_left = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Left", :]
df_final_right = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Right", :]


### summary statistics for Left Eye###

#groupby left for the videos
df_final_left_groups_sti = df_final_left.groupby("Stimulus") 
df_final_left_groups_sti_cat = df_final_left.groupby(["Stimulus", "Category"])

# summary statistics
df_summary_stat_left = pd.DataFrame()

## add general information
df_summary_stat_left["Tracking Ratio [%] Mean"] = df_final_left_groups_sti["Tracking Ratio [%]"].mean()
df_summary_stat_left["Participant"] = [df_initial.loc[1,"Participant"]] * 15

## Features
#statistics I want for all features
statistics = ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25),lambda x: x.quantile(0.75)]
#kurtosis
def get_kurtosis(feature):
    return df_final_left_groups_sti[feature].apply(pd.DataFrame.kurt).values


## Duration:
# Fixation Duration
# # 1.step: calcaulte the statistics 
event_duration = df_final_left_groups_sti_cat.agg({"Event Duration [ms]": statistics})

# 2. to get dataframe after aggretaion with two groups
event_duration.columns = ['Event_duration_mean', "Event_duration_max", "Event_duration_min", "Event_duration_median", "Event_duration_std", "Event_duration_skew",
"Event_duration_q25", "Event_duration_q75"]
event_duration = event_duration.reset_index()

# 3. add Fixation duration to dataframe
def get_event_duration(event):
    return event_duration.loc[event_duration["Category"] == event, ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]].values
 

df_summary_stat_left[["Fixation Duration Mean [ms]", 
                        "Fixation Duration Max [ms]", 
                        "Fixation Duration Min [ms]",
                        "Fixation Duration Median [ms]",
                        "Fixation Duration Std [ms]",
                        "Fixation Duration Skew [ms]",
                        "Fixation Duration Quantil 25 [ms]",
                        "Fixation Duration Quantil 75 [ms]"]] = get_event_duration("Fixation") 
# index matching causes Nan values, we havt to drop out the indexes, so we can add the values or we just take the values!

## Saccade Duration
# 4. add Saccade duration to dataframe
df_summary_stat_left[["Saccade Duration Mean [ms]", 
                        "Saccade Duration Max [ms]", 
                        "Saccade Duration Min [ms]",
                        "Saccade Duration Median [ms]",
                        "Saccade Duration Std [ms]",
                        "Saccade Duration Skew [ms]",
                        "Saccade Duration Quantil 25 [ms]",
                        "Saccade Duration Quantil 75 [ms]"]] = get_event_duration("Saccade") 
# index matching causes Nan values, we havt to drop out the indexes, so we can add the values or we just take the values!

# kurtosis
kurt = df_final_left_groups_sti_cat["Event Duration [ms]"].apply(pd.DataFrame.kurt)
kurt = kurt.reset_index()
# add to dataframe 
df_summary_stat_left["Fixation Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Fixation", "Event Duration [ms]"].values
df_summary_stat_left["Saccade Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Saccade", "Event Duration [ms]"].values
df_summary_stat_left["Blink Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Blink", "Event Duration [ms]"].values
#values otherwise we would add NaN, because of the indexes

## Mean Duration of Blink
#brauche ich hier die anderen-> max, min??
df_summary_stat_left[["Blink Duration Mean [ms]", 
                        "Blink Duration Max [ms]", 
                        "Blink Duration Min [ms]",
                        "Blink Duration Median [ms]",
                        "Blink Duration Std [ms]",
                        "Blink Duration Skew [ms]",
                        "Blink Duration Quantil 25 [ms]",
                        "Blink Duration Quantil 75 [ms]"]] = get_event_duration("Blink")

## saccade fixation ratio
#Das Verhältnis zweier Zahlen zeigt, wie viel Mal die erste Zahl größer ist
#als die zweite oder welchen Anteil die erste Zahl von der zweiten ausmacht.
def get_sacc_fix_ratio(fix, sacc):
    return df_summary_stat_left[fix]/df_summary_stat_left[sacc]

df_summary_stat_left["Fixation Saccade Ratio Mean"] = get_sacc_fix_ratio("Fixation Duration Mean [ms]","Saccade Duration Mean [ms]")
df_summary_stat_left["Fixation Saccade Ratio Max"] = get_sacc_fix_ratio("Fixation Duration Max [ms]", "Saccade Duration Max [ms]")
df_summary_stat_left["Fixation Saccade Ratio Min"] = get_sacc_fix_ratio("Fixation Duration Min [ms]","Saccade Duration Min [ms]" )
df_summary_stat_left["Fixation Saccade Ratio Median"] = get_sacc_fix_ratio("Fixation Duration Median [ms]","Saccade Duration Median [ms]" )
df_summary_stat_left["Fixation Saccade Ratio Std"] = get_sacc_fix_ratio("Fixation Duration Std [ms]", "Saccade Duration Std [ms]")
df_summary_stat_left["Fixation Saccade Ratio Skew"] = get_sacc_fix_ratio("Fixation Duration Skew [ms]","Saccade Duration Skew [ms]" )
df_summary_stat_left["Fixation Saccade Ratio Kurtosis"] = get_sacc_fix_ratio("Fixation Duration Kurtosis [ms]","Saccade Duration Kurtosis [ms]") #.values

## Fixation Number
# take on fixation value and count how often it appears
fix_count= df_final_left_groups_sti.agg({"Fixation Average Pupil Size X [px]": "count"})
df_summary_stat_left["Fixation Number"] = fix_count

## Blink Number
blink_count = df_final_left_groups_sti_cat.count()["Trial"][:, "Blink"]
df_summary_stat_left["Blink Number"] = blink_count

## Fixation Dispersion x und y
fix_dis_x_y = df_final_left_groups_sti.agg({"Fixation Dispersion X [px]": statistics, "Fixation Dispersion Y [px]": statistics })

fix_dis_x_y.columns = ['fix_dis_x_mean', "fix_dis_x_max", "fix_dis_x_min", "fix_dis_x_median", "fix_dis_x_std", "fix_dis_x_skew", "fix_dis_x_q25","fix_dis_x_q75",
'fix_dis_y_mean', "fix_dis_y_max", "fix_dis_y_min", "fix_dis_y_median", "fix_dis_y_std", "fix_dis_y_skew", "fix_dis_y_q25", "fix_dis_y_q75"]

df_summary_stat_left[["Fixation Dispersion X Mean [px]", 
                        "Fixation Dispersion X Max [px]", 
                        "Fixation Dispersion X Min [px]",
                        "Fixation Dispersion X Median [px]",
                        "Fixation Dispersion X Std [px]",
                        "Fixation Dispersion X Skew [px]",
                        "Fixation Dispersion X Quantil 25 [px]",
                        "Fixation Dispersion X Quantil 75 [px]",
                        
                        "Fixation Dispersion Y Mean [px]", 
                        "Fixation Dispersion Y Max [px]", 
                        "Fixation Dispersion Y Min [px]",
                        "Fixation Dispersion Y Median [px]",
                        "Fixation Dispersion Y Std [px]",
                        "Fixation Dispersion Y Skew [px]",
                        "Fixation Dispersion Y Quantil 25 [px]",
                        "Fixation Dispersion Y Quantil 75 [px]"]] = fix_dis_x_y

df_summary_stat_left["Fixation Dispersion X Kurtosis [px]"] = get_kurtosis("Fixation Dispersion X [px]")
df_summary_stat_left["Fixation Dispersion Y Kurtosis [px]"] = get_kurtosis("Fixation Dispersion Y [px]")

## Saccade Amplitude
sacc_ampl = df_final_left_groups_sti.agg({"Saccade Amplitude [°]": statistics})

sacc_ampl.columns = ['sacc_ampl_mean', "sacc_ampl_max", "sacc_ampl_min", "sacc_ampl_median", "sacc_ampl_std", "sacc_ampl_skew",
"sacc_ampl_q25", "sacc_ampl_q75"]
#sacc_ampl = sacc_ampl.reset_index()

df_summary_stat_left[["Saccade Amplitude Mean [°]", 
                        "Saccade Amplitude Max [°]", 
                        "Saccade Amplitude Min [°]",
                        "Saccade Amplitude Median [°]",
                        "Saccade Amplitude Std [°]",
                        "Saccade Amplitude Skew [°]",
                        "Saccade Amplitude Quantil 25 [°]",
                        "Saccade Amplitude Quantil 75 [°]"]] = sacc_ampl

df_summary_stat_left["Saccade Amplitude Kurtosis [°]"] = get_kurtosis("Saccade Amplitude [°]")

sacc_features = df_final_left_groups_sti.agg({'Saccade Acceleration Average [°/s²]': statistics,
       'Saccade Acceleration Peak [°/s²]': statistics, 
       'Saccade Deceleration Peak [°/s²]': statistics, 
       'Saccade Velocity Average [°/s]': statistics,
       'Saccade Velocity Peak [°/s]': statistics,  
       'Saccade Peak Velocity at [%]': statistics})

sacc_features.columns = ['sacc_acc_avg_mean', "sacc_acc_avg_max", "sacc_acc_avg_min", "sacc_acc_avg_median", "sacc_acc_avg_std", "sacc_acc_avg_skew", 
"sacc_acc_avg_q25", "sacc_acc_avg_q75",

'sacc_acc_peak_mean', "sacc_acc_peak_max", "sacc_acc_peak_min", "sacc_acc_peak_median", "sacc_acc_peak_std", "sacc_acc_peak_skew",
"sacc_acc_peak_q25", "sacc_acc_peak_q75",

'sacc_dec_peak_mean', "sacc_dec_peak_max", "sacc_dec_peak_min", "sacc_dec_peak_median", "sacc_dec_peak_std", "sacc_dec_peak_skew",
"sacc_dec_peak_q25", "sacc_dec_peak_q75",

'sacc_vel_avg_mean', "sacc_vel_avg_max", "sacc_vel_avg_min", "sacc_vel_avg_median", "sacc_vel_avg_std", "sacc_vel_avg_skew",
"sacc_vel_avg_q25", "sacc_vel_avg_q75",

'sacc_vel_peak_mean', "sacc_vel_peak_max", "sacc_vel_peak_min", "sacc_vel_peak_median", "sacc_vel_peak_std", "sacc_vel_peak_skew",
"sacc_vel_peak_q25", "sacc_vel_peak_q75",

'sacc_vel_peak_percent_mean', "sacc_vel_peak_percent_max", "sacc_vel_peak_percent_min", "sacc_vel_peak_percent_median", "sacc_vel_peak_percent_std", 
"sacc_vel_peak_percent_skew", "sacc_vel_peak_percent_q25", "sacc_vel_peak_percent_q75"]

df_summary_stat_left[["Saccade Acceleration Average [°/s²] Mean", 
                        "Saccade Acceleration Average [°/s²] Max", 
                        "Saccade Acceleration Average [°/s²] Min",
                        "Saccade Acceleration Average [°/s²] Median",
                        "Saccade Acceleration Average [°/s²] Std",
                        "Saccade Acceleration Average [°/s²] Skew]",
                        "Saccade Acceleration Average [°/s²] Quantil 25]",
                        "Saccade Acceleration Average [°/s²] Quantil 75]",

                        "Saccade Acceleration Peak [°/s²] Mean", 
                        "Saccade Acceleration Peak [°/s²] Max", 
                        "Saccade Acceleration Peak [°/s²] Min",
                        "Saccade Acceleration Peak [°/s²] Median",
                        "Saccade Acceleration Peak [°/s²] Std",
                        "Saccade Acceleration Peak [°/s²] Skew]",
                        "Saccade Acceleration Peak [°/s²] Quantil 25]",
                        "Saccade Acceleration Peak [°/s²] Quantil 75]",

                        "Saccade Deceleration Peak [°/s²] Mean", 
                        "Saccade Deceleration Peak [°/s²] Max", 
                        "Saccade Deceleration Peak [°/s²] Min",
                        "Saccade Deceleration Peak [°/s²] Median",
                        "Saccade Deceleration Peak [°/s²] Std",
                        "Saccade Deceleration Peak [°/s²] Skew]",
                        "Saccade Deceleration Peak [°/s²] Quantil 25]",
                        "Saccade Deceleration Peak [°/s²] Quantil 75]",

                        "Saccade Velocity Average [°/s²] Mean", 
                        "Saccade Velocity Average [°/s²] Max", 
                        "Saccade Velocity Average [°/s²] Min",
                        "Saccade Velocity Average [°/s²] Median",
                        "Saccade Velocity Average [°/s²] Std",
                        "Saccade Velocity Average [°/s²] Skew]",
                        "Saccade Velocity Average [°/s²] Quantil 25]",
                        "Saccade Velocity Average [°/s²] Quantil 75]",

                        "Saccade Velocity Peak [°/s²] Mean", 
                        "Saccade Velocity Peak [°/s²] Max", 
                        "Saccade Velocity Peak [°/s²] Min",
                        "Saccade Velocity Peak [°/s²] Median",
                        "Saccade Velocity Peak [°/s²] Std",
                        "Saccade Velocity Peak [°/s²] Skew]",
                        "Saccade Velocity Peak [°/s²] Quantil 25]",
                        "Saccade Velocity Peak [°/s²] Quantil 75]",

                        "Saccade Velocity Peak [%] Mean", 
                        "Saccade Velocity Peak [%] Max", 
                        "Saccade Velocity Peak [%] Min",
                        "Saccade Velocity Peak [%] Median",
                        "Saccade Velocity Peak [%] Std",
                        "Saccade Velocity Peak [%] Skew]",
                        "Saccade Velocity Peak [%] Quantil 25]",
                        "Saccade Velocity Peak [%] Quantil 75]"
                        ]] = sacc_features

df_summary_stat_left["Saccade Acceleration Average [°/s²] Kurtosis"] = get_kurtosis("Saccade Acceleration Average [°/s²]")
df_summary_stat_left["Saccade Acceleration Peak [°/s²] Kurtosis"] = get_kurtosis("Saccade Acceleration Peak [°/s²]")
df_summary_stat_left["Saccade Deceleration Peak [°/s²] Kurtosis"] = get_kurtosis("Saccade Deceleration Peak [°/s²]")
df_summary_stat_left["Saccade Velocity Average [°/s²] Kurtosis"] = get_kurtosis("Saccade Velocity Average [°/s]")
df_summary_stat_left["Saccade Velocity Peak [°/s²] Kurtosis"] = get_kurtosis("Saccade Velocity Peak [°/s]")
df_summary_stat_left["Saccade Velocity Peak [%] Kurtosis"] = get_kurtosis("Saccade Peak Velocity at [%]")

## Saccade Length: Distance of saccade in pixels
# distanz = sqrt((end_X - start_x)^2 + (end_y - start_y)^2)

# 1. calcualte distance and make new column in df
def diff(start,end):
    return (df_final_left[end].astype(float) - df_final_left[start].astype(float))**2 

x_dist = diff("Saccade End Position X [px]", "Saccade Start Position X [px]" )
y_dist = diff("Saccade End Position Y [px]", "Saccade Start Position Y [px]" )

distance = np.sqrt(x_dist + y_dist)
df_final_left["Saccade Length [px]"] = distance

# 2 calcualte mean vor each video and so on
sacc_length = df_final_left_groups_sti.agg({"Saccade Length [px]": statistics})

sacc_length.columns = ['sacc_length_mean', "sacc_length_max", "sacc_length_min", "sacc_length_median", "sacc_length_std", 
"sacc_length_skew", "sacc_length_q25", "sacc_length_q75"]
#sacc_ampl = sacc_ampl.reset_index()

df_summary_stat_left[["Saccade Length Mean [px]", 
                        "Saccade Length Max [px]", 
                        "Saccade Length Min [px]",
                        "Saccade Length Median [px]",
                        "Saccade Length Std [px]",
                        "Saccade Length Skew [px]]",
                        "Saccade Length Quantil 25 [px]]",
                        "Saccade Length Quantil 75 [px]]"]] = sacc_length

df_summary_stat_left["Saccade Length Kurtosis [px]"] = get_kurtosis("Saccade Length [px]")

print(df_summary_stat_left)

##Pupil diameters 

# pupil size with emotions
# we need a baseline, where the Person is neutral, before experiment
# I can use everything before video1 from raw data
# standardization wie value/devided by mean from before
# than we can calculate sd, mean, etc. 
# get neutral pupil size before video1 starts as a baseline for pupilsize
# for all VP different!!!!,have to change the data

df_raw_data = pd.read_csv(r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_RawData\Raw Data - Raw Data - 01.txt"
, na_values = "-", usecols= ["Stimulus", "Pupil Diameter Left [mm]", "Pupil Diameter Right [mm]"])

df_welcome_to_instruction = df_raw_data.loc[(df_raw_data["Stimulus"] == "welcome") | (df_raw_data["Stimulus"] == "transition1") 
| (df_raw_data["Stimulus"] == 'https://www.unipark.de/uc/Mind-Wandering/510e') 
| (df_raw_data["Stimulus"] == "https://www.unipark.de/uc/Mind-Wandering/510e/ospe.php?qb")
| (df_raw_data["Stimulus"] == "instruction")
]

pupil_diameter_right_mean = df_welcome_to_instruction["Pupil Diameter Right [mm]"].mean()
pupil_diameter_left_mean = df_welcome_to_instruction["Pupil Diameter Left [mm]"].mean()

#Pupil diameters 
#for each participant different pupils, but we want to compare them
# that why we need a way to standaralize them, devide them by a baseline, which we get from data before watching the video
#import features_pupil_diameter

# divide all pupil diameters for fixation in the last 10 seconds with pupil mean
df_final_left["Fixation Average Pupil Diameter [mm] standardized"] = df_final_left.loc[:, "Fixation Average Pupil Diameter [mm]"]/pupil_diameter_left_mean #.std()

# summary statistics
df_summary_stat_left[["Fixation Average Pupil Diameter [mm] Mean",
"Fixation Average Pupil Diameter [mm] Max",
"Fixation Average Pupil Diameter [mm] Min",
"Fixation Average Pupil Diameter [mm] Median",
"Fixation Average Pupil Diameter [mm] Std",
"Fixation Average Pupil Diameter [mm] Skew",
"Fixation Average Pupil Diameter [mm] Quantil25",
"Fixation Average Pupil Diameter [mm] Quantil75"]] = df_final_left_groups_sti.agg({"Fixation Average Pupil Diameter [mm] standardized":
statistics})

df_summary_stat_left["Fixation Average Pupil Diameter [mm] Kurtosis"] = get_kurtosis("Fixation Average Pupil Diameter [mm] standardized")

#### todo ###
##Vergence



