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

## necessary data
df_vid1 = pd.DataFrame(df_split.get_group("vid1"))
df_vid1_nec = vid_nec(df_vid1)

df_vid2 = pd.DataFrame(df_split.get_group("vid2"))
df_vid2_nec = vid_nec(df_vid2)

df_vid3 = pd.DataFrame(df_split.get_group("vid3"))
df_vid3_nec = vid_nec(df_vid3)

df_vid4 = pd.DataFrame(df_split.get_group("vid4"))
df_vid4_nec = vid_nec(df_vid4)

df_vid5 = pd.DataFrame(df_split.get_group("vid5"))
df_vid5_nec = vid_nec(df_vid5)

df_vid6 = pd.DataFrame(df_split.get_group("vid6"))
df_vid6_nec = vid_nec(df_vid6)

df_vid7 = pd.DataFrame(df_split.get_group("vid7"))
df_vid7_nec = vid_nec(df_vid7)

df_vid8 = pd.DataFrame(df_split.get_group("vid8"))
df_vid8_nec = vid_nec(df_vid8)

df_vid9 = pd.DataFrame(df_split.get_group("vid9"))
df_vid9_nec = vid_nec(df_vid9)

df_vid10 = pd.DataFrame(df_split.get_group("vid10"))
df_vid10_nec = vid_nec(df_vid10)

df_vid11 = pd.DataFrame(df_split.get_group("vid11"))
df_vid11_nec = vid_nec(df_vid11)

df_vid12 = pd.DataFrame(df_split.get_group("vid12"))
df_vid12_nec = vid_nec(df_vid12)

df_vid13 = pd.DataFrame(df_split.get_group("vid13"))
df_vid13_nec = vid_nec(df_vid13)

df_vid14 = pd.DataFrame(df_split.get_group("vid14"))
df_vid14_nec = vid_nec(df_vid14)

df_vid15 = pd.DataFrame(df_split.get_group("vid15"))
df_vid15_nec = vid_nec(df_vid15)

##all necessary datas together
df_concat = pd.concat([df_vid1_nec, df_vid2_nec, df_vid3_nec, df_vid4_nec, df_vid5_nec, df_vid6_nec, 
df_vid7_nec,df_vid8_nec, df_vid9_nec, df_vid10_nec, df_vid11_nec, df_vid12_nec, df_vid13_nec, df_vid14_nec,
 df_vid15_nec])

### prepare eye traacking data ###

## blink duration:
# 1. 150 - 250 mseconds  Stern, J. A., Skelly, J.J. 1984. The eyeblink and workload considerations. Proceedings of the
#human factors society, 28th Annual meeting Santa Monica: Human Factors Society. 

# 2. 100 - 400 mseconds 
# Schiffman, H.R., Sensation and Perception. An Integrated Approach, New York: John Wiley and Sons, Inc., 2001

# => threshold:  500 mseconds  ?
blink_duration = df_concat.loc[df_concat["Category"] == "Blink", :]
blink_threshold = 500

# df without to long blinks
df_concat_blink = df_concat.loc[(df_concat["Category"] != "Blink") |
                                ((df_concat["Category"] == "Blink") & (df_concat["Event Duration [ms]"] < blink_threshold))]

## combine left/right eye => just take one eye for statistics
#split data in right an dleft eye
df_final_left = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Left", :]
df_final_right = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Right", :]


### summary statistics ###

#groupby left for the videos
df_final_left_groups_sti = df_final_left.groupby("Stimulus") 
df_final_left_groups_sti_cat = df_final_left.groupby(["Stimulus", "Category"])

# summary statistics
df_summary_stat_left = pd.DataFrame()

## add general information
df_summary_stat_left["Tracking Ratio [%] Mean"] = df_final_left_groups_sti["Tracking Ratio [%]"].mean()# per group.mean()
df_summary_stat_left["Participant"] = [df_initial.loc[1,"Participant"]] * 15

## Features
## Duration:
# Fixation Duration
# 1.step: calcaulte the statistics
event_duration = df_final_left_groups_sti_cat.agg({"Event Duration [ms]": ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25),
lambda x: x.quantile(0.75)]})

# 2. to get dataframe after aggretaion with two groups
event_duration.columns = ['Event_duration_mean', "Event_duration_max", "Event_duration_min", "Event_duration_median", "Event_duration_std", "Event_duration_skew",
"Event_duration_q25", "Event_duration_q75"]
event_duration = event_duration.reset_index()

# kurtosis
kurt = df_final_left_groups_sti_cat["Event Duration [ms]"].apply(pd.DataFrame.kurt)
kurt = kurt.reset_index()
# Fixation
df_summary_stat_left["Fixation Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Fixation", "Event Duration [ms]"].values
# Saccade
df_summary_stat_left["Saccade Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Saccade", "Event Duration [ms]"].values
# Blink
df_summary_stat_left["Blink Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Blink", "Event Duration [ms]"].values
#values otherwise we would add NaN, because of the indexes

# 3. add Fixation duration to dataframe
df_summary_stat_left[["Fixation Duration Mean [ms]", 
                        "Fixation Duration Max [ms]", 
                        "Fixation Duration Min [ms]",
                        "Fixation Duration Median [ms]",
                        "Fixation Duration Std [ms]",
                        "Fixation Duration Skew [ms]",
                        "Fixation Duration Quantil 25 [ms]",
                        "Fixation Duration Quantil 75 [ms]"]] = event_duration.loc[event_duration["Category"] == "Fixation",
                                                                 ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]].values
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
                        "Saccade Duration Quantil 75 [ms]"]] = event_duration.loc[event_duration["Category"] == "Saccade",
                                                                 ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]].values #.reset_index()
# index matching causes Nan values, we havt to drop out the indexes, so we can add the values or we just take the values!

## Mean Duration of Blink
#brauche ich hier die anderen-> max, min??
df_summary_stat_left[["Blink Duration Mean [ms]", 
                        "Blink Duration Max [ms]", 
                        "Blink Duration Min [ms]",
                        "Blink Duration Median [ms]",
                        "Blink Duration Std [ms]",
                        "Blink Duration Skew [ms]",
                        "Blink Duration Quantil 25 [ms]",
                        "Blink Duration Quantil 75 [ms]"]] = event_duration.loc[event_duration["Category"] == "Blink",
                                                                 ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]].values

## saccade fixation ratio saccade duration/fixation duration
#Das Verhältnis zweier Zahlen zeigt, wie viel Mal die erste Zahl größer ist
#als die zweite oder welchen Anteil die erste Zahl von der zweiten ausmacht.

df_summary_stat_left["Fixation Saccade Ratio Mean"] = df_summary_stat_left["Fixation Duration Mean [ms]"]/df_summary_stat_left["Saccade Duration Mean [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Max"] = df_summary_stat_left["Fixation Duration Max [ms]"]/df_summary_stat_left["Saccade Duration Max [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Min"] = df_summary_stat_left["Fixation Duration Min [ms]"]/df_summary_stat_left["Saccade Duration Min [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Median"] = df_summary_stat_left["Fixation Duration Median [ms]"]/df_summary_stat_left["Saccade Duration Median [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Std"] = df_summary_stat_left["Fixation Duration Std [ms]"]/df_summary_stat_left["Saccade Duration Std [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Skew"] = df_summary_stat_left["Fixation Duration Skew [ms]"]/df_summary_stat_left["Saccade Duration Skew [ms]"]
df_summary_stat_left["Fixation Saccade Ratio Kurtosis"] = df_summary_stat_left["Fixation Duration Kurtosis [ms]"]/df_summary_stat_left["Saccade Duration Kurtosis [ms]"].values

## Fixation Number
# take on fixation value and count how often it appears
fix_count= df_final_left_groups_sti.agg({"Fixation Average Pupil Size X [px]": "count"})
df_summary_stat_left["Fixation Number"] = fix_count

## Blink Number
blink_count = df_final_left_groups_sti_cat.count()["Trial"][:, "Blink"]
df_summary_stat_left["Blink Number"] = blink_count

## Fixation Dispersion x und y
fix_dis_x_y = df_final_left_groups_sti.agg({"Fixation Dispersion X [px]": ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)], 
"Fixation Dispersion Y [px]": ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)] })

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

df_summary_stat_left["Fixation Dispersion X Kurtosis [px]"] = df_final_left_groups_sti["Fixation Dispersion X [px]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Fixation Dispersion Y Kurtosis [px]"] = df_final_left_groups_sti["Fixation Dispersion Y [px]"].apply(pd.DataFrame.kurt).values

## Saccade Amplitude
sacc_ampl = df_final_left_groups_sti.agg({"Saccade Amplitude [°]": ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]})

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

df_summary_stat_left["Saccade Amplitude Kurtosis [°]"] = df_final_left_groups_sti["Saccade Amplitude [°]"].apply(pd.DataFrame.kurt).values

sacc_features = df_final_left_groups_sti.agg({'Saccade Acceleration Average [°/s²]': ["mean", "max", "min", "median", "std", "skew",
 lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
       'Saccade Acceleration Peak [°/s²]': ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)], 
       'Saccade Deceleration Peak [°/s²]': ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)], 
       'Saccade Velocity Average [°/s]': ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
       'Saccade Velocity Peak [°/s]': ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],  
       'Saccade Peak Velocity at [%]': ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]})

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

df_summary_stat_left["Saccade Acceleration Average [°/s²] Kurtosis"] = df_final_left_groups_sti["Saccade Acceleration Average [°/s²]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Saccade Acceleration Peak [°/s²] Kurtosis"] = df_final_left_groups_sti["Saccade Acceleration Peak [°/s²]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Saccade Deceleration Peak [°/s²] Kurtosis"] = df_final_left_groups_sti["Saccade Deceleration Peak [°/s²]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Saccade Velocity Average [°/s²] Kurtosis"] = df_final_left_groups_sti["Saccade Velocity Average [°/s]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Saccade Velocity Peak [°/s²] Kurtosis"] = df_final_left_groups_sti["Saccade Velocity Peak [°/s]"].apply(pd.DataFrame.kurt).values
df_summary_stat_left["Saccade Velocity Peak [%] Kurtosis"] = df_final_left_groups_sti["Saccade Peak Velocity at [%]"].apply(pd.DataFrame.kurt).values

## Saccade Length: Distance of saccade in pixels
# distanz = sqrt((end_X - start_x)^2 + (end_y - start_y)^2)

# 1. calcualte distance and make new column in df
x_dist = (df_final_left["Saccade End Position X [px]"].astype(float) - df_final_left["Saccade Start Position X [px]"].astype(float))**2 
y_dist = (df_final_left["Saccade End Position Y [px]"].astype(float) - df_final_left["Saccade Start Position Y [px]"].astype(float))**2 

distance = np.sqrt(x_dist + y_dist)
df_final_left["Saccade Length [px]"] = distance

# 2 calcualte mean vor each video and so on
sacc_length = df_final_left_groups_sti.agg({"Saccade Length [px]": ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]})

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

df_summary_stat_left["Saccade Length Kurtosis [px]"] = df_final_left_groups_sti["Saccade Length [px]"].apply(pd.DataFrame.kurt).values

print(df_summary_stat_left)

#### todo ###
##Pupil diameters 
##Vergence



