####### data preparation Mind Wandering ######

### dependecies ###
import pandas as pd 
import numpy as np

def read_data(participants_id: str) -> pd.DataFrame:
    """
    input: 
        participants_id
    output: 
        dataframe for participants_id 
    """

    df_all = pd.read_csv(
        r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_Eventstatistics\Event Statistics - Single - " + participants_id + ".txt", 
        na_values = "-"
    )

    return df_all


def drop_unused_columns(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    input:
        all data for one participant
    output:
        same data without used columns
    """

    df_initial = df_all.drop(
        columns = [
            "Trial Start Raw Time [ms]",  
            "Trial Start Time of Day [h:m:s:ms]", 
            "Export Start Trial Time [ms]",
            "Color", 
            "Category Group",
            "Event Start Raw Time [ms]",
            "Event End Raw Time [ms]", 
            "Annotation Name", 
            "Annotation Description", 
            "Annotation Tags",
            'Fixation Position X [px]', 
            'Fixation Position Y [px]', 
            "Mouse Position X [px]",
            "Mouse Position Y [px]", 
            "Scroll Direction X", 
            "Scroll Direction Y",
            "Port Status", 
            "Content", 
            "Index"
        ]
    ) #, inplace = True)
    return df_initial

### cut last 10 seconds of videos ###

# 1. df split in vidoes => more efficint
# 2. function for finding the last  10 seconds from each video (it is the MW time)
#      2.1 find the interval, which we are interested in [end of video-10s, end of vide]
#      2.2 which events are in this interval: if event start time > start_vid1 und event end time < end_vid1
# 3. perform this functoin for each video
# 4. bring all df for each video together

def get_video_end_time_ms(df_video: pd.DataFrame) -> int:
    """
    input:  ? videonumber and length of videos
    output:
        end time for this video in ms
    

    """
    pass


def get_event_end_time_of_last_event(df_video: pd.DataFrame) -> float:
    """
    input:
        video data (sorted by time) of one participant
    output:
        start time of last event for video in ms
    """

    event_start_time_of_last_event = df_video.loc[:, "Event End Trial Time [ms]"].max()
    return event_start_time_of_last_event

def cut_last_10sec_of_one_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the data of the last 10s of a video (starting by the last event, which is in the video)

    input:
        data (sorted by time) for one video of one participant
    output:
        data of the last 10sec of the input data
    """

    # number of last seconds
    video_end_time_ms = get_event_end_time_of_last_event(df)
    video_10sec_before_end_time_ms = video_end_time_ms - 10_000 # -10_000ms

    #get df last seconds: 
    #df_vid_final = df.loc[(df_event_start_trial > video_10sec_before_end_time_ms) & (df_event_end_trial <= video_end_time_ms), :]
    df_vid_final = df.loc[(df["Event Start Trial Time [ms]"] > video_10sec_before_end_time_ms) & (df["Event End Trial Time [ms]"] <= video_end_time_ms), :]

    return df_vid_final


# input: data_initial 
# output: necessary video moments
def cut_last_10sec_of_videos(df):
    df_split = df.groupby("Stimulus")
    ## necessary data together
    df_concat = pd.DataFrame()
    for i in range(1,16):
        df_temp = pd.DataFrame(df_split.get_group("vid" + str(i)))
        df_concat = pd.concat([df_concat, cut_last_10sec_of_one_video(df_temp)]) 
    return df_concat

### prepare eye tracking data ###

## blink duration:
# 1. 150 - 250 mseconds  Stern, J. A., Skelly, J.J. 1984. The eyeblink and workload considerations. Proceedings of the
#human factors society, 28th Annual meeting Santa Monica: Human Factors Society. 

# 2. 100 - 400 mseconds 
# Schiffman, H.R., Sensation and Perception. An Integrated Approach, New York: John Wiley and Sons, Inc., 2001

# => threshold:  500 mseconds  ?

# input: df_10_sec_of_videos
# output: df_left_eye
def df_prepare_eye_tracking_data(df):
    blink_duration = df.loc[df["Category"] == "Blink", :]
    blink_threshold = 500

    # df without too long blinks
    df_concat_blink = df.loc[(df["Category"] != "Blink") |
                                    ((df["Category"] == "Blink") & (df["Event Duration [ms]"] < blink_threshold))]

    ## split data in right and left eye => just take one eye for statistics
    df_final_left = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Left", :]
    df_final_right = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Right", :]

    return (df_final_left, df_final_right)

### summary statistics for Left Eye ###
 #statitsic kurtosis
def get_kurtosis(df_groups_st, feature):
    return df_groups_st[feature].apply(pd.DataFrame.kurt).values

# input: df grouped in stimuli and category
def get_statistics_event_duration(df_groups_stimuli_category, statistics):
    # # 1.step: calcaulte the statistics 
    df_event_duration = df_groups_stimuli_category.agg({"Event Duration [ms]": statistics})

    # 2. to get dataframe after aggretaion with two groups
    df_event_duration.columns = ['Event_duration_mean', "Event_duration_max", "Event_duration_min", "Event_duration_median", "Event_duration_std", "Event_duration_skew",
    "Event_duration_q25", "Event_duration_q75"]
    df_event_duration = df_event_duration.reset_index()
    return df_event_duration

# input: df grouped in stimuli and category with statiistics, event we want to see
# output: df with statistics for event duration
def get_one_event_statistics_event_duration(df_statistics_event_duration, event):
    return df_statistics_event_duration.loc[df_statistics_event_duration["Category"] == event, ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]].values
 # index matching causes Nan values, we havt to drop out the indexes, so we can add the values or we just take the values!

## saccade fixation ratio
#Das Verhältnis zweier Zahlen zeigt, wie viel Mal die erste Zahl größer ist
#als die zweite oder welchen Anteil die erste Zahl von der zweiten ausmacht.
def get_sacc_fix_ratio(df_summary_stat, fix, sacc):
    return df_summary_stat[fix]/df_summary_stat[sacc]

# difference between two points
def diff(df,start,end):
        return (df[end].astype(float) - df[start].astype(float))**2 

# input: prepared dataframe for one eye
# output: df_summary statistics
def summary_statistics(df):
    #groupby left for the videos
    df_groups_sti = df.groupby("Stimulus") 
    df_groups_sti_cat = df.groupby(["Stimulus", "Category"])
    # summary statistics
    df_summary_stat = pd.DataFrame()
    
    ## add general information
    df_summary_stat["Tracking Ratio [%] Mean"] = df_groups_sti["Tracking Ratio [%]"].mean()
    df_summary_stat["Stimulus"] = ['vid1','vid10','vid11','vid12','vid13','vid14','vid15','vid2','vid3','vid4','vid5','vid6','vid7','vid8','vid9']
    df_summary_stat["Participant"] = [df.loc[:,"Participant"].values[0]] * 15
    
    ## Features
    #statistics I want for all features
    statistics = ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25),lambda x: x.quantile(0.75)]
    ## Event Duration:
    df_statistics_event_duration = get_statistics_event_duration(df_groups_sti_cat, statistics)
    
    ## Fixation Duration
    # add Fixation duration to dataframe
    df_summary_stat[["Fixation Duration Mean [ms]", 
                            "Fixation Duration Max [ms]", 
                            "Fixation Duration Min [ms]",
                            "Fixation Duration Median [ms]",
                            "Fixation Duration Std [ms]",
                            "Fixation Duration Skew [ms]",
                            "Fixation Duration Quantil 25 [ms]",
                            "Fixation Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Fixation")
    
    ## Saccade Duration
    # add Saccade duration to dataframe
    df_summary_stat[["Saccade Duration Mean [ms]", 
                            "Saccade Duration Max [ms]", 
                            "Saccade Duration Min [ms]",
                            "Saccade Duration Median [ms]",
                            "Saccade Duration Std [ms]",
                            "Saccade Duration Skew [ms]",
                            "Saccade Duration Quantil 25 [ms]",
                            "Saccade Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Saccade")

    #print(get_one_event_statistics_event_duration(df_statistics_event_duration, "Blink"))
    ## Mean Duration of Blink
    #brauche ich hier die anderen-> max, min??
    df_summary_stat[["Blink Duration Mean [ms]", 
                        "Blink Duration Max [ms]", 
                        "Blink Duration Min [ms]",
                        "Blink Duration Median [ms]",
                        "Blink Duration Std [ms]",
                        "Blink Duration Skew [ms]",
                        "Blink Duration Quantil 25 [ms]",
                        "Blink Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Blink") #12 statt 15?!
    
    # kurtosis
    kurt = df_groups_sti_cat["Event Duration [ms]"].apply(pd.DataFrame.kurt)
    kurt = kurt.reset_index()
    # add to dataframe 
    df_summary_stat["Fixation Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Fixation", "Event Duration [ms]"].values
    df_summary_stat["Saccade Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Saccade", "Event Duration [ms]"].values
    df_summary_stat["Blink Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Blink", "Event Duration [ms]"].values
    #values otherwise we would add NaN, because of the indexes

    ## saccade fixation ratio
    df_summary_stat["Fixation Saccade Ratio Mean"] = get_sacc_fix_ratio(df_summary_stat, "Fixation Duration Mean [ms]","Saccade Duration Mean [ms]")
    df_summary_stat["Fixation Saccade Ratio Max"] = get_sacc_fix_ratio(df_summary_stat, "Fixation Duration Max [ms]", "Saccade Duration Max [ms]")
    df_summary_stat["Fixation Saccade Ratio Min"] = get_sacc_fix_ratio(df_summary_stat,"Fixation Duration Min [ms]","Saccade Duration Min [ms]" )
    df_summary_stat["Fixation Saccade Ratio Median"] = get_sacc_fix_ratio(df_summary_stat,"Fixation Duration Median [ms]","Saccade Duration Median [ms]" )
    df_summary_stat["Fixation Saccade Ratio Std"] = get_sacc_fix_ratio(df_summary_stat,"Fixation Duration Std [ms]", "Saccade Duration Std [ms]")
    df_summary_stat["Fixation Saccade Ratio Skew"] = get_sacc_fix_ratio(df_summary_stat,"Fixation Duration Skew [ms]","Saccade Duration Skew [ms]" )
    df_summary_stat["Fixation Saccade Ratio Kurtosis"] = get_sacc_fix_ratio(df_summary_stat, "Fixation Duration Kurtosis [ms]","Saccade Duration Kurtosis [ms]") #.values

    ## Fixation Number
    # take on fixation value and count how often it appears
    fix_count= df_groups_sti.agg({"Fixation Average Pupil Size X [px]": "count"})
    df_summary_stat["Fixation Number"] = fix_count

    ## Blink Number
    blink_count = df_groups_sti_cat.count()["Trial"][:, "Blink"]
    df_summary_stat["Blink Number"] = blink_count

    ## Fixation Dispersion x und y
    fix_dis_x_y = df_groups_sti.agg({"Fixation Dispersion X [px]": statistics, "Fixation Dispersion Y [px]": statistics })

    fix_dis_x_y.columns = ['fix_dis_x_mean', "fix_dis_x_max", "fix_dis_x_min", "fix_dis_x_median", "fix_dis_x_std", "fix_dis_x_skew", "fix_dis_x_q25","fix_dis_x_q75",
    'fix_dis_y_mean', "fix_dis_y_max", "fix_dis_y_min", "fix_dis_y_median", "fix_dis_y_std", "fix_dis_y_skew", "fix_dis_y_q25", "fix_dis_y_q75"]

    df_summary_stat[["Fixation Dispersion X Mean [px]", 
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

    df_summary_stat["Fixation Dispersion X Kurtosis [px]"] = get_kurtosis(df_groups_sti, "Fixation Dispersion X [px]")
    df_summary_stat["Fixation Dispersion Y Kurtosis [px]"] = get_kurtosis(df_groups_sti, "Fixation Dispersion Y [px]")

    ## Saccade Amplitude
    sacc_ampl = df_groups_sti.agg({"Saccade Amplitude [°]": statistics})

    sacc_ampl.columns = ['sacc_ampl_mean', "sacc_ampl_max", "sacc_ampl_min", "sacc_ampl_median", "sacc_ampl_std", "sacc_ampl_skew",
    "sacc_ampl_q25", "sacc_ampl_q75"]
    #sacc_ampl = sacc_ampl.reset_index()

    df_summary_stat[["Saccade Amplitude Mean [°]", 
                            "Saccade Amplitude Max [°]", 
                            "Saccade Amplitude Min [°]",
                            "Saccade Amplitude Median [°]",
                            "Saccade Amplitude Std [°]",
                            "Saccade Amplitude Skew [°]",
                            "Saccade Amplitude Quantil 25 [°]",
                            "Saccade Amplitude Quantil 75 [°]"]] = sacc_ampl

    df_summary_stat["Saccade Amplitude Kurtosis [°]"] = get_kurtosis(df_groups_sti, "Saccade Amplitude [°]")

    sacc_features = df_groups_sti.agg({'Saccade Acceleration Average [°/s²]': statistics,
        'Saccade Acceleration Peak [°/s²]': statistics, 'Saccade Deceleration Peak [°/s²]': statistics, 
        'Saccade Velocity Average [°/s]': statistics, 'Saccade Velocity Peak [°/s]': statistics,  
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

    df_summary_stat[["Saccade Acceleration Average [°/s²] Mean", 
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

    df_summary_stat["Saccade Acceleration Average [°/s²] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Acceleration Average [°/s²]")
    df_summary_stat["Saccade Acceleration Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Acceleration Peak [°/s²]")
    df_summary_stat["Saccade Deceleration Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Deceleration Peak [°/s²]")
    df_summary_stat["Saccade Velocity Average [°/s²] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Velocity Average [°/s]")
    df_summary_stat["Saccade Velocity Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Velocity Peak [°/s]")
    df_summary_stat["Saccade Velocity Peak [%] Kurtosis"] = get_kurtosis(df_groups_sti,"Saccade Peak Velocity at [%]")

    ## Saccade Length: Distance of saccade in pixels
    # distanz = sqrt((end_X - start_x)^2 + (end_y - start_y)^2)

    # 1. calcualte distance and make new column in df
    x_dist = diff(df,"Saccade End Position X [px]", "Saccade Start Position X [px]" )
    y_dist = diff(df,"Saccade End Position Y [px]", "Saccade Start Position Y [px]" )

    distance = np.sqrt(x_dist + y_dist)
    df["Saccade Length [px]"] = distance

    # 2 calcualte mean vor each video and so on
    sacc_length = df_groups_sti.agg({"Saccade Length [px]": statistics})

    sacc_length.columns = ['sacc_length_mean', "sacc_length_max", "sacc_length_min", "sacc_length_median", "sacc_length_std", 
    "sacc_length_skew", "sacc_length_q25", "sacc_length_q75"]
    #sacc_ampl = sacc_ampl.reset_index()

    df_summary_stat[["Saccade Length Mean [px]", 
                            "Saccade Length Max [px]", 
                            "Saccade Length Min [px]",
                            "Saccade Length Median [px]",
                            "Saccade Length Std [px]",
                            "Saccade Length Skew [px]]",
                            "Saccade Length Quantil 25 [px]]",
                            "Saccade Length Quantil 75 [px]]"]] = sacc_length

    df_summary_stat["Saccade Length Kurtosis [px]"] = get_kurtosis(df_groups_sti,"Saccade Length [px]")

    return df_summary_stat
    ##Pupil diameters 

    # pupil size with emotions
    # we need a baseline, where the Person is neutral, before experiment
    # I can use everything before video1 from raw data
    # standardization wie value/devided by mean from before
    # than we can calculate sd, mean, etc. 
    # get neutral pupil size before video1 starts as a baseline for pupilsize

    # # for all VP different!!!!,have to change the data
    # df_raw_data = pd.read_csv(r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_RawData\Raw Data - Raw Data - 01.txt"
    # , na_values = "-", usecols= ["Stimulus", "Pupil Diameter Left [mm]", "Pupil Diameter Right [mm]"])

    # df_welcome_to_instruction = df_raw_data.loc[(df_raw_data["Stimulus"] == "welcome") | (df_raw_data["Stimulus"] == "transition1") 
    # | (df_raw_data["Stimulus"] == 'https://www.unipark.de/uc/Mind-Wandering/510e') 
    # | (df_raw_data["Stimulus"] == "https://www.unipark.de/uc/Mind-Wandering/510e/ospe.php?qb")
    # | (df_raw_data["Stimulus"] == "instruction")
    # ]

    # pupil_diameter_right_mean = df_welcome_to_instruction["Pupil Diameter Right [mm]"].mean()
    # pupil_diameter_left_mean = df_welcome_to_instruction["Pupil Diameter Left [mm]"].mean()

    # #Pupil diameters 
    # #for each participant different pupils, but we want to compare them
    # # that why we need a way to standaralize them, devide them by a baseline, which we get from data before watching the video
    # #import features_pupil_diameter

    # # divide all pupil diameters for fixation in the last 10 seconds with pupil mean
    # df["Fixation Average Pupil Diameter [mm] standardized"] = df.loc[:, "Fixation Average Pupil Diameter [mm]"]/pupil_diameter_left_mean #.std()

    # # summary statistics
    # df_summary_stat[["Fixation Average Pupil Diameter [mm] Mean",
    # "Fixation Average Pupil Diameter [mm] Max",
    # "Fixation Average Pupil Diameter [mm] Min",
    # "Fixation Average Pupil Diameter [mm] Median",
    # "Fixation Average Pupil Diameter [mm] Std",
    # "Fixation Average Pupil Diameter [mm] Skew",
    # "Fixation Average Pupil Diameter [mm] Quantil25",
    # "Fixation Average Pupil Diameter [mm] Quantil75"]] = df_groups_sti.agg({"Fixation Average Pupil Diameter [mm] standardized":
    # statistics})

    # df_summary_stat["Fixation Average Pupil Diameter [mm] Kurtosis"] = get_kurtosis(df_groups_sti, "Fixation Average Pupil Diameter [mm] standardized")

#### todo ###
##Vergence


### write data
def write_data(df, eye):
    df.to_csv('W:WCT/04_Mind-Wandering-Labstudy/04_Daten/04_Prepared_data/00_Julia/summary_statistics'+eye+".txt", index = False, sep=',')

def main():
    # irgendwas an dieser for shcleife ist falsch
    #df = pd.DataFrame()
    df_summary_statistic_left_eye_all_participants = pd.DataFrame()
    df_summary_statistic_right_eye_all_participants = pd.DataFrame()

    list = [f'{i:>02}' for i in range(2, 3)]
    for participants_id in list:
        # input: participants_id # output: dataframe to work on
        df = read_data(participants_id)
        df_use =drop_unused_columns(df)

        # input: data_initial # output: necessary videomoments
        df_10_sec_of_videos = cut_last_10sec_of_videos(df_use)

        # input: df_10_sec_of_videos # output: (df_left_eye, df_right_eye)
        df_left_eye =  df_prepare_eye_tracking_data(df_10_sec_of_videos)[0]
        df_right_eye = df_prepare_eye_tracking_data(df_10_sec_of_videos)[1]
        
        # input: prepared dataframe for one eye # output: df_summary statistics
        df_summary_statistics_left_eye = summary_statistics(df_left_eye)
        #df_summary_statistics_right_eye = summary_statistics(df_right_eye)

        df_summary_statistic_left_eye_all_participants = pd.concat([df_summary_statistic_left_eye_all_participants, df_summary_statistics_left_eye])
        #df_summary_statistic_right_eye_all_participants = pd.concat([df_summary_statistic_right_eye_all_participants, df_summary_statistics_right_eye])
        
        print(df_summary_statistics_left_eye)
        print(type(pd.DataFrame(df_summary_statistics_left_eye)))
        print(pd.DataFrame(df_summary_statistics_left_eye))
    
    write_data(pd.DataFrame(df_summary_statistics_left_eye), "_left_eye_all_participants")
    #write_data(summary_statistics_left_eye, "right_eye")
    return df_summary_statistic_left_eye_all_participants

if __name__ == "__main__":
    main()

# file pro Thema, namespace