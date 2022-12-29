####### data preparation Mind Wandering ######

### dependecies ###
import pandas as pd 
import numpy as np
import os

###
# Define paths
#read_path = "C:\Users\Julia\Desktop\Daten-BA\SMI_Eventstatistics"
#read_path = r'W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\01_Raw_Data'
# save_path = r'W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\04_Prepared_data\01_MW_Probes'
# excluded_path = r'W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_Excluded_ids.csv'

 # ['Event Statistics - Single - 01.txt',
def read_data(filename: str) -> pd.DataFrame:
    """
    input: 
        filename as "Event Statistics - Single - 01.txt"
    output: 
        dataframe for one file, which is one participants
    """

    # df_all = pd.read_csv(
    #     r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_Eventstatistics\Event Statistics - Single - " + participants_id + ".txt", 
    #     na_values = "-"
    # )
    
    df_all = pd.read_csv(
        r"C:\Users\Julia\Desktop\Daten-BA\SMI_Eventstatistics" + "\\" + filename, 
        #r""read_path+ "\\" + filename, 
        na_values = "-"
    )
    return df_all

def write_data(df, eye):
    """
    saves the data in a file

    input: 
        final summary statiscs for one eye
        left or right eye
    """
    df.to_csv('W:WCT/04_Mind-Wandering-Labstudy/04_Daten/04_Prepared_data/00_Julia/summary_statistics'+eye+".txt", index = False, sep=',')

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
    ) 
    return df_initial
### prepare eye tracking data ###

## blink duration:
# 1. 150 - 250 mseconds  Stern, J. A., Skelly, J.J. 1984. The eyeblink and workload considerations. Proceedings of the
#human factors society, 28th Annual meeting Santa Monica: Human Factors Society. 

# 2. 100 - 400 mseconds 
# Schiffman, H.R., Sensation and Perception. An Integrated Approach, New York: John Wiley and Sons, Inc., 2001

# => threshold:  500 mseconds  ?

def pupil_size_baseline(df: pd.DataFrame) -> float:
    """
    input: initial data
    output: baseline for pupil correction
    """
    df_baseline_data = df.loc[(df["Stimulus"] == "welcome"),:] 
    return df_baseline_data["Fixation Average Pupil Diameter [mm]"].mean()

def substractive_baseline_correction(baseline: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    input: basline of pupil size
    output: dataframe with new column
    """
    df["Fixation Average Pupil Diameter [mm] subtractive baseline correction"] = df.loc[:, "Fixation Average Pupil Diameter [mm]"] - baseline
    return df

def df_prepare_eye_tracking_data(df: pd.DataFrame):
    """
    returns data which are under the blinkthreshold in a tupel for left and right eye
    
    input:
        data initial

    output:
        data for left and right eye under a threshold and with pupilsize baseline correction
    """
    #add pupil baseline
    pupilsize_baseline = pupil_size_baseline(df) 
    df_substractive_baseline_correction = substractive_baseline_correction(pupilsize_baseline, df)

    # df without too long blinks
    blink_threshold = 500
    
    df_concat_blink = df_substractive_baseline_correction.loc[(df_substractive_baseline_correction["Category"] != "Blink") |
                                    ((df_substractive_baseline_correction["Category"] == "Blink") & (df_substractive_baseline_correction["Event Duration [ms]"] < blink_threshold))]
    print(df_concat_blink)
    ## split data in right and left eye => just take one eye for statistics
    df_final_left = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Left", :]
    df_final_right = df_concat_blink.loc[df_concat_blink["Eye L/R"] == "Right", :]

    return (df_final_left, df_final_right)


### cut last 10 seconds of videos ###

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


def cut_last_10sec_of_videos(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the data of the last 10s of all videos (starting by the last event, which is in the video)

    input: 
        data (sorted by time) for of one participant
    output: 
        data of the last 10sec of the input data

    """
    df_split = df.groupby("Stimulus")
    ## necessary data together
    df_concat = pd.DataFrame()
    for i in range(1,16):
        # hier existiert einmal ein video nicht,, vp 13
        # if abfrage ob das vid existiert
        if "vid" + str(i) in df_split.groups.keys():
            df_temp = pd.DataFrame(df_split.get_group("vid" + str(i)))
            df_concat = pd.concat([df_concat, cut_last_10sec_of_one_video(df_temp)]) 
    return df_concat

### summary statistics for one Eye ###

# probleme bei file 82
def get_kurtosis(df_groups_stimulus: pd.DataFrame, feature: str):
    """
    returns kurtosis
    
    input:
        data groupey by stimulus

    output:
        array of kurtosis
    """
    
    return df_groups_stimulus[feature].apply(pd.DataFrame.kurt) #.values

def get_statistics_event_duration(df_groups_stimuli_category, statistics):
    """
    returns statistics for all events
    
    input:
        df grouped in stimuli and category

    output:
        statistics for event duration
    """
    # calculate statistics 
    df_event_duration = df_groups_stimuli_category.agg({"Event Duration [ms]": statistics})

    # set index as final dataframe
    df_event_duration.columns = ['Event_duration_mean', "Event_duration_max", "Event_duration_min", "Event_duration_median", "Event_duration_std", "Event_duration_skew",
    "Event_duration_q25", "Event_duration_q75"]
    df_event_duration = df_event_duration.reset_index().set_index("Stimulus") 

    return df_event_duration


def get_one_event_statistics_event_duration(df_statistics_event_duration: pd.DataFrame, event: str) -> pd.DataFrame:
    """
    returns statistics for all events
    
    input:
        data grouped in stimuli and category with statiistics,
        event we want to see

    output:
        data with statistics for one event of Saccade, Blink or Fixation
    """
    return df_statistics_event_duration.loc[df_statistics_event_duration["Category"] == event, ["Event_duration_mean", "Event_duration_max", "Event_duration_min",
                                                                 "Event_duration_median","Event_duration_std", "Event_duration_skew",
                                                                 "Event_duration_q25", "Event_duration_q75"]]#.values
 # index matching causes Nan values, we havt to drop out the indexes, so we can add the values or we just take the values!


def get_saccade_fixation_ratio(df_summary_stat: pd.DataFrame, fixation: str, saccade: str) -> float:
    """
    returns ration between saccade and fixation, 
    Das Verhältnis zweier Zahlen zeigt, wie viel Mal die erste Zahl größer ist
    als die zweite oder welchen Anteil die erste Zahl von der zweiten ausmacht.
    
    input:
        data of statistic summary
        names of the colum we want to have the ratio

    output:
        data with statistics for one event of Saccade, Blink or Fixation
    """
    return df_summary_stat[fixation]/df_summary_stat[saccade]


def diff(df,start,end):
    """
    return the difference between two points in a dataframe

    input: 
        data, start and end column

    output:
        squared differences 
    """
    return (df[end].astype(float) - df[start].astype(float))**2 

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    return the features and statistics of the input data for on eye

    input: 
        prepared data

    output:
        statistic summary
    """
    #groupby left for the videos
    df_groups_sti = df.groupby("Stimulus") 
    df_groups_sti_cat = df.groupby(["Stimulus", "Category"]) # das benutze ich nur für event duration und blink duration
    # summary statistics
    df_summary_stat = pd.DataFrame(index = ["vid1","vid10","vid11","vid12","vid13","vid14","vid15","vid2","vid3","vid4","vid5","vid6","vid7","vid8","vid9"])
    
    ## add general information
    # problem ist hier, ich brauche df_summary_stat mit 15 videos, da bei tracking ratio es rausfällt
    df_summary_stat["Tracking Ratio [%] Mean"] = df_groups_sti["Tracking Ratio [%]"].mean()#.reset_index() finde einen weg wie ich den Index verändern kann, entweder zu vid oder zu nummern 
    df_summary_stat["Stimulus"] = ["vid1","vid10","vid11","vid12","vid13","vid14","vid15","vid2","vid3","vid4","vid5","vid6","vid7","vid8","vid9"]
    df_summary_stat["Participant"] = [df.loc[:,"Participant"].values[0]] * 15
    
    ## Features
    #statistics I want for all features
    statistics = ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25),lambda x: x.quantile(0.75)]
    ## Event Duration:
    df_statistics_event_duration = get_statistics_event_duration(df_groups_sti_cat, statistics)
    
    ## Fixation Duration
    # add Fixation duration to dataframe
    df_fixation_duration = get_one_event_statistics_event_duration(df_statistics_event_duration, "Fixation").reset_index() #array, no df pr panda series(only for 1dimensional)
    #df_summary_stat = pd.concat([df_summary_stat, df_fixation_duration], axis=1)
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
    #df_summary_stat["Fixation Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Fixation", "Event Duration [ms]"].values

    #df_summary_stat["Saccade Duration Kurtosis [ms]"] = pd.Series(kurt.loc[kurt["Category"]=="Saccade", ["Stimulus","Event Duration [ms]"]].values)
    #df_summary_stat["Blink Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Blink", "Event Duration [ms]"].values
    #values otherwise we would add NaN, because of the indexes

    ## saccade fixation ratio
    df_summary_stat["Fixation Saccade Ratio Mean"] = get_saccade_fixation_ratio(df_summary_stat, "Fixation Duration Mean [ms]","Saccade Duration Mean [ms]")
    df_summary_stat["Fixation Saccade Ratio Max"] = get_saccade_fixation_ratio(df_summary_stat, "Fixation Duration Max [ms]", "Saccade Duration Max [ms]")
    df_summary_stat["Fixation Saccade Ratio Min"] = get_saccade_fixation_ratio(df_summary_stat,"Fixation Duration Min [ms]","Saccade Duration Min [ms]" )
    df_summary_stat["Fixation Saccade Ratio Median"] = get_saccade_fixation_ratio(df_summary_stat,"Fixation Duration Median [ms]","Saccade Duration Median [ms]" )
    df_summary_stat["Fixation Saccade Ratio Std"] = get_saccade_fixation_ratio(df_summary_stat,"Fixation Duration Std [ms]", "Saccade Duration Std [ms]")
    df_summary_stat["Fixation Saccade Ratio Skew"] = get_saccade_fixation_ratio(df_summary_stat,"Fixation Duration Skew [ms]","Saccade Duration Skew [ms]" )
    #df_summary_stat["Fixation Saccade Ratio Kurtosis"] = get_saccade_fixation_ratio(df_summary_stat, "Fixation Duration Kurtosis [ms]","Saccade Duration Kurtosis [ms]") #.values

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

    # here längen mismatch 14 statt 15
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

    #pupil Diameters
    df_summary_stat[["Fixation Average Pupil Diameter [mm] Mean",
    "Fixation Average Pupil Diameter [mm] Max",
    "Fixation Average Pupil Diameter [mm] Min",
    "Fixation Average Pupil Diameter [mm] Median",
    "Fixation Average Pupil Diameter [mm] Std",
    "Fixation Average Pupil Diameter [mm] Skew",
    "Fixation Average Pupil Diameter [mm] Quantil25",
    "Fixation Average Pupil Diameter [mm] Quantil75"]] = df_groups_sti.agg({"Fixation Average Pupil Diameter [mm] subtractive baseline correction": statistics})
    
    df_summary_stat["Fixation Average Pupil Diameter [mm] Kurtosis"] = get_kurtosis(df_groups_sti, "Fixation Average Pupil Diameter [mm] subtractive baseline correction")
    
    #### todo ###
    ##Vergence
    return df_summary_stat
 
def main():
   
    df_summary_statistic_left_eye_all_participants = pd.DataFrame()
    df_summary_statistic_right_eye_all_participants = pd.DataFrame()
  
    #list = [f'{i:>02}' for i in [1,3,4,5,6,7,9,10]]
    list = [f'{i:>02}' for i in [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,48, 51, 55,57,58,60, 62,64,65,66,67,69,72, 73,74,75,76,77,78,79,80,81, 84,85,86,88,90,91,92,95]]

    #navigate to directory
    os.chdir("C:/Users/Julia/Desktop/Daten-BA/SMI_Eventstatistics") 
    #get all files in directory
    list = os.listdir() 
    # remove participants
    list.remove("Event Statistics - Single - 82.txt")
    for filename in list:
        print(filename)
        # input: filename # output: dataframe to work on
        df = read_data(filename)
        df_use =drop_unused_columns(df)
        
        # input: df_10_sec_of_videos # output: (df_left_eye, df_right_eye)
        df_left_eye =  df_prepare_eye_tracking_data(df_use)[0]
        df_right_eye = df_prepare_eye_tracking_data(df_use)[1]
        
        # input: data_initial # output: necessary videomoments
        df_10_sec_of_videos_left = cut_last_10sec_of_videos(df_left_eye)
        df_10_sec_of_videos_right = cut_last_10sec_of_videos(df_right_eye)

        # input: prepared dataframe for one eye # output: df_summary statistics
        df_summary_statistics_left_eye = summary_statistics(df_10_sec_of_videos_left)
        #df_summary_statistics_right_eye = summary_statistics(df_10_sec_of_videos_right)

        df_summary_statistic_left_eye_all_participants = pd.concat([df_summary_statistic_left_eye_all_participants, df_summary_statistics_left_eye])
        #df_summary_statistic_right_eye_all_participants = pd.concat([df_summary_statistic_right_eye_all_participants, df_summary_statistics_right_eye])
    
    #write_data(pd.DataFrame(df_summary_statistics_left_eye), "_left_eye_all_participants")
    write_data(pd.DataFrame(df_summary_statistic_left_eye_all_participants), "_left_eye_all_participants")

    #write_data(summary_statistics_left_eye, "_right_eye_all_participants")
    return df_summary_statistic_left_eye_all_participants

if __name__ == "__main__":
    main()

