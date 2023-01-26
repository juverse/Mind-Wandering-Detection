####### data preparation Mind Wandering ######

### dependecies ###
import pandas as pd 
import numpy as np
import os
import vergence_raw_data as vergence

###
# Define paths

read_path = r"C:\Users\Julia\Desktop\Daten-BA\SMI_Eventstatistics"
# r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\02_BeGaze\SMI_Eventstatistics"
save_path = r"W:\WCT\04_Mind-Wandering-Labstudy\04_Daten\04_Prepared_data\00_Julia"

def read_data(filename: str) -> pd.DataFrame:
    """
    input: 
        filename as "Event Statistics - Single - 01.txt"
    output: 
        dataframe for one file, which is one participants
    """
    
    df_all = pd.read_csv(
        read_path + "\\" + filename, 
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
    df.to_csv(save_path + "\\" + eye +".csv", index = False)
    

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
    returns data which are under the blinkthreshold in a tupel for left and right eye and adds the pupil baseline
    
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
        if "vid" + str(i) in df_split.groups.keys():
            df_temp = pd.DataFrame(df_split.get_group("vid" + str(i)))
            df_concat = pd.concat([df_concat, cut_last_10sec_of_one_video(df_temp)]) 
    return df_concat

### summary statistics for one Eye ###

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
                                                                 "Event_duration_q25", "Event_duration_q75"]]


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
    df_groups_stimulus = df.groupby("Stimulus") 
    df_groups_stimulus_category = df.groupby(["Stimulus", "Category"]) # das benutze ich nur für event duration und blink duration
   
    df_summary_statistics = pd.DataFrame(index = ["vid1","vid10","vid11","vid12","vid13","vid14","vid15","vid2","vid3","vid4","vid5","vid6","vid7","vid8","vid9"])
    
    ## add general information
    df_summary_statistics["Tracking Ratio [%] Mean"] = df_groups_stimulus["Tracking Ratio [%]"].mean()
    df_summary_statistics["Stimulus"] = ["vid1","vid10","vid11","vid12","vid13","vid14","vid15","vid2","vid3","vid4","vid5","vid6","vid7","vid8","vid9"]
    df_summary_statistics["Participant"] = [df.loc[:,"Participant"].values[0]] * 15
    
    ## Features
    #statistics I want for all features
    statistics = ["mean", "max", "min", "median", "std", "skew", lambda x: x.quantile(0.25),lambda x: x.quantile(0.75)]
    ## Event Duration:
    df_statistics_event_duration = get_statistics_event_duration(df_groups_stimulus_category, statistics)
    
    ## Fixation Duration
    # add Fixation duration to dataframe

    df_summary_statistics[["Fixation Duration Mean [ms]", 
                            "Fixation Duration Max [ms]", 
                            "Fixation Duration Min [ms]",
                            "Fixation Duration Median [ms]",
                            "Fixation Duration Std [ms]",
                            "Fixation Duration Skew [ms]",
                            "Fixation Duration Quantil 25 [ms]",
                            "Fixation Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Fixation")
    
    ## Saccade Duration
    # add Saccade duration to dataframe
    df_summary_statistics[["Saccade Duration Mean [ms]", 
                            "Saccade Duration Max [ms]", 
                            "Saccade Duration Min [ms]",
                            "Saccade Duration Median [ms]",
                            "Saccade Duration Std [ms]",
                            "Saccade Duration Skew [ms]",
                            "Saccade Duration Quantil 25 [ms]",
                            "Saccade Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Saccade")

    ## Mean Duration of Blink
    #brauche ich hier die anderen-> max, min??
    df_summary_statistics[["Blink Duration Mean [ms]", 
                        "Blink Duration Max [ms]", 
                        "Blink Duration Min [ms]",
                        "Blink Duration Median [ms]",
                        "Blink Duration Std [ms]",
                        "Blink Duration Skew [ms]",
                        "Blink Duration Quantil 25 [ms]",
                        "Blink Duration Quantil 75 [ms]"]] = get_one_event_statistics_event_duration(df_statistics_event_duration, "Blink") #12 statt 15?!
    
    # kurtosis
    kurt = df_groups_stimulus_category["Event Duration [ms]"].apply(pd.DataFrame.kurt)
    kurt = kurt.reset_index()
   
    df_summary_statistics["Fixation Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Fixation",:].reset_index().set_index("Stimulus")["Event Duration [ms]"]
    df_summary_statistics["Saccade Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Saccade",:].reset_index().set_index("Stimulus")["Event Duration [ms]"]
    df_summary_statistics["Blink Duration Kurtosis [ms]"] = kurt.loc[kurt["Category"]=="Blink", :].reset_index().set_index("Stimulus")["Event Duration [ms]"]

    ## saccade fixation ratio
    df_summary_statistics["Fixation Saccade Ratio Mean"] = get_saccade_fixation_ratio(df_summary_statistics, "Fixation Duration Mean [ms]","Saccade Duration Mean [ms]")
    df_summary_statistics["Fixation Saccade Ratio Max"] = get_saccade_fixation_ratio(df_summary_statistics, "Fixation Duration Max [ms]", "Saccade Duration Max [ms]")
    df_summary_statistics["Fixation Saccade Ratio Min"] = get_saccade_fixation_ratio(df_summary_statistics,"Fixation Duration Min [ms]","Saccade Duration Min [ms]" )
    df_summary_statistics["Fixation Saccade Ratio Median"] = get_saccade_fixation_ratio(df_summary_statistics,"Fixation Duration Median [ms]","Saccade Duration Median [ms]" )
    df_summary_statistics["Fixation Saccade Ratio Std"] = get_saccade_fixation_ratio(df_summary_statistics,"Fixation Duration Std [ms]", "Saccade Duration Std [ms]")
    df_summary_statistics["Fixation Saccade Ratio Skew"] = get_saccade_fixation_ratio(df_summary_statistics,"Fixation Duration Skew [ms]","Saccade Duration Skew [ms]" )
    df_summary_statistics["Fixation Saccade Ratio Kurtosis"] = get_saccade_fixation_ratio(df_summary_statistics, "Fixation Duration Kurtosis [ms]","Saccade Duration Kurtosis [ms]")

    ## Fixation Number
    # take on fixation value and count how often it appears
    fix_count= df_groups_stimulus.agg({"Fixation Average Pupil Size X [px]": "count"})
    df_summary_statistics["Fixation Number"] = fix_count

    ## Blink Number
    blink_count = df_groups_stimulus_category.count()["Trial"][:, "Blink"]
    df_summary_statistics["Blink Number"] = blink_count

    ## Fixation Dispersion x und y
    fix_dis_x_y = df_groups_stimulus.agg({"Fixation Dispersion X [px]": statistics, "Fixation Dispersion Y [px]": statistics })

    fix_dis_x_y.columns = ['fix_dis_x_mean', "fix_dis_x_max", "fix_dis_x_min", "fix_dis_x_median", "fix_dis_x_std", "fix_dis_x_skew", "fix_dis_x_q25","fix_dis_x_q75",
    'fix_dis_y_mean', "fix_dis_y_max", "fix_dis_y_min", "fix_dis_y_median", "fix_dis_y_std", "fix_dis_y_skew", "fix_dis_y_q25", "fix_dis_y_q75"]

    df_summary_statistics[["Fixation Dispersion X Mean [px]", 
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

    df_summary_statistics["Fixation Dispersion X Kurtosis [px]"] = get_kurtosis(df_groups_stimulus, "Fixation Dispersion X [px]")
    df_summary_statistics["Fixation Dispersion Y Kurtosis [px]"] = get_kurtosis(df_groups_stimulus, "Fixation Dispersion Y [px]")

    ## Saccade Amplitude
    sacc_ampl = df_groups_stimulus.agg({"Saccade Amplitude [°]": statistics})

    sacc_ampl.columns = ['sacc_ampl_mean', "sacc_ampl_max", "sacc_ampl_min", "sacc_ampl_median", "sacc_ampl_std", "sacc_ampl_skew",
    "sacc_ampl_q25", "sacc_ampl_q75"]

    df_summary_statistics[["Saccade Amplitude Mean [°]", 
                            "Saccade Amplitude Max [°]", 
                            "Saccade Amplitude Min [°]",
                            "Saccade Amplitude Median [°]",
                            "Saccade Amplitude Std [°]",
                            "Saccade Amplitude Skew [°]",
                            "Saccade Amplitude Quantil 25 [°]",
                            "Saccade Amplitude Quantil 75 [°]"]] = sacc_ampl

    df_summary_statistics["Saccade Amplitude Kurtosis [°]"] = get_kurtosis(df_groups_stimulus, "Saccade Amplitude [°]")

    sacc_features = df_groups_stimulus.agg({'Saccade Acceleration Average [°/s²]': statistics,
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

    df_summary_statistics[["Saccade Acceleration Average [°/s²] Mean", 
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

    df_summary_statistics["Saccade Acceleration Average [°/s²] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Acceleration Average [°/s²]")
    df_summary_statistics["Saccade Acceleration Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Acceleration Peak [°/s²]")
    df_summary_statistics["Saccade Deceleration Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Deceleration Peak [°/s²]")
    df_summary_statistics["Saccade Velocity Average [°/s²] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Velocity Average [°/s]")
    df_summary_statistics["Saccade Velocity Peak [°/s²] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Velocity Peak [°/s]")
    df_summary_statistics["Saccade Velocity Peak [%] Kurtosis"] = get_kurtosis(df_groups_stimulus,"Saccade Peak Velocity at [%]")

    ## Saccade Length: Distance of saccade in pixels
    # distanz = sqrt((end_X - start_x)^2 + (end_y - start_y)^2)

    # calcualte distance and make new column in df
    x_dist = diff(df,"Saccade End Position X [px]", "Saccade Start Position X [px]" )
    y_dist = diff(df,"Saccade End Position Y [px]", "Saccade Start Position Y [px]" )

    distance = np.sqrt(x_dist + y_dist)
    df["Saccade Length [px]"] = distance

    # calcualte mean vor each video and so on
    sacc_length = df_groups_stimulus.agg({"Saccade Length [px]": statistics})

    sacc_length.columns = ['sacc_length_mean', "sacc_length_max", "sacc_length_min", "sacc_length_median", "sacc_length_std", 
    "sacc_length_skew", "sacc_length_q25", "sacc_length_q75"]

    df_summary_statistics[["Saccade Length Mean [px]", 
                            "Saccade Length Max [px]", 
                            "Saccade Length Min [px]",
                            "Saccade Length Median [px]",
                            "Saccade Length Std [px]",
                            "Saccade Length Skew [px]]",
                            "Saccade Length Quantil 25 [px]]",
                            "Saccade Length Quantil 75 [px]]"]] = sacc_length

    df_summary_statistics["Saccade Length Kurtosis [px]"] = get_kurtosis(df_groups_stimulus,"Saccade Length [px]")

    #pupil Diameters
    df_summary_statistics[["Fixation Average Pupil Diameter [mm] Mean",
    "Fixation Average Pupil Diameter [mm] Max",
    "Fixation Average Pupil Diameter [mm] Min",
    "Fixation Average Pupil Diameter [mm] Median",
    "Fixation Average Pupil Diameter [mm] Std",
    "Fixation Average Pupil Diameter [mm] Skew",
    "Fixation Average Pupil Diameter [mm] Quantil25",
    "Fixation Average Pupil Diameter [mm] Quantil75"]] = df_groups_stimulus.agg({"Fixation Average Pupil Diameter [mm] subtractive baseline correction": statistics})
    
    df_summary_statistics["Fixation Average Pupil Diameter [mm] Kurtosis"] = get_kurtosis(df_groups_stimulus, "Fixation Average Pupil Diameter [mm] subtractive baseline correction")
     
    return df_summary_statistics
 
def main():
   
    df_summary_statistic_left_eye_all_participants = pd.DataFrame()
    df_summary_statistic_right_eye_all_participants = pd.DataFrame()
  
    #navigate to directory
    os.chdir("C:/Users/Julia/Desktop/Daten-BA/SMI_Eventstatistics") 
    #get all files in directory
    list = os.listdir() 
    
    # remove excluded participants from validation Data and Erhebungsprotokoll
    excluded_participants_ids = [2, 8, 11, 12, 13, 24, 25, 27, 30, 31, 38, 41, 43, 47, 50, 52, 54, 59, 63, 69, 70, 75, 82, 84, 89, 91, 101, 102, 103, 104, 105, 999]
    excluded_participants_filenname = []
    for i in excluded_participants_ids:
        excluded_participants_filenname.append("Event Statistics - Single - " + str(i) +".txt")

    list_without_excluded = [x for x in list if x not in excluded_participants_filenname]
    list_without_excluded.remove('Event Statistics - Single - 02.txt')
    list_without_excluded.remove('Event Statistics - Single - 08.txt')

    for filename in list_without_excluded:
        print(filename)
        # input: filename # output: dataframe to work on
        df = read_data(filename)
        df_use =drop_unused_columns(df)
        
        # input: df_use # output: (df_left_eye, df_right_eye)
        df_left_eye =  df_prepare_eye_tracking_data(df_use)[0]
        df_right_eye = df_prepare_eye_tracking_data(df_use)[1]
        
        # input: dataframe for one eye e # output: necessary videomoments
        df_10_sec_of_videos_left = cut_last_10sec_of_videos(df_left_eye)
        df_10_sec_of_videos_right = cut_last_10sec_of_videos(df_right_eye)

        # input: last 10 seconds dataframe for one eye # output: df_summary statistics
        df_summary_statistics_left_eye = summary_statistics(df_10_sec_of_videos_left)
        df_summary_statistics_right_eye = summary_statistics(df_10_sec_of_videos_right)

        df_summary_statistic_left_eye_all_participants = pd.concat([df_summary_statistic_left_eye_all_participants, df_summary_statistics_left_eye])
        df_summary_statistic_right_eye_all_participants = pd.concat([df_summary_statistic_right_eye_all_participants, df_summary_statistics_right_eye])
    
    # exclusion with tracking ratio
    df_left_tracking_ratio_over70 = df_summary_statistic_left_eye_all_participants.loc[df_summary_statistic_left_eye_all_participants["Tracking Ratio [%] Mean"] >= 70.0]
    df_right_tracking_ratio_over70 = df_summary_statistic_right_eye_all_participants.loc[df_summary_statistic_right_eye_all_participants["Tracking Ratio [%] Mean"] >= 70.0]


    #vergence
    #vergence_all_participanats = vergence.all_angles()
    #df_summary_statistic_left_eye_all_participants.join(vergence_all_participanats, how='left', lsuffix='_event_statistics', rsuffix='_raw_data')


    write_data(pd.DataFrame(df_left_tracking_ratio_over70), "left_eye_without_excluded_participant_trackingratio_over70")
    write_data(pd.DataFrame(df_right_tracking_ratio_over70), "right_eye_without_excluded_participant_trackingratio_over70")

    return df_summary_statistic_left_eye_all_participants

if __name__ == "__main__":
    main()

