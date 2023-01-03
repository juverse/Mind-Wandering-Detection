### vergence

## dependnecies
import data_preparation as dp
import pandas as pd
import numpy as np
import os

def read_data(filename: str) -> pd.DataFrame:
    """
    input: 
        filename as "Event Statistics - Single - 01.txt"
    output: 
        dataframe for one file, which is one participants
    """

    df_all = pd.read_csv(
        r"C:\Users\Julia\Desktop\Daten-BA\SMI_RawData" + "\\" + filename, 
        na_values = "-"
    )
    return df_all


# get 10 last seconds of one video
### cut last 10 seconds of videos ###

def cut_last_10sec_of_one_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the data of the last 10s of a video (starting by the last event, which is in the video)

    input:
        data (sorted by time) for one video of one participant
    output:
        data of the last 10sec of the input data
    """

    # number of last seconds
    video_end_time_ms = df.loc[:, "RecordingTime [ms]"].max()
    video_10sec_before_end_time_ms = video_end_time_ms - 10_000 # -10_000ms

    #get df last seconds: 
    #df_vid_final = df.loc[(df_event_start_trial > video_10sec_before_end_time_ms) & (df_event_end_trial <= video_end_time_ms), :]
    df_vid_final = df.loc[(df["RecordingTime [ms]"] > video_10sec_before_end_time_ms) & (df["RecordingTime [ms]"] <= video_end_time_ms), :]

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
        # hier existiert einmal ein video nicht, vp 13
        # if abfrage ob das vid existiert
        if "vid" + str(i) in df_split.groups.keys():
            df_temp = pd.DataFrame(df_split.get_group("vid" + str(i)))
            df_concat = pd.concat([df_concat, cut_last_10sec_of_one_video(df_temp)]) 
    return df_concat

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) #arcos um winkel zu bekommen


# veregnce angles
# 10 seconds of videos
# gaze vectors
# calcualte angle between gaze vector left and right

def vergence_angles(df):
    gaze_vektor_left = df.loc[:, ["Gaze Vector Left X", "Gaze Vector Left Y","Gaze Vector Left Z"]].values
    gaze_vektor_right = df.loc[:, ["Gaze Vector Right X", "Gaze Vector Right Y","Gaze Vector Right Z"]].values
    result = []
    for vector in range(len(gaze_vektor_left)):
        vec1 = gaze_vektor_right[vector]
        vec2 = gaze_vektor_left[vector]
        result.append(angle_between(unit_vector(vec1), unit_vector(vec2)))
    return result

# pupil distance
# abstand zwischen linker und rechter fixation
# fixationspunkt der Augen

#'Pupil Position Right X [px]', 'Pupil Position Right Y [px]',
#      'Pupil Position Left X [px]', 'Pupil Position Left Y [px]',
def vergence_distance(df):
    x_right = df["Pupil Position Right X [px]"]
    y_right = df["Pupil Position Right Y [px]"]

    x_left = df["Pupil Position Left X [px]"]
    y_left = df["Pupil Position Left Y [px]"]

    distance = np.sqrt((x_right-x_left)**2 + (y_right-y_left)**2)
    return distance

def summary_statistics(df, vergence_angles, vergence_distance):
    df["Veregence Angles [rad]"] = vergence_angles
    df["Pupil Distance [px]"] = vergence_distance
    df_groups_sti = df.groupby("Stimulus") 
    statistics = ["mean", "std"]

    df_summary_statistics = pd.DataFrame(index = ["vid1","vid10","vid11","vid12","vid13","vid14","vid15","vid2","vid3","vid4","vid5","vid6","vid7","vid8","vid9"])
    df_summary_statistics["Participant"] = [df.loc[:,"Participant"].values[0]] * 15

    vergence = df_groups_sti.agg({"Veregence Angles [rad]": statistics})
    vergence.columns = ["vergence_mean", "vergence_std"]
    df_summary_statistics[["Veregence Angles Mean [rad]", "Veregence Angles Std [rad]"]] = vergence

    distance = df_groups_sti.agg({"Pupil Distance [px]": statistics})
    distance.columns = ["distance_mean", "distance_std"]
    df_summary_statistics[["Pupil Distance Mean [px]","Pupil Distance Std [px]"]] = distance
    return df_summary_statistics 

def all_angles():
    os.chdir("C:/Users/Julia/Desktop/Daten-BA/SMI_RawData")
    #get all files in directory
    list = os.listdir() 
    
    # remove participants
    #list.remove("Event Statistics - Single - 82.txt")
    vergence = pd.DataFrame()
    df_summary_statistic_all_participants = pd.DataFrame()
    for filename in list:
        print(filename)
        # input: filename # output: dataframe to work on
        df = read_data(filename)
        
        df_10_sec_of_videos = cut_last_10sec_of_videos(df)
        
        list_vergence_angles = vergence_angles(df_10_sec_of_videos)
        serie_vergence_distance = vergence_distance(df)
        
        df_summary_statistics = summary_statistics(df_10_sec_of_videos, list_vergence_angles, serie_vergence_distance)
        df_summary_statistic_all_participants = pd.concat([df_summary_statistic_all_participants, df_summary_statistics])

    return df_summary_statistic_all_participants

