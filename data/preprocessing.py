import argparse
import os
import xml.etree.ElementTree as ET
import mne
import pandas as pd
import numpy as np 

def list_files(directory, sorted_dir):
    """List all files (i.e. their paths) in the dataset directory. Need sorted argument
     when directories' names contain numbers having different number of digits """
    files = []
    if sorted_dir:
        list_dir = sorted(os.listdir(directory), key=lambda filename: int(filename.split('.')[0]))  # key needed to get folder in numeric order
    else:
        list_dir = os.listdir(directory)
    for filename in list_dir:
        # if filename.endswith(".csv"):
        single = os.path.join(directory, filename)
        files.append(single)
    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    sessions_dir = list_files(args.sessions_path, sorted_dir=True)

    for dir_id in range(len(sessions_dir)):
        dir = sessions_dir[dir_id]

        # SESSION.XML --------------------------------------------------------------
        labels_file = os.path.join(dir, "session.xml")
        tree = ET.parse(labels_file)
        root = tree.getroot() 

        # Get labels
        feltEmo = root.attrib['feltEmo']
        feltArsl = root.attrib['feltArsl']
        feltVlnc = root.attrib['feltVlnc']
        
        # Get subject infos
        session = root.attrib['cutNr']
        subject = root[0].attrib['id']
        print("\033[94mCurrently considerig sub:{}, session:{}\033[0m".format(subject, session))

        # PHYSIOLOGICAL DATA -------------------------------------------------------
        physio_file = os.path.join(dir, "Part_{}_S_Trial{}_emotion.bdf".format(subject, int(session)//2))
        raw = mne.io.read_raw_bdf(physio_file, preload=True)
        # documentation for mne raw: https://mne.tools/1.0/auto_tutorials/raw/10_raw_overview.html#sphx-glr-auto-tutorials-raw-10-raw-overview-py

        # Get general info from file
        n_time_samps = raw.n_times  # number of samples
        time_secs = raw.times  # corresponding second [s] of each sample
        ch_names = raw.ch_names
        n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
        sfreq = raw.info['sfreq']
        highpass = raw.info['highpass']
        lowpass = raw.info['lowpass']

        # Resample all data from 256 Hz to 128 Hz, passing status channels as stimuli
        # documentation: https://mne.tools/0.24/auto_tutorials/preprocessing/30_filtering_resampling.html
        # OBS: this function applies first a brick-wall filter at the Nyquist frequency of the desired new sampling rate (i.e. 64Hz)
        raw = raw.resample(sfreq=128, stim_picks=46)

        # Get status channel to extract video's initial and ending samples (to remove baseline pre/post-stimulus)
        status_ch, time = raw[-1]  # extract last channel
        video_indices = np.where(np.diff(status_ch) > 0)[1]  # check indices of rising edges along dim 1
        if video_indices.size != 2:  # if there are not 2 rising edges
            raise NameError('Physiological data: Recording ended prematurely!')
        video_indices = video_indices + (1,1)  # to get real indices (right after rising edges, not before)

        # Get channels of interest
        EEG_CH = ch_names[0:32]
        SELECTED_EEG_CH = ["F3", "F7", "FC5", "T7", "P7",
                        "F4", "F8", "FC6", "T8", "P8"]
        ECG_CH = ch_names[32:35]
        GSR_CH = ch_names[40]

        # PREPROCESSING 
        # EEG ------------------------------------------------------------------------------------------- 
        raw_eeg = raw.copy().pick_channels(EEG_CH)
        # Referencing to average reference 
        # documentation: https://mne.tools/dev/generated/mne.set_eeg_reference.html
        raw_eeg = raw_eeg.set_eeg_reference(ref_channels='average')
        # Artifact removal and filtering  
        # documentation: https://mne.tools/0.24/auto_tutorials/preprocessing/30_filtering_resampling.html
        # Power line at 50 Hz, as proved with plots below
        # Band pass FIR filter from 1 - 45 Hz => still need to apply notch filter at 50Hz,
        # since the filter is not acting upon the 50Hz component (neglectable attenuation)
        raw_eeg = raw_eeg.notch_filter(50)
        raw_eeg = raw_eeg.filter(l_freq=1,  h_freq=45)
        # OBS the order between notch and bandpass filter is inrelevant (TRIED)
        # EOG removal: not considered for now, TODO?
        # ECG -------------------------------------------------------------------------------------------
        raw_ecg = raw.copy().pick_channels(ECG_CH)
        # Artifact removal and filtering  
        # documentation: https://mne.tools/0.24/auto_tutorials/preprocessing/30_filtering_resampling.html
        # Power line at 50 Hz, as proved with plots below
        # Band pass FIR filter from 0.5 - 45 Hz => still need to apply notch filter at 50Hz,
        # since the filter is not acting upon the 50Hz component (neglectable attenuation)
        raw_ecg = raw_ecg.notch_filter(50)
        raw_ecg = raw_ecg.filter(l_freq=0.5,  h_freq=45)
        # OBS the order between notch and bandpass filter is inrelevant (TRIED)
        # GSR -------------------------------------------------------------------------------------------
        raw_gsr = raw.copy().pick_channels([GSR_CH])
        # Artifact removal and filtering  
        # documentation: https://mne.tools/0.24/auto_tutorials/preprocessing/30_filtering_resampling.html
        # Power line at 50 Hz, as proved with plots below
        # Low pass FIR filter at 60 Hz => still need to apply notch filter at 50Hz
        raw_gsr = raw_gsr.notch_filter(50)
        raw_gsr = raw_gsr.filter(l_freq=None,  h_freq=60)
        # OBS the order between notch and bandpass filter is inrelevant (TRIED)

        # Extract data (removing baseline pre/post-stimulus)
        eeg_data, time = raw_eeg[SELECTED_EEG_CH, video_indices[0]:video_indices[1]]
        ecg_data, time = raw_ecg[:, video_indices[0]:video_indices[1]]
        gsr_data, time = raw_gsr[:, video_indices[0]:video_indices[1]]

        # Baseline correction for GSR (to homologate with others, who instead had high-pass filters)
        baseline, _ = raw_gsr[:, video_indices[0]-26:video_indices[0]]  # 26 samples with 128 Hz <=> 200ms
        if np.array(baseline).all()==0:  # empty baseline 
            raise NameError('GSR data: Baseline not found (missing data)!')
        gsr_data = gsr_data - np.mean(baseline)

        # GAZE DATA ----------------------------------------------------------------
        gaze_file = os.path.join(dir, "P{}-Rec1-All-Data-New_Section_{}.tsv".format(subject, session))
        gaze_df = pd.read_csv(gaze_file, sep = '\t', skiprows=23)

        # Extract important columns
        gaze_df = gaze_df[["Number", "GazePointXLeft", "GazePointYLeft", "DistanceLeft", "PupilLeft", "ValidityLeft",
                            "GazePointXRight", "GazePointYRight", "DistanceRight", "PupilRight", "ValidityRight", "Event"]]

        # Take rows with timestamp between ones of "event"=="MovieStart" e "MovieEnd" 
        video_start = gaze_df.index[gaze_df["Event"] == "MovieStart"].tolist()
        if not video_start: # empty list 
            raise NameError('Gaze data: Recording not started!')
        else:
            video_start = video_start[0] + 1  # need to get data right after video started
            # OBS: first row has always StimuliName == "No Media" but, since considering 
            # events to segment data, we will include these samples too, even if too many 
            # samples w.r.t. physio; not big deal since will take samples from end, just
            # might slightly be for baseline, but in case only missing a sample in the mean 

        video_end = gaze_df.index[gaze_df["Event"] == "MovieEnd"].tolist()
        if not video_end: # empty list 
            raise NameError('Gaze data: Recording ended prematurely!')
        else:
            video_end = video_end[0]

        # BASELINE CASE
        # last_baseline_samples = gaze_df[["PupilLeft", "PupilRight"]].iloc[0:video_start-1, :]

        gaze_df = gaze_df.iloc[video_start:video_end, :]

        # Remove rows with extra events (as "KeyPress", "RightMouseClick" or "LeftMouseClick") 
        key_indices = gaze_df.index[gaze_df["Event"] == "KeyPress"].tolist()
        r_mouse_indices = gaze_df.index[gaze_df["Event"] == "RightMouseClick"].tolist()
        l_mouse_indices = gaze_df.index[gaze_df["Event"] == "LeftMouseClick"].tolist()
        gaze_df = gaze_df.drop(key_indices + r_mouse_indices + l_mouse_indices)
        if np.unique(gaze_df["Event"].tolist()).size != 1:
            raise NameError('Gaze data: Unconsidered events!')

        # We removed all needed rows, so now need reset index column to avoid problem with index-iloc combos
        gaze_df = gaze_df.reset_index(drop=True)  

        # Check not having missed samples
        samples_id = [int(x) for x in gaze_df["Number"].tolist()]
        if np.sum(np.diff(samples_id) > 1) > 0:
            raise NameError('Gaze data: Missing samples!')

        # Compute mean pupilar size
        l_pupil_dim = [float(x) for x in gaze_df["PupilLeft"].tolist()]
        r_pupil_dim = [float(x) for x in gaze_df["PupilRight"].tolist()]
        mean_pupil_dim = []
        for i in range(len(l_pupil_dim)):
            if l_pupil_dim[i] >= 0  and r_pupil_dim[i] >= 0 and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
            # OBS: pupil can't be too big, if it is then error, and not considering values also in other columns (see test below)!
                mean_pupil_dim.append( round((l_pupil_dim[i] + r_pupil_dim[i])/2, 7) )
            elif l_pupil_dim[i] >= 0 and l_pupil_dim[i] < 9 and r_pupil_dim[i] < 0:
                mean_pupil_dim.append(l_pupil_dim[i])
            elif l_pupil_dim[i] < 0 and r_pupil_dim[i] >= 0 and r_pupil_dim[i] < 9:
                mean_pupil_dim.append(r_pupil_dim[i])
            else:
                mean_pupil_dim.append(-1)

        # Compute mean gaze coordinates
        l_gaze_x = [float(x) for x in gaze_df["GazePointXLeft"].tolist()]
        r_gaze_x = [float(x) for x in gaze_df["GazePointXRight"].tolist()]
        mean_gaze_x = []
        for i in range(len(l_gaze_x)):
            if l_gaze_x[i] >= 0  and r_gaze_x[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_gaze_x.append( round((l_gaze_x[i] + r_gaze_x[i])/2, 7) )
            elif l_gaze_x[i] >= 0 and l_pupil_dim[i] < 9 and r_gaze_x[i] < 0:
                mean_gaze_x.append(l_gaze_x[i])
            elif l_gaze_x[i] < 0 and r_gaze_x[i] >= 0 and r_pupil_dim[i] < 9:
                mean_gaze_x.append(r_gaze_x[i])
            else:
                mean_gaze_x.append(-1)
        
        l_gaze_y = [float(x) for x in gaze_df["GazePointYLeft"].tolist()]
        r_gaze_y = [float(x) for x in gaze_df["GazePointYRight"].tolist()]
        mean_gaze_y = []
        for i in range(len(l_gaze_y)):
            if l_gaze_y[i] >= 0  and r_gaze_y[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_gaze_y.append( round((l_gaze_y[i] + r_gaze_y[i])/2, 7) )
            elif l_gaze_y[i] >= 0 and l_pupil_dim[i] < 9 and r_gaze_y[i] < 0:
                mean_gaze_y.append(l_gaze_y[i])
            elif l_gaze_y[i] < 0 and r_gaze_y[i] >= 0 and r_pupil_dim[i] < 9:
                mean_gaze_y.append(r_gaze_y[i])
            else:
                mean_gaze_y.append(-1)

        # Compute mean eye distance
        l_eye_dist = [float(x) for x in gaze_df["DistanceLeft"].tolist()]
        r_eye_dist = [float(x) for x in gaze_df["DistanceRight"].tolist()]
        mean_eye_dist = []
        for i in range(len(l_eye_dist)):
            if l_eye_dist[i] >= 0  and r_eye_dist[i] >= 0  and l_pupil_dim[i] < 9  and r_pupil_dim[i] < 9:
                mean_eye_dist.append( round((l_eye_dist[i] + r_eye_dist[i])/2, 7) )
            elif l_eye_dist[i] >= 0 and l_pupil_dim[i] < 9 and r_eye_dist[i] < 0:
                mean_eye_dist.append(l_eye_dist[i])
            elif l_eye_dist[i] < 0 and r_eye_dist[i] >= 0 and r_pupil_dim[i] < 9:
                mean_eye_dist.append(r_eye_dist[i])
            else:
                mean_eye_dist.append(-1)

        # SAVE CURRENT TRIAL IN NEW DATASET (IN CSV FORMAT) ------------------------
        path_name = os.path.join(args.save_path, 'S'+f"{int(subject):02}")
        if not os.path.exists(path_name):
                os.makedirs(path_name)
                
        trial_data = [eeg_data.T, ecg_data.T, gsr_data.T, mean_pupil_dim, np.stack((mean_gaze_x, mean_gaze_y), axis=1), mean_eye_dist]
        signals_types = ["EEG", "ECG", "GSR", "PUPIL", "GAZE_COORD", "EYE_DIST"]
        for i in range(len(signals_types)):
            file_name = os.path.join(path_name, '{}_{}.csv'.format(int(session)//2 - 1, signals_types[i]))
            np.savetxt(file_name, trial_data[i], delimiter=",")

        trial_labels = [feltArsl, feltVlnc]
        labels_types = ["feltArsl", "feltVlnc"]
        for i in range(len(labels_types)):
            label_file_name = os.path.join(path_name, 'labels_{}.csv'.format(labels_types[i]))
            f = open(label_file_name, 'a+')  # open file in append mode, create if not exists
            f.write(trial_labels[i] + "\n")
            f.close()

    