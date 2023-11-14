import argparse
import os
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from scipy import signal
import copy

from tqdm import tqdm

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

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

class DatasetCreator:
    """loops over subject folders.
    Params:
        preproc_data: Folder that contains one folder per subject,
                each of which contain respective label files.
        label_kind: Which label to use ["Arsl" or "Vlnc"].
        physio_f: sampling frequency of physiological data.
        gaze_f: sampling frequency of gaze data.
        block_len : number of seconds extracted from the end of each given sample.
        sample_len: length in seconds of each generated sample.
        overlap: percentage of overlapping between samples."""
    def __init__(self, preproc_data, label_kind,
                 physio_f, gaze_f, block_len, sample_len, overlap, verbose=False):
        self.preproc_data = preproc_data
        self.label_kind = label_kind
        self.physio_f = physio_f
        self.gaze_f = gaze_f
        self.block_len = block_len
        self.sample_len = sample_len
        self.overlap = overlap
        self.verbose = verbose

    def save_to_list(self):
        """Grabs data from hierarchical structure and unpacks all values.
        Add gathered data to a list as a single sample."""
        sub_dir = list_files(self.preproc_data, sorted_dir=False)
        data_list = []
        labels = []
        for dir in tqdm(sub_dir, desc='Reading data'):  # for each subject/folder
            subj = int(dir[-2:])

            # Get labels for current subject and label kind
            all_labels = np.genfromtxt(os.path.join(dir, 'labels_felt{}.csv'  # return Dataframe
                                    .format(self.label_kind)), delimiter=',')

            # Get each original sample and create dataset samples
            id_trials = [x.split("/")[-1].partition("_")[0] for x in list_files(dir, sorted_dir=False)] # get beggining of files
            id_trials = sorted(np.unique(id_trials)[:-1], key=lambda x: int(x))  # remove duplicates, "label", and sort
            for i, id in enumerate(tqdm(id_trials, desc=f'Subject {subj}')):
                pupil_data = np.genfromtxt(os.path.join(dir, '{}_PUPIL.csv'
                                    .format(id)), delimiter=',')
                gaze_data = np.genfromtxt(os.path.join(dir, '{}_GAZE_COORD.csv'
                                    .format(id)), delimiter=',')
                eye_dist_data = np.genfromtxt(os.path.join(dir, '{}_EYE_DIST.csv'
                                    .format(id)), delimiter=',')
                gsr_data = np.genfromtxt(os.path.join(dir, '{}_GSR-NO-BASE.csv'
                                    .format(id)), delimiter=',')
                eeg_data = np.genfromtxt(os.path.join(dir, '{}_EEG.csv'
                                     .format(id)), delimiter=',')
                ecg_data = np.genfromtxt(os.path.join(dir, '{}_ECG.csv'
                                   .format(id)), delimiter=',')

                # Extract last <block_len> seconds for each data
                n_points_block_gaze = self.block_len * self.gaze_f
                n_points_block_physio = self.block_len * self.physio_f

                pupil_data = pupil_data[-n_points_block_gaze:]
                gaze_data = gaze_data[-n_points_block_gaze:]
                eye_dist_data = eye_dist_data[-n_points_block_gaze:]
                gsr_data = gsr_data[-n_points_block_physio:]
                eeg_data = eeg_data[-n_points_block_physio:]
                ecg_data = ecg_data[-n_points_block_physio:]

                # Extract samples of <sample_len> seconds
                n_points_sample_gaze = self.sample_len * self.gaze_f
                n_points_sample_physio = self.sample_len * self.physio_f
                overlap_step_gaze = int(n_points_sample_gaze * self.overlap)
                overlap_step_physio = int(n_points_sample_physio * self.overlap)

                for j, k in zip(range(0, n_points_block_gaze - overlap_step_gaze, n_points_sample_gaze - overlap_step_gaze),
                                range(0, n_points_block_physio - overlap_step_physio, n_points_sample_physio - overlap_step_physio)):
                    pupil = pupil_data[j : j + n_points_sample_gaze]
                    gaze_coord = gaze_data[j : j + n_points_sample_gaze]
                    eye_dist = eye_dist_data[j : j + n_points_sample_gaze]
                    gsr = gsr_data[k : k + n_points_sample_physio]
                    eeg = eeg_data[k : k + n_points_sample_physio]
                    ecg = ecg_data[k : k + n_points_sample_physio]
                    
                   
                    if (len(pupil) != n_points_sample_gaze or len(gaze_coord) != n_points_sample_gaze or len(eye_dist) != n_points_sample_gaze or
                        len(gsr) != n_points_sample_physio or len(eeg) != n_points_sample_physio or len(ecg) != n_points_sample_physio):
                        # sanity check on the samples
                        print("\033[93mData segment went wrong at subject: {}, sample: {}!\033[0m".format(subj, id))

                    # Check gaze data noise
                    clean_pupil = pupil[pupil != -1]
                    clean_gaze_coord = gaze_coord[gaze_coord != -1]
                    clean_eye_dist = eye_dist[eye_dist != -1]
                    if len(clean_pupil)/len(pupil) < 0.6 or len(clean_gaze_coord)/len(gaze_coord) < 0.6 or len(clean_eye_dist)/len(eye_dist) < 0.6:
                        if self.verbose:
                            print("\033[93mGaze segment too noisy for subject: {}, sample: {}, segment:{}!\033[0m"
                                    .format(subj, id, str(j//(n_points_sample_gaze - overlap_step_gaze))))
                        continue

                    # Create single variable containing all gaze information
                    eye = np.column_stack((pupil, gaze_coord, eye_dist))

                    # Create tensor for each modality
                    eye = torch.FloatTensor(eye)  # transforms list into a serialized (string) tf.TensorProto proto
                    gsr = torch.FloatTensor(gsr)
                    eeg = torch.FloatTensor(eeg)
                    ecg = torch.FloatTensor(ecg)
                    label = all_labels[i]
                    if label <= 0 and label > 9:  # sanity check on the labels
                        print("\033[93mLabels went wrong at subject: {}, sample: {}!\033[0m".format(subj, id))
                    if label <=3: label = 0
                    elif label >=7: label = 2
                    else: label = 1

                    data_list.append([eye, gsr, eeg, ecg])
                    labels.append(label)

        return data_list, labels

def std_for_SNR(signal, noise, snr):
    '''Compute the gain to be applied to the noise to achieve the given SNR in dB'''
    signal_power = np.var(signal.numpy())
    noise_power = np.var(noise.numpy())
    # Gain to apply to noise, from SNR formula: SNR=10*log_10(signal_power/noise_power)
    g = np.sqrt(10.0 ** (-snr/10) * signal_power / noise_power)
    return g

def add_gauss_noise(sample, snr):
    sample = copy.deepcopy(sample)  # need to not modify directly passed sample
    sample_size = sample[1].size()  # GSR data size
    # Create tensor with physio size, 1 ch, but initialized with standard normal distribution
    gaussian_noise = torch.empty(sample_size).normal_(mean=0,std=1)
    # Apply same noise to all modalities, but with std needed to obtain given SNR -------------------

    # Gaze modalities: need downsampled noise, apply different std for channel since very different
    ch_noise = torch.FloatTensor(signal.resample(gaussian_noise.numpy(), sample[0].size()[0]))
    for i in range(4):
        idx = np.where(sample[0][:, i]!=-1)[0]  # indeces elements to scale
        sample[0][idx, i] = sample[0][idx, i] + std_for_SNR(sample[0][idx, i], ch_noise, snr) * ch_noise[idx].T

    # Physio modalities: apply different std for modality
    for i in range(1, len(sample)):
        if len(sample[i].size()) > 1:
            noise = gaussian_noise.repeat(sample[i].size()[1],1).t()
        else:
            noise = gaussian_noise
        sample[i] = sample[i] + std_for_SNR(sample[i], noise, snr) * noise
    return sample

def scaled_sample(sample, min, max):
    sample = copy.deepcopy(sample)  # need to not modify directly passed sample
    # Compute scaling factor
    alpha = random.uniform(min, max)
    # Apply same scaling factor to all modalities

    # Gaze modalities: need exclude -1, apply different mask for channel since may be different
    for i in range(4):
        idx = np.where(sample[0][:, i]!=-1)[0]  # indeces elements to scale
        sample[0][idx, i] = alpha * sample[0][idx, i]

    # Physio modalities
    for i in range(1, len(sample)):
        sample[i] = alpha * sample[i]
    return sample

def load_dataset(data, labels, scaling, noise, m, SNR):
    '''Params:
        scaling (bool): if scaling is applied
        noise (bool): if noise is applied
        m: multiplyg factor for augmentation (m=1 if no augm)
        SNR: which SNR to be used '''
    dataset = []

    for i in range(len(data)):
        sample = data[i]
        label= labels[i]
        dataset.append([sample, label])
        if scaling:
            dataset.append([scaled_sample(sample, 0.7, 0.8), label])
            dataset.append([scaled_sample(sample, 1.2, 1.3), label])
            if noise:
                for j in range(0, m-3, 3):
                    dataset.append([add_gauss_noise(sample, snr=SNR), label])
                    dataset.append([add_gauss_noise(scaled_sample(sample, 0.7, 0.8), snr=SNR), label])
                    dataset.append([add_gauss_noise(scaled_sample(sample, 1.2, 1.3), snr=SNR), label])
        elif noise:
            for j in range(m-1):
                dataset.append([add_gauss_noise(sample, snr=SNR), label])

    return dataset  # list of lists, each containing a sample(=list of tensors, one per modality) and label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_data_path', type=str, default='hci-tagging-database/preproc_data', help='Path to folder where preprocessed data was saved')
    parser.add_argument('--save_path', type=str, default='hci-tagging-database/torch_datasets', help='Path to save .pt files')
    parser.add_argument('--label_kind', type=str, default='Arsl', help="Choose valence (Vlnc) or arousal (Arsl) label")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    assert args.label_kind in ["Arsl", "Vlnc"]
    print("Creating dataset for label: ", args.label_kind)

    set_seed(args.seed)

    d = DatasetCreator(args.preproc_data_path, args.label_kind, physio_f = 128, gaze_f = 60, block_len = 30, sample_len=10, overlap = 0, verbose=args.verbose)  # create object
    data, labels = d.save_to_list()  # call method
    print(len(data))

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=args.seed, stratify=labels)
    
    # Augmentation
    m = 30 # Number of augmented signals for each original sample
    SNR = 5

    train_data = load_dataset(X_train, y_train, scaling=True, noise=True, m=m, SNR=SNR)
    test_data = load_dataset(X_test, y_test, scaling=False, noise=False, m=1, SNR=None)

    print("Len train before augmentation: ", len(X_train))
    print("Len train after augmentation: ", len(train_data))
    print("Len test: ", len(test_data))
    print("Tot dataset: ", len(train_data) + len(test_data))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    torch.save(train_data, f'{args.save_path}/train_augmented_data_{args.label_kind}.pt')
    torch.save(test_data,  f'{args.save_path}/test_data_{args.label_kind}.pt')
