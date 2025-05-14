import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hypercomplex_layers import PHConv1d, PHMLinear

class eyePHBase(nn.Module): 
    "Base for the eye Modality."
    def __init__(self, n=4, units=128):
        super(eyePHBase, self).__init__()  # call the parent constructor
        self.C1 = PHConv1d(n, 4, units, kernel_size=3, stride=1)
        self.BN1 = nn.BatchNorm1d(units)
        self.C2 = PHConv1d(n, units, units*2, kernel_size=3, stride=1)
        self.BN2 = nn.BatchNorm1d(units*2)
        # self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.C1(inputs)
        x = F.relu(self.BN1(x))
        x = self.C2(x)
        x = F.relu(self.BN2(x))
        # x = F.relu(self.D3(x))
        x = torch.nn.AdaptiveAvgPool1d(output_size=1)(x).squeeze(-1)
        return x

class GSRPHBase(nn.Module):  
    "Base for the GSR Modality."
    def __init__(self, n=1, units=130):
        super(GSRPHBase, self).__init__()  # call the parent constructor
        self.D1 = PHMLinear(n, 1280, units)
        self.BN1 = nn.BatchNorm1d(units)
        # self.D2 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.D1(inputs).squeeze(1) # remove the channel dimension
        x = F.relu(self.BN1(x))
        # x = F.relu(self.D2(x))
        return x

class EEGPHBase(nn.Module):  
    "Base for the EEG Modality."
    def __init__(self, n=10, units=1020):
        super(EEGPHBase, self).__init__()  # call the parent constructor
        self.C1 = PHConv1d(n, 10, units, kernel_size=3, stride=1)
        self.BN1 = nn.BatchNorm1d(units)
        self.C2 = PHConv1d(n, units, units*2, kernel_size=3, stride=1)
        self.BN2 = nn.BatchNorm1d(units*2)
        # self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.C1(inputs)
        x = F.relu(self.BN1(x))
        x = self.C2(x)
        x = F.relu(self.BN2(x))
        # x = F.relu(self.D3(x))
        x = torch.nn.AdaptiveAvgPool1d(output_size=1)(x).squeeze(-1)
        return x

class ECGPHBase(nn.Module): 
    "Base for the ECG Modality."
    def __init__(self, n=3, units=513):
        super(ECGPHBase, self).__init__()  # call the parent constructor
        self.C1 = PHConv1d(n, 3, units, kernel_size=3, stride=1)
        self.BN1 = nn.BatchNorm1d(units)
        self.C2 = PHConv1d(n, units, units*2, kernel_size=3, stride=1)
        self.BN2 = nn.BatchNorm1d(units*2)
        # self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.C1(inputs)
        x = F.relu(self.BN1(x))
        x = self.C2(x)
        x = F.relu(self.BN2(x))
        # x = F.relu(self.D3(x))
        x = torch.nn.AdaptiveAvgPool1d(output_size=1)(x).squeeze(-1)
        return x

class H2(nn.Module): # aka ConvHyperNet
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    def __init__(self, dropout_rate, units=1024, n=4, n_eye=4, n_gsr=1, n_eeg=10, n_ecg=3):
        super(H2, self).__init__()  # call the parent constructor
        self.eye = eyePHBase(n=n_eye)
        self.gsr = GSRPHBase(n=n_gsr)
        self.eeg = EEGPHBase(n=n_eeg)
        self.ecg = ECGPHBase(n=n_ecg)
        self.drop1 = nn.Dropout(dropout_rate)
        # self.D1 = PHMLinear(n, 1792, 1792)
        # self.BN1 = nn.BatchNorm1d(1792)
        self.D2 = PHMLinear(n, 3452, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = PHMLinear(n, units, units//2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.BN3 = nn.BatchNorm1d(units//2)
        self.D4 = PHMLinear(n, units//2, units//4)
        self.drop3 = nn.Dropout(dropout_rate)
        self.out_3 = nn.Linear(units//4, 3)
    
    def get_features(self, eye, gsr, eeg, ecg, level='encoder'):
        assert level in ['encoder', 'classifier']
        eye_out = self.eye(eye)
        gsr_out = self.gsr(gsr)
        eeg_out = self.eeg(eeg)
        ecg_out = self.ecg(ecg)
        concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
        if level == 'encoder':
            return concat
        x = F.relu(self.BN2(self.D2(concat)))
        x = F.relu(self.BN3(self.D3(x)))
        x = F.relu(self.D4(x))
        return x

    def forward(self, eye, gsr, eeg, ecg):
        eye_out = self.eye(eye)
        gsr_out = self.gsr(gsr)
        eeg_out = self.eeg(eeg)
        ecg_out = self.ecg(ecg)
        concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
        # x = self.D1(concat)
        # x = F.relu(self.BN1(x))
        x = self.D2(concat)
        x = F.relu(self.BN2(x))
        x = self.drop1(x)
        x = self.D3(x)
        x = F.relu(self.BN3(x))
        x = self.drop2(x)
        x = F.relu(self.D4(x))
        x = self.drop3(x)
        out = self.out_3(x)  # Softmax would be applied directly by CrossEntropyLoss, because labels=classes
        return out
