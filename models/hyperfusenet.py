import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hypercomplex_layers import PHMLinear

class eyeBase(nn.Module): 
    "Base for the eye Modality."
    def __init__(self, units=128):
        super(eyeBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(600*4, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

class GSRBase(nn.Module):  
    "Base for the GSR Modality."
    def __init__(self, units=128):
        super(GSRBase, self).__init__()  # call the parent constructor
        self.D1 = nn.Linear(1280, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.D1(inputs).squeeze(1)
        x = F.relu(self.BN1(x))
        x = F.relu(self.D2(x))
        return x

class EEGBase(nn.Module):  
    "Base for the EEG Modality."
    def __init__(self, units=1024):
        super(EEGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280*10, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

class ECGBase(nn.Module): 
    "Base for the ECG Modality."
    def __init__(self, units=512):
        super(ECGBase, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280*3, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x

class HyperFuseNet(nn.Module): 
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    def __init__(self, dropout_rate, units=1024, n=4):
        super(HyperFuseNet, self).__init__()  # call the parent constructor
        self.eye = eyeBase()
        self.gsr = GSRBase()
        self.eeg = EEGBase()
        self.ecg = ECGBase()
        self.drop = nn.Dropout(dropout_rate)
        self.D1 = PHMLinear(n, 1792, 1792)
        self.BN1 = nn.BatchNorm1d(1792)
        self.D2 = PHMLinear(n, 1792, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = PHMLinear(n, units, units//2)
        self.BN3 = nn.BatchNorm1d(units//2)
        self.D4 = PHMLinear(n, units//2, units//4)
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
        x = F.relu(self.BN1(self.D1(concat)))
        x = F.relu(self.BN2(self.D2(x)))
        x = F.relu(self.BN3(self.D3(x)))
        x = F.relu(self.D4(x))
        return x

    def forward(self, eye, gsr, eeg, ecg):
        eye_out = self.eye(eye)
        gsr_out = self.gsr(gsr)
        eeg_out = self.eeg(eeg)
        ecg_out = self.ecg(ecg)
        concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
        x = self.D1(concat)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = self.drop(x)
        x = self.D3(x)
        x = F.relu(self.BN3(x))
        x = F.relu(self.D4(x))
        out = self.out_3(x)  # Softmax would be applied directly by CrossEntropyLoss, because labels=classes
        return out