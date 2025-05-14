import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hypercomplex_layers import PHMLinear

class eyeBasev2(nn.Module): 
    "Base for the eye Modality."
    def __init__(self, units=128):
        super(eyeBasev2, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(600*4, units)
        self.BN1 = nn.BatchNorm1d(units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        return x

class GSRBasev2(nn.Module):  
    "Base for the GSR Modality."
    def __init__(self, units=128):
        super(GSRBasev2, self).__init__()  # call the parent constructor
        self.D1 = nn.Linear(1280, units)
        self.BN1 = nn.BatchNorm1d(units)

    def forward(self, inputs):
        x = self.D1(inputs).squeeze(1) # remove the channel dimension
        x = F.relu(self.BN1(x))
        return x

class EEGBasev2(nn.Module):  
    "Base for the EEG Modality."
    def __init__(self, units=1024):
        super(EEGBasev2, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280*10, units)
        self.BN1 = nn.BatchNorm1d(units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        return x

class ECGBasev2(nn.Module): 
    "Base for the ECG Modality."
    def __init__(self, units=512):
        super(ECGBasev2, self).__init__()  # call the parent constructor
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(1280*3, units)
        self.BN1 = nn.BatchNorm1d(units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        return x

class HyperFuseNetv2(nn.Module): 
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    def __init__(self, dropout_rate, units=1024, n=4):
        super(HyperFuseNetv2, self).__init__()  # call the parent constructor
        self.eye = eyeBasev2()
        self.gsr = GSRBasev2()
        self.eeg = EEGBasev2()
        self.ecg = ECGBasev2()
        self.drop1 = nn.Dropout(dropout_rate)
        self.D2 = PHMLinear(n, 1792, units)
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
        
        x = self.D2(concat)
        x = F.relu(self.BN2(x))
        x = self.drop1(x)
        x = self.D3(x)
        x = F.relu(self.BN3(x))
        x = self.drop2(x)
        x = F.relu(self.D4(x))
        x = self.drop3(x)
        out = self.out_3(x)
        return out

    def forward(self, eye, gsr, eeg, ecg):
        eye_out = self.eye(eye)
        gsr_out = self.gsr(gsr)
        eeg_out = self.eeg(eeg)
        ecg_out = self.ecg(ecg)
        concat = torch.cat([eye_out, gsr_out, eeg_out, ecg_out], dim=1)
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
