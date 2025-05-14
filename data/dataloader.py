import numpy as np
import torch

def noise_injection(sample, noise_factor=0.05):
    return sample + noise_factor * np.random.normal(size=sample.shape)

class MyDataset(torch.utils.data.Dataset):
    """
    Class to store a given dataset.
    Parameters:
    - samples: list of lists, each containing list of tensors (modalities) + labels
    """

    def __init__(self, samples):
        self.num_samples = len(samples)
        self.data = samples

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        sample, label = self.data[idx]
        
        # Put channels first and add a dimension if the signal has 1 channel (e.g. GSR)
        sample = [signal.transpose(1, 0) if len(signal.shape) > 1 else signal.unsqueeze(0) for signal in sample]

        return sample, label

def MyDataLoader(train_file, test_file, train_batch_size, test_batch_size, num_workers=1):
    print("----Loading dataset----")
    
    training = torch.load(train_file)  # Loads an object saved with torch.save() from a file
    validation = torch.load(test_file)  # Loads an object saved with torch.save() from a file
    
    train_dataset = MyDataset(training)
    eval_dataset = MyDataset(validation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    
    y_train = [y for x, y in training]
    _, train_distr = np.unique(y_train, return_counts=True) # number of labels in train dataset, for each class
    weights = sum(train_distr)/train_distr
    sample_weights= weights/sum(weights)  # sample_weights in case of unbalanced data

    print('Dataset: MAHNOB-HCI')
    print("#Training samples: ", len(train_dataset))
    print("#Validation samples: ", len(eval_dataset))
    print("#Training distribution: ", train_distr)
    print("-------------------------")

    return train_loader, eval_loader, sample_weights