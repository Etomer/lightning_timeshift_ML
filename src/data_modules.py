import h5py, glob, torch
import lightning as L
from torch.utils.data import Dataset, DataLoader,random_split
from scipy.io import wavfile


# --------------- Data Modules --------------------------------

class PairedFFTDataModule(L.LightningDataModule):
    """
    Data Module for datasets saved in the "raw" format expected by the ML-models. 
    Mostly intended to be used for real data instead of simulated data.
    
    Expected datasets of .hdf5 file "data_dir" points at:
        "input" : shape = (n_examples, 4, n_fft_components)
        "gt" : shape = (n_examples), 
            
    """
    def __init__(self, data_dir: str = "./"):
        raise NotImplementedError

    def prepare_data(self):
        pass


class ImpulseResponseDataModule(L.LightningDataModule):
    """
    Data Module for datasets stored as impulse responses. This enables efficient storage 
    since any sound can be simulated for a given room. Code for simmulating sound and 
    repackaging it into the format of "PairedFFTDataModule" is contained in this class.
    
    Expected datasets of .hdf5 file "data_dir" points at:
    input : shape = (n_rooms, n_mics, rir_len)
    gt : shape = (n_rooms, n_mics)

    Expects "sound_dir" to point to a directory containing ".wav" sound files.
    """

    def __init__(self, data_dir: str, sound_dir: str):
        raise NotImplementedError

    def prepare_data(self):
        pass

class MovingImpulseResponseDataModule(L.LightningDataModule):
    """
    Data Module for datasets stored as impulse responses, with the addition that the 
    sound source is moving (not modeled as continous motion but as if the sound 
    source teleports along a path). This enables efficient storage 
    since any sound can be simulated for a given room. Code for simmulating sound and 
    repackaging it into the format of "PairedFFTDataModule" is contained in this class.
    
    Expected datasets of .hdf5 file "data_dir" points at:
    input : shape = (n_rooms, n_mics, n_source_locations, rir_len)
    gt : shape = (n_rooms, n_mics)

    Expects "sound_dir" to point to a directory containing ".wav" sound files.
    """
    def __init__(self,
        data_path: str,
        sound_dir: str,
        sample_length: int = 10000, #number of samples
        max_freq: float = 4000.0,  # Hz
        batch_size : int = 1,
        n_mics_per_batch = 17,
        max_shift : float = 500, #samples
        n_shift_bins : int = 500,
        validation_percentage = 0.05,
        transform = None,
        data_val_path : str = None,
        ):
        super().__init__()
        self.data_path = data_path
        self.sound_dir = sound_dir
        self.sample_length = sample_length
        self.max_freq = max_freq
        self.batch_size = batch_size
        self.n_mics_per_batch = n_mics_per_batch
        self.max_shift = max_shift
        self.n_shift_bins = n_shift_bins
        self.validation_percentage = validation_percentage
        
        # Open dataset
        self.h5file = h5py.File(data_path,"r")
        self.dataset_metadata = self.h5file.attrs
        self.dataset = dataset_hdf5(self.h5file, n_mics_per_batch) 
        if data_val_path is None:
            self.dataset_test = self.dataset
            self.dataset_train = torch.utils.data.Subset(self.dataset, range(int(len(self.dataset)*(1-validation_percentage))))
            self.dataset_val = torch.utils.data.Subset(self.dataset, range(int(len(self.dataset)*(1-validation_percentage)), len(self.dataset)))
        else:
            self.dataset_test = self.dataset
            self.dataset_train = self.dataset
            self.dataset_val = dataset_hdf5(h5py.File(data_val_path,"r"), n_mics_per_batch) 

        # Find sound files
        self.sound_files = glob.glob(sound_dir + "*.wav")
        if not transform:
            self.transform = lambda x : x
        else:
            self.transform = transform

        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass
        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     validation_size = int(self.validation_percentage*self.dataset.X.shape[0])
        #     self.dataset_train, self.dataset_val = random_split(self.dataset, [self.dataset.X.shape[0] - validation_size, validation_size])

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.dataset_test = self.dataset

    # def to_paired_fft_format(self,X,y):
    #     # rewritten from batched verison
    #     return X,y
    #     X = X.unsqueeze(0)
    #     y = y.unsqueeze(0)
    #     X,y = self.batch_to_paired_fft_format(X,y)
    #     X = X.squeeze(0)
    #     y = y.squeeze(0)
    #     return X,y

    def batch_to_paired_fft_format(self, batch):
        X,y = batch
        #const
        rir_len = self.dataset_metadata["rir_len"]
        batch_size = X.shape[0]
        n_sound_positions = X.shape[2]
        sim_sample_length = self.sample_length + rir_len # length used when simmulating sound, needs to be longer than the final sound to have echo in the entire sample

        # randomly select audio pieces to play
        signals = torch.zeros(batch_size,sim_sample_length)
        for batch_i in range(batch_size):
            sound_file = self.sound_files[torch.randint(len(self.sound_files),(1,))]
            fs, signal = wavfile.read(sound_file)
            if fs != self.dataset_metadata["fs"]:
                raise Exception("Please use .wav with the same sampling frequency as the simmulated impulse responses sampling frequency (probably 16 kHz")
            start = torch.randint(len(signal) - sim_sample_length, (1,))
            signals[batch_i,:] = torch.tensor(signal[start : start + sim_sample_length])
        
        # splitting the sound into the different speaker positions
        piece_length = signals.shape[1] // n_sound_positions
        signals = signals[:,:piece_length*n_sound_positions].reshape(batch_size, n_sound_positions, piece_length) 

        # simmulate sound from each speaker position, using zero-padding to length (signals.shape[2] + rir_len) to avoid errors from impulse response wrapping around
        signals = torch.fft.irfft(torch.fft.rfft(X, signals.shape[2] + rir_len) * torch.fft.rfft(signals, signals.shape[2] + rir_len).unsqueeze(1))

        # Summing and concatenating the contribution from the different speaker locations
        fin_signal = torch.zeros(batch_size, self.n_mics_per_batch, signals.shape[3] + piece_length*(n_sound_positions-1))
        for i in range(n_sound_positions):
            fin_signal[:,:,i*piece_length:piece_length*i+signals.shape[3]] = signals[:,:,i]
        fin_signal = fin_signal[:,:,-self.sample_length:]

        # Only keep frequency components lower than max_freq_component
        max_freq_component = int(self.max_freq*self.sample_length/self.dataset_metadata["fs"])
        fin_signal = torch.fft.rfft(fin_signal)[:,:,:max_freq_component].unsqueeze(2)

        # NOTE: if n_mics_per_batch is not odd, then we will compute all pairs of microphones except one
        fin_signal = torch.concatenate([torch.concatenate([fin_signal, fin_signal.roll(i + 1, 1)], dim=2) for i in range((self.n_mics_per_batch-1) // 2)],dim=1)  # organize sounds pairwise, 
        fin_signal = fin_signal.view(batch_size * fin_signal.shape[1],2,fin_signal.shape[3])

        # concatenate real and imaginary parts
        fin_signal = torch.concatenate([fin_signal.real, fin_signal.imag], dim=1)

        # compute the ground truth distanc differences 
        y = torch.concatenate([y - y.roll(i + 1, 1) for i in range((self.n_mics_per_batch-1) // 2)], dim=1).view(-1) # compute pairwise distance-difference
        y *= self.dataset_metadata["fs"]/self.dataset_metadata["speed_of_sound"] # rescale distance-difference to sample_difference
        bin_edges = torch.linspace(-self.max_shift,self.max_shift,self.n_shift_bins+1)
        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")
        y = (y.unsqueeze(1) < bin_edges).to(torch.long).argmax(dim=1) - 1  # Bins the values in y, since argmax finds first occurence where condition is met.

        return fin_signal,y


    def my_collate_fn(self,batch):
        batch = torch.stack([b[0] for b in batch],dim=0),torch.stack([b[1] for b in batch],dim=0)
        batch = self.batch_to_paired_fft_format(batch)
        batch = self.transform(batch)
        return batch


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, collate_fn=self.my_collate_fn,num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=self.my_collate_fn,num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=self.my_collate_fn,num_workers=7)



# --------------- Helper --------------------------------

class dataset_hdf5(Dataset):

    def __init__(self, dataset, n_mics_per_batch):

        self.X = dataset["input"]
        self.y = dataset["gt"]
        self.n_mics_per_batch = n_mics_per_batch
        #self.to_paired_fft_format = to_paired_fft_format

    def __getitem__(self, idx):
        mics = torch.randperm(self.X.shape[1])[:self.n_mics_per_batch].sort().values # sorting is just because indexing in hdf5 requires it 
        return torch.tensor(self.X[idx,mics]), torch.tensor(self.y[idx,mics])
    
    def __len__(self):
        return self.X.shape[0]
