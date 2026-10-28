"""
Torch dataset for nanopore reads

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py"""

import os
import random
import zipfile

import numpy as np
from torch.utils.data import Dataset, Sampler

from main.scripts.constants import TRAIN_DATA


class BaseNanoporeDataset(Dataset):
    """Base dataset class that contains Nanopore data

    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of
    byte-strings with the bases appended with ''.

    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.

    Args:
        data (str): dir with the npz files
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        wick (bool): if we are using the processed wick dataset
    """

    def __init__(
        self,
        data_dir: str,
        split: float = 0.95,
        shuffle: bool = True,
        seed: int = None,
        is_wick: bool = False,
        is_synthetic: bool = False,
        numpy_dir: str = "",
        image_start: int | None = None,
        image_end: int | None = None,
    ):
        super(BaseNanoporeDataset, self).__init__()

        self.data_dir = data_dir
        self.decoding_dict = {1: "A", 2: "C", 3: "G", 4: "T", 0: ""}
        self.encoding_dict = {"A": 1, "C": 2, "G": 3, "T": 4, "": 0}
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.is_wick = is_wick
        self.is_synthetic = is_synthetic
        self.numpy_dir = numpy_dir
        self.image_start = image_start
        self.image_end = image_end

        self.files_list = self._find_files()
        self.num_samples_per_file = self._get_samples_per_file()
        self.total_num_samples = np.sum(np.array(self.num_samples_per_file))
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self._split_train_validation()
        self._get_samplers()

        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

        self.token_sos = 1
        self.token_eos = 2
        self.token_pad = 0

    def __len__(self):
        """Number of samples"""
        return self.total_num_samples

    def __getitem__(self, idx):
        """Get a set of samples by idx

        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.

        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict=self.loaded_train_data, idx=idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict=self.loaded_validation_data, idx=idx[1])
        else:
            raise IndexError(
                "Given index not in train or validation files indices: " + str(idx[0])
            )

    def _find_files(self):
        """Finds list of files to read"""
        files = []
        if self.is_synthetic:
            fast5_dir = os.path.join(self.data_dir, "fast5")
            folders = []
            if self.image_end is not None:
                start_index = self.image_start if self.image_start is not None else 0
                end_index = self.image_end
                image_folders = set(
                    [f"image_{i}" for i in range(start_index, end_index)]
                )
                folders = [
                    os.path.join(fast5_dir, f)
                    for f in os.listdir(fast5_dir)
                    if (
                        os.path.isdir(os.path.join(fast5_dir, f)) and f in image_folders
                    )
                ]
            else:
                folders = [
                    os.path.join(fast5_dir, f)
                    for f in os.listdir(fast5_dir)
                    if os.path.isdir(os.path.join(fast5_dir, f))
                ]
            print(f"Found {len(folders)} folders")
            for directory in folders:
                nn_inputs_dir = os.path.join(directory, self.numpy_dir)
                all_files = [
                    os.path.join(nn_inputs_dir, f)
                    for f in os.listdir(nn_inputs_dir)
                    if os.path.isfile(os.path.join(nn_inputs_dir, f))
                ]
                npz_files = list(
                    filter(lambda f: os.path.splitext(f)[1] == ".npz", all_files)
                )
                files.extend(npz_files)
        elif self.is_wick:
            # in this case we want to iterate through all training folders and retrieve files from nn_inputs
            folders = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f != "genomes"
            ]

            for directory in folders:
                folder_name = os.path.basename(directory)

                if folder_name in TRAIN_DATA:
                    nn_inputs_dir = os.path.join(directory, self.numpy_dir)
                    if not os.path.isdir(nn_inputs_dir):
                        raise FileNotFoundError(f"No nn_inputs found for {folder_name}")

                    for f in os.listdir(nn_inputs_dir):
                        if (
                            os.path.isfile(os.path.join(nn_inputs_dir, f))
                            and os.path.splitext(f)[1].lower() == ".npz"
                        ):
                            files.append(os.path.join(nn_inputs_dir, f))
        else:
            for f in os.listdir(self.data_dir):
                if f.endswith(".npz"):
                    files.append(f)
            files = list(map(lambda f: os.path.join(self.data_dir, f), sorted(files)))
        return files

    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name"""
        samples_per_file = list()
        for f in self.files_list:
            metadata = read_metadata(f)
            samples_per_file.append(
                metadata[0][1][0]
            )  # [array_num, shape, first elem shape]
        return samples_per_file

    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split"""

        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        # num_validation_files = len(self.files_list) - num_train_files

        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)

        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])

        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))

        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))

        return None

    def _get_samplers(self):
        """Add samplers"""
        self.train_sampler = IdxSampler(self.train_idxs, data_source=self)
        self.validation_sampler = IdxSampler(self.validation_idxs, data_source=self)
        return None

    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it"""
        arr = np.load(self.files_list[idx])
        x = arr["x"]
        y = arr["y"]
        return self.process({"x": x, "y": y})

    def get_data(self, data_dict, idx):
        """Slices the data for given indices"""
        return {"x": data_dict["x"][idx], "y": data_dict["y"][idx]}

    def process(self, data_dict):
        """Processes the data into a ready for training format"""

        y = data_dict["y"]
        if y.dtype != "U1":
            y = y.astype("U1")
        y = self.encode(y)
        data_dict["y"] = y
        return data_dict

    def encode(self, y_arr):
        """Encode the labels"""

        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y

    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        y[y == str(self.token_pad)] = ""
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences


class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """

    def __init__(self, idxs, data_source=None):
        self.idxs = idxs
        self.data_source = data_source

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)


def read_metadata(file_name):
    """Read the metadata of a npz file

    Args:
        filename (str): .npz file that we want to read the metadata from

    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, fortran order, dtype
    """
    zip_file = zipfile.ZipFile(file_name, mode="r")
    arr_names = zip_file.namelist()

    metadata = []
    for arr_name in arr_names:
        fp = zip_file.open(arr_name, "r")
        version = np.lib.format.read_magic(fp)

        if version[0] == 1:
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
        elif version[0] == 2:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
        else:
            print("File format not detected!")
        metadata.append((arr_name, shape, fortran_order, dtype))
        fp.close()
    zip_file.close()
    return metadata
