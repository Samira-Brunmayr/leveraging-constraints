"""
Base dataset class that iterates over fast5 files for basecalling

Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
"""

import re
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from main.dataset_processing.normalization import (
    med_mad,
    normalize_signal_from_read_data,
)
from main.dataset_processing.read import read_fast5


class BaseFast5Dataset(Dataset):
    """Base dataset class that iterates over fast5 files for basecalling"""

    def __init__(
        self,
        data_dir=None,
        fast5_list=None,
        recursive=True,
        buffer_size=100,
        window_size=2000,
        window_overlap=400,
        trim_signal=True,
    ):
        """
        Args:
            data_dir (str): dir where the fast5 file
            fast5_list (str): file with a list of files to be processed
            recursive (bool): if the data_dir should be searched recursively
            buffer_size (int): number of fast5 files to read

        data_dir and fast5_list are esclusive
        """

        super(BaseFast5Dataset, self).__init__()

        self.data_dir = data_dir
        self.recursive = recursive
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.trim_signal = trim_signal

        if fast5_list is None:
            self.data_files = self.find_all_fast5_files()
        else:
            self.data_files = self.read_fast5_list(fast5_list)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self.process_reads(self.data_files[idx])

    def find_all_fast5_files(self):
        """Find all fast5 files in a dir recursively"""
        # find all the files that we have to process
        files_list = list()
        for path in Path(self.data_dir).rglob("*.fast5"):
            files_list.append(str(path))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def read_fast5_list(self, fast5_list):
        """Read a text file with the reads to be processed"""

        if isinstance(fast5_list, list):
            return self.buffer_list(fast5_list, self.buffer_size)

        files_list = list()
        with open(fast5_list, "r") as f:
            for line in f:
                files_list.append(line.strip("\n"))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def buffer_list(self, files_list, buffer_size):
        buffered_list = list()
        for i in range(0, len(files_list), buffer_size):
            buffered_list.append(files_list[i : i + buffer_size])
        return buffered_list

    def trim(self, signal, window_size=40, threshold_factor=2.4, min_elements=3):
        """

        from: https://github.com/nanoporetech/bonito/blob/master/bonito/fast5.py
        """

        min_trim = 10
        signal = signal[min_trim:]

        med, mad = med_mad(signal[-(window_size * 100) :])

        threshold = med + mad * threshold_factor
        num_windows = len(signal) // window_size

        seen_peak = False

        for pos in range(num_windows):
            start = pos * window_size
            end = start + window_size
            window = signal[start:end]
            if len(window[window > threshold]) > min_elements or seen_peak:
                seen_peak = True
                if window[-1] > threshold:
                    continue
                return min(end + min_trim, len(signal)), len(signal)

        return min_trim, len(signal)

    def chunk(self, signal, chunksize, overlap):
        """
        Convert a read into overlapping chunks before calling

        The first N datapoints will be cut out so that the window ends perfectly
        with the number of datapoints of the read.
        """
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)

        T = signal.shape[0]
        if chunksize == 0:
            chunks = signal[None, :]
        elif T < chunksize:
            chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
        else:
            stub = (T - overlap) % (chunksize - overlap)
            chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)

        return chunks.unsqueeze(1)

    def normalize(self, read_data):
        return normalize_signal_from_read_data(read_data)

    def process_reads(self, read_list):
        """
        Args:
            read_list (list): list of files to be processed

        Returns:
            two arrays, the first one with the normalzized chunked data,
            the second one with the read ids of each chunk.
        """
        chunks_list = list()
        id_list = list()
        l_list = list()

        for read_file in read_list:
            reads_data = read_fast5(read_file)

            for read_id in reads_data.keys():
                read_data = reads_data[read_id]
                norm_signal = self.normalize(read_data)

                if self.trim_signal:
                    trim, _ = self.trim(norm_signal[:8000])
                    norm_signal = norm_signal[trim:]

                chunks = self.chunk(norm_signal, self.window_size, self.window_overlap)
                num_chunks = chunks.shape[0]

                # synthetic data changes the read id so lets check for that also:
                match = re.search(r"!([0-9a-fA-F\-]{36})!", read_id)
                if match:
                    read_id = match.group(1)

                uuid_fields = uuid.UUID(read_id).fields
                id_arr = np.zeros((num_chunks, 6), dtype=int)
                for i, uf in enumerate(uuid_fields):
                    id_arr[:, i] = uf

                id_list.append(id_arr)
                l_list.append(np.full((num_chunks,), len(norm_signal)))
                chunks_list.append(chunks)

        out = {
            "x": torch.vstack(chunks_list).squeeze(1),
            "id": np.vstack(id_list),
            "len": np.concatenate(l_list),
        }
        return out
