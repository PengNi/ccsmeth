from torch.utils.data import Dataset
import linecache
import os
import numpy as np

from .utils.process_utils import base2code_dna


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


# =====================================================================================
def parse_a_line(line):
    # chrom, chrom_pos, strand, seq_name, loc,
    #     fkmer_seq, npass_fwd, fkmer_im, fkmer_isd, fkmer_pm, fkmer_psd,
    #     fkmer_sn, fkmer_map,
    #     rkmer_seq, npass_rev, rkmer_im, rkmer_isd, rkmer_pm, rkmer_psd,
    #     rkmer_sn, rkmer_map,
    #     methy_label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    fkmer = np.array([base2code_dna[x] for x in words[5]])
    fpass = np.array([int(words[6])] * len(fkmer))
    fipdm = np.array([float(x) for x in words[7].split(",")], dtype=float)
    fipdsd = np.array([float(x) for x in words[8].split(",")], dtype=float) if words[8] != "." else 0
    fpwm = np.array([float(x) for x in words[9].split(",")], dtype=float)
    fpwsd = np.array([float(x) for x in words[10].split(",")], dtype=float) if words[10] != "." else 0
    fsn = np.array([float(x) for x in words[11].split(",")], dtype=float) if words[11] != "." else 0
    fmap = np.array([float(x) for x in words[12].split(",")], dtype=float) if words[12] != "." else 0

    rkmer = np.array([base2code_dna[x] for x in words[13]])
    rpass = np.array([int(words[14])] * len(rkmer))
    ripdm = np.array([float(x) for x in words[15].split(",")], dtype=float)
    ripdsd = np.array([float(x) for x in words[16].split(",")], dtype=float) if words[16] != "." else 0
    rpwm = np.array([float(x) for x in words[17].split(",")], dtype=float)
    rpwsd = np.array([float(x) for x in words[18].split(",")], dtype=float) if words[18] != "." else 0
    rsn = np.array([float(x) for x in words[19].split(",")], dtype=float) if words[19] != "." else 0
    rmap = np.array([float(x) for x in words[20].split(",")], dtype=float) if words[20] != "." else 0

    label = int(words[21])

    return sampleinfo, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fsn, fmap, \
        rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rsn, rmap, label


class FeaData(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        # self.max_subreads = max_subreads
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass


# FeaData2 ======================================================
# ChunkDataset hasn't being accepted
# https://github.com/pytorch/pytorch/pull/26547

# https://github.com/pytorch/text/issues/130
# https://github.com/pytorch/text/blob/0b4718d7827b7f278cd3169af7f2587c1f663a27/torchtext/datasets/unsupervised_learning.py
def generate_offsets(filename):
    offsets = []
    with open(filename, "r") as rf:
        offsets.append(rf.tell())
        while rf.readline():
            offsets.append(rf.tell())
    return offsets


class FeaData2(Dataset):
    def __init__(self, filename, offsets, linenum, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = linenum
        self._transform = transform

        self._offsets = offsets
        self._data_stream = open(self._filename, 'r')
        self._current_offset = 0

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        self._data_stream.seek(offset)
        line = self._data_stream.readline()
        # with open(self._filename, "r") as rf:
        #     rf.seek(offset)
        #     line = rf.readline()
        output = parse_a_line(line)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def close(self):
        self._data_stream.close()


class FeaData3(Dataset):
    def __init__(self, filename, offsets, linenum, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = linenum
        self._transform = transform

        self._offsets = offsets
        # self._data_stream = open(self._filename, 'r')
        self._current_offset = 0

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        # self._data_stream.seek(offset)
        # line = self._data_stream.readline()
        with open(self._filename, "r") as rf:
            rf.seek(offset)
            line = rf.readline()
        output = parse_a_line(line)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass


# Dataloader for aggregate mode ==================================================================
def parse_a_line_aggre(line):
    # chrom, pos, strand, [offset_location_from_centerpos], [modprobs_histogram], [coverage], label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:3])

    offsets = np.array([int(x) for x in words[3].split(",")])
    histos = np.array([[float(y) for y in x.split(",")] for x in words[4].split(";")])
    covs = np.array([int(x) for x in words[5].split(",")])

    label = int(words[6])

    return sampleinfo, offsets, histos, covs, label


class AggreFeaData(Dataset):
    def __init__(self, filename, transform=None):
        # print(">>>using linecache to access '{}'<<<\n"
        #       ">>>after done using the file, "
        #       "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        # self.max_subreads = max_subreads
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line_aggre(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def close(self):
        pass
