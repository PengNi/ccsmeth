from torch.utils.data import Dataset
import linecache
import os
import numpy as np
# import random

from utils.process_utils import base2code_dna


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


# =====================================================================================
def parse_a_line(line):
    # chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
    # kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    kmer = np.array([base2code_dna[x] for x in words[5]])
    ipd_means = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
    ipd_stds = np.array([float(x) for x in words[8].split(",")], dtype=np.float)
    pw_means = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
    pw_stds = np.array([float(x) for x in words[10].split(",")], dtype=np.float)

    label = int(words[13])

    # return sampleinfo, kmer, ipd_means, ipd_stds, pw_means, pw_stds, ipd_subs, pw_subs, label
    return sampleinfo, kmer, ipd_means, ipd_stds, pw_means, pw_stds, label


class FeaData(Dataset):
    def __init__(self, filename, transform=None):
        print(">>>using linecache to access '{}'<<<\n"
              ">>>after done using the file, "
              "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
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


# =====================================================================================
def parse_a_line2s(line):
    # chrom, abs_loc, strand, holeid, depth_all,
    # kmer_seq, kmer_depth, kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws,
    # kmer_seq2, kmer_depth2, kmer_ipdm2, kmer_ipds2, kmer_pwm2, kmer_pws2, kmer_subr_ipds2, kmer_subr_pws2,
    # label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    kmer = np.array([base2code_dna[x] for x in words[5]])
    ipd_means = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
    ipd_stds = np.array([float(x) for x in words[8].split(",")], dtype=np.float)
    pw_means = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
    pw_stds = np.array([float(x) for x in words[10].split(",")], dtype=np.float)

    kmer2 = np.array([base2code_dna[x] for x in words[13]])
    ipd_means2 = np.array([float(x) for x in words[15].split(",")], dtype=np.float)
    ipd_stds2 = np.array([float(x) for x in words[16].split(",")], dtype=np.float)
    pw_means2 = np.array([float(x) for x in words[17].split(",")], dtype=np.float)
    pw_stds2 = np.array([float(x) for x in words[18].split(",")], dtype=np.float)

    label = int(words[21])

    return sampleinfo, kmer, ipd_means, ipd_stds, pw_means, pw_stds, \
        kmer2, ipd_means2, ipd_stds2, pw_means2, pw_stds2, \
        label


class FeaData2s(Dataset):
    def __init__(self, filename, transform=None):
        print(">>>using linecache to access '{}'<<<\n"
              ">>>after done using the file, "
              "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
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
            output = parse_a_line2s(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


# =====================================================================================
def parse_a_line2(line):
    # chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
    # kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    kmer = np.array([base2code_dna[x] for x in words[5]])
    height, width = len(kmer), len(base2code_dna.keys())

    ipd_means = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
    ipd_m_mat = np.zeros((1, height, width), dtype=np.float)
    ipd_m_mat[0, np.arange(len(kmer)), kmer] = ipd_means
    pw_means = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
    pw_m_mat = np.zeros((1, height, width), dtype=np.float)
    pw_m_mat[0, np.arange(len(kmer)), kmer] = pw_means
    mat_ccs_mean = np.concatenate((ipd_m_mat, pw_m_mat), axis=0)  # (C=2, H, W)

    ipd_stds = np.array([float(x) for x in words[8].split(",")], dtype=np.float)
    ipd_s_mat = np.zeros((1, height, width), dtype=np.float)
    ipd_s_mat[0, np.arange(len(kmer)), kmer] = ipd_stds
    pw_stds = np.array([float(x) for x in words[10].split(",")], dtype=np.float)
    pw_s_mat = np.zeros((1, height, width), dtype=np.float)
    pw_s_mat[0, np.arange(len(kmer)), kmer] = pw_stds
    mat_ccs_std = np.concatenate((ipd_s_mat, pw_s_mat), axis=0)  # (C=2, H, W)

    label = int(words[13])

    # return sampleinfo, kmer, mat_ccs_mean, mat_ccs_std, mat_subs, label
    return sampleinfo, kmer, mat_ccs_mean, mat_ccs_std, label


class FeaData2(Dataset):
    def __init__(self, filename, transform=None):
        print(">>>using linecache to access '{}'<<<\n"
              ">>>after done using the file, "
              "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
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
            output = parse_a_line2(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data
