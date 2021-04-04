from torch.utils.data import Dataset
import linecache
import os
import numpy as np
import random

from utils.process_utils import base2code_dna


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


# =====================================================================================
def parse_a_line(line, max_subreads):
    # chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
    # kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    kmer = np.array([base2code_dna[x] for x in words[5]])
    ipd_means = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
    ipd_stds = np.array([float(x) for x in words[8].split(",")], dtype=np.float)
    pw_means = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
    pw_stds = np.array([float(x) for x in words[10].split(",")], dtype=np.float)

    try:
        ipd_subs = [[float(y) for y in str(x).split(",")] for x in words[11].split(";")]
        pw_subs = [[float(y) for y in str(x).split(",")] for x in words[12].split(";")]
    except ValueError:
        ipd_subs = [[0.0]*len(kmer)]
        pw_subs = [[0.0]*len(kmer)]
    assert (len(ipd_subs) == len(pw_subs))
    sub_idxs = list(range(len(ipd_subs)))
    random.shuffle(sub_idxs)
    ipd_subs = [ipd_subs[idx] for idx in sub_idxs]
    pw_subs = [pw_subs[idx] for idx in sub_idxs]
    if len(ipd_subs) < max_subreads:
        fold = max_subreads // len(ipd_subs)
        remainder = max_subreads % len(ipd_subs)
        ipd_subs = ipd_subs * fold + ipd_subs[:remainder]
        pw_subs = pw_subs * fold + pw_subs[:remainder]
    ipd_subs = np.array(ipd_subs[:max_subreads], dtype=np.float)
    pw_subs = np.array(pw_subs[:max_subreads], dtype=np.float)

    label = int(words[13])

    return sampleinfo, kmer, ipd_means, ipd_stds, pw_means, pw_stds, ipd_subs, pw_subs, label


class FeaData(Dataset):
    def __init__(self, filename, max_subreads=5, transform=None):
        print(">>>using linecache to access '{}'<<<\n"
              ">>>after done using the file, "
              "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        self.max_subreads = max_subreads
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line(line, self.max_subreads)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


# =====================================================================================
def parse_a_line2(line, max_subreads):
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

    ipd_subs = [[float(y) for y in str(x).split(",")] for x in words[11].split(";")]
    pw_subs = [[float(y) for y in str(x).split(",")] for x in words[12].split(";")]
    assert (len(ipd_subs) == len(pw_subs))
    sub_idxs = list(range(len(ipd_subs)))
    random.shuffle(sub_idxs)
    ipd_subs = [ipd_subs[idx] for idx in sub_idxs]
    pw_subs = [pw_subs[idx] for idx in sub_idxs]
    if len(ipd_subs) < max_subreads:
        fold = max_subreads // len(ipd_subs)
        remainder = max_subreads % len(ipd_subs)
        ipd_subs = ipd_subs * fold + ipd_subs[:remainder]
        pw_subs = pw_subs * fold + pw_subs[:remainder]
    ipd_subs = np.array(ipd_subs[:max_subreads])
    pw_subs = np.array(pw_subs[:max_subreads])

    ipdmat_subs = np.zeros((max_subreads, height, width), dtype=np.float)
    pwmat_subs = np.zeros((max_subreads, height, width), dtype=np.float)
    for idx in range(max_subreads):
        ipdmat_subs[idx, np.arange(len(kmer)), kmer] = ipd_subs[idx]
        pwmat_subs[idx, np.arange(len(kmer)), kmer] = pw_subs[idx]
    mat_subs = np.concatenate((ipdmat_subs, pwmat_subs), axis=0)  # (C=2*max_subreads, H, W)

    label = int(words[13])

    return sampleinfo, kmer, mat_ccs_mean, mat_ccs_std, mat_subs, label


class FeaData2(Dataset):
    def __init__(self, filename, max_subreads=5, transform=None):
        print(">>>using linecache to access '{}'<<<\n"
              ">>>after done using the file, "
              "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        self.max_subreads = max_subreads
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            return None
        else:
            output = parse_a_line2(line, self.max_subreads)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data


# # =====================================================================================
# def parse_a_line(line):
#     # chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
#     # kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label
#     words = line.strip().split("\t")
#
#     sampleinfo = "\t".join(words[0:5])
#
#     kmer = np.array([base2code_dna[x] for x in words[5]])
#     ipd_means = np.array([float(x) for x in words[7].split(",")])
#     pw_means = np.array([float(x) for x in words[9].split(",")])
#     label = int(words[13])
#
#     return sampleinfo, kmer, ipd_means, pw_means, label
#
#
# class FeaData(Dataset):
#     def __init__(self, filename, transform=None):
#         print(">>>using linecache to access '{}'<<<\n"
#               ">>>after done using the file, "
#               "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
#         self._filename = os.path.abspath(filename)
#         self._total_data = 0
#         self._transform = transform
#         with open(filename, "r") as f:
#             self._total_data = len(f.readlines())
#
#     def __getitem__(self, idx):
#         line = linecache.getline(self._filename, idx + 1)
#         if line == "":
#             return None
#         else:
#             output = parse_a_line(line)
#             if self._transform is not None:
#                 output = self._transform(output)
#             return output
#
#     def __len__(self):
#         return self._total_data
#
#
# # =====================================================================================
# def parse_a_line3(line):
#     # chrom, abs_loc, strand, holeid, depth_all, kmer_seq, kmer_depth, \
#     # kmer_ipdm, kmer_ipds, kmer_pwm, kmer_pws, kmer_subr_ipds, kmer_subr_pws, label
#     words = line.strip().split("\t")
#
#     sampleinfo = "\t".join(words[0:5])
#     kmer = np.array([base2code_dna[x] for x in words[5]])
#     height, width = len(kmer), len(base2code_dna.keys())
#
#     ipd_means = np.array([float(x) for x in words[7].split(",")])
#     ipd_m_mat = np.zeros((1, height, width))
#     ipd_m_mat[0, np.arange(len(kmer)), kmer] = ipd_means
#
#     pw_means = np.array([float(x) for x in words[9].split(",")])
#     pw_m_mat = np.zeros((1, height, width))
#     pw_m_mat[0, np.arange(len(kmer)), kmer] = pw_means
#
#     fea_mat = np.concatenate((ipd_m_mat, pw_m_mat), axis=0)  # (C, H, W)
#
#     label = int(words[13])
#
#     return sampleinfo, fea_mat, label
#
#
# class FeaData3(Dataset):
#     def __init__(self, filename, transform=None):
#         print(">>>using linecache to access '{}'<<<\n"
#               ">>>after done using the file, "
#               "remember to use linecache.clearcache() to clear cache for safety<<<".format(filename))
#         self._filename = os.path.abspath(filename)
#         self._total_data = 0
#         self._transform = transform
#         with open(filename, "r") as f:
#             self._total_data = len(f.readlines())
#
#     def __getitem__(self, idx):
#         line = linecache.getline(self._filename, idx + 1)
#         if line == "":
#             return None
#         else:
#             output = parse_a_line3(line)
#             if self._transform is not None:
#                 output = self._transform(output)
#             return output
#
#     def __len__(self):
#         return self._total_data
