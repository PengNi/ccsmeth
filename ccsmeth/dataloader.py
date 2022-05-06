from torch.utils.data import Dataset
import linecache
import os
import numpy as np
# import random

from .utils.process_utils import base2code_dna


def clear_linecache():
    # linecache should be treated carefully
    linecache.clearcache()


# =====================================================================================
def parse_a_line(line):
    # chrom, chrom_pos, strand, seq_name, loc,
    #     fkmer_seq, npass_fwd, fkmer_im, fkmer_isd, fkmer_pm, fkmer_psd,
    #     fkmer_qual, fkmer_map,
    #     rkmer_seq, npass_rev, rkmer_im, rkmer_isd, rkmer_pm, rkmer_psd,
    #     rkmer_qual, rkmer_map,
    #     methy_label
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:5])

    fkmer = np.array([base2code_dna[x] for x in words[5]])
    fpass = np.array([int(words[6])] * len(fkmer))
    fipdm = np.array([float(x) for x in words[7].split(",")], dtype=np.float)
    fipdsd = np.array([float(x) for x in words[8].split(",")], dtype=np.float) if words[8] != "." else 0
    fpwm = np.array([float(x) for x in words[9].split(",")], dtype=np.float)
    fpwsd = np.array([float(x) for x in words[10].split(",")], dtype=np.float) if words[10] != "." else 0
    fqual = np.array([float(x) for x in words[11].split(",")], dtype=np.float)
    fmap = np.array([float(x) for x in words[12].split(",")], dtype=np.float) if words[12] != "." else 0

    rkmer = np.array([base2code_dna[x] for x in words[13]])
    rpass = np.array([int(words[14])] * len(rkmer))
    ripdm = np.array([float(x) for x in words[15].split(",")], dtype=np.float)
    ripdsd = np.array([float(x) for x in words[16].split(",")], dtype=np.float) if words[16] != "." else 0
    rpwm = np.array([float(x) for x in words[17].split(",")], dtype=np.float)
    rpwsd = np.array([float(x) for x in words[18].split(",")], dtype=np.float) if words[18] != "." else 0
    rqual = np.array([float(x) for x in words[19].split(",")], dtype=np.float)
    rmap = np.array([float(x) for x in words[20].split(",")], dtype=np.float) if words[20] != "." else 0

    label = int(words[21])

    return sampleinfo, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
        rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, label


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
