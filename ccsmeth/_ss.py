"""
strand-specific/single-strand processing
"""
import numpy as np
import torch

from .utils.constants_torch import use_cuda, FloatTensor


# for call_modifications.py ==================================================
def _call_mods1s_1time(features_onebatch, model, device):
    bkmers, bpasss, bipdms, bipdsds, bpwms, bpwsds, bsns, bmaps = features_onebatch
    _, vlogits = model(FloatTensor(bkmers, device), FloatTensor(bpasss, device),
                       FloatTensor(bipdms, device), FloatTensor(bipdsds, device),
                       FloatTensor(bpwms, device), FloatTensor(bpwsds, device),
                       FloatTensor(bsns, device), FloatTensor(bmaps, device))
    if use_cuda:
        vlogits = vlogits.cpu()
    return vlogits.data.numpy()


def _call_mods1s(features_batch, model, batch_size, device=0):
    sampleinfo, fkmers, fpasss, fipdms, fipdsds, fpwms, fpwsds, fsns, fmaps, \
        rkmers, rpasss, ripdms, ripdsds, rpwms, rpwsds, rsns, rmaps, _ = features_batch
    # labels = np.reshape(labels, (len(labels)))

    pred_info_f, pred_info_r = [], []
    batch_num = 0
    for i in np.arange(0, len(sampleinfo), batch_size):
        batch_s, batch_e = i, i + batch_size
        b_sampleinfo = sampleinfo[batch_s:batch_e]

        b_fkmers = np.array(fkmers[batch_s:batch_e])
        b_fpasss = np.array(fpasss[batch_s:batch_e])
        b_fipdms = np.array(fipdms[batch_s:batch_e])
        b_fipdsds = np.array(fipdsds[batch_s:batch_e])
        b_fpwms = np.array(fpwms[batch_s:batch_e])
        b_fpwsds = np.array(fpwsds[batch_s:batch_e])
        b_fsns = np.array(fsns[batch_s:batch_e])
        b_fmaps = np.array(fmaps[batch_s:batch_e])

        b_rkmers = np.array(rkmers[batch_s:batch_e])
        b_rpasss = np.array(rpasss[batch_s:batch_e])
        b_ripdms = np.array(ripdms[batch_s:batch_e])
        b_ripdsds = np.array(ripdsds[batch_s:batch_e])
        b_rpwms = np.array(rpwms[batch_s:batch_e])
        b_rpwsds = np.array(rpwsds[batch_s:batch_e])
        b_rsns = np.array(rsns[batch_s:batch_e])
        b_rmaps = np.array(rmaps[batch_s:batch_e])

        # b_labels = np.array(labels[batch_s:batch_e])
        if len(b_sampleinfo) > 0:
            logits_f = _call_mods1s_1time((b_fkmers, b_fpasss, b_fipdms, b_fipdsds, b_fpwms, b_fpwsds, b_fsns, b_fmaps), 
                                          model, device)
            logits_r = _call_mods1s_1time((b_rkmers, b_rpasss, b_ripdms, b_ripdsds, b_rpwms, b_rpwsds, b_rsns, b_rmaps),
                                          model, device)
            for idx in range(len(b_sampleinfo)):
                # chromosome, pos, strand, holeid, loc, depth, prob_0, prob_1, called_label, seq
                b_sampleinfo[idx] = b_sampleinfo[idx].split("\t")
                holeid = b_sampleinfo[idx][3]
                loc_f = int(b_sampleinfo[idx][4])
                loc_r = loc_f + 1  # reverse strand, not sure
                prob_0_f, prob_1_f = logits_f[idx][0], logits_f[idx][1]
                prob_0_r, prob_1_r = logits_r[idx][0], logits_r[idx][1]
                prob_1_norm_f = round(prob_1_f / (prob_0_f + prob_1_f), 6)
                prob_1_norm_r = round(prob_1_r / (prob_0_r + prob_1_r), 6)
                pred_info_f.append((holeid, loc_f, prob_1_norm_f))
                pred_info_r.append((holeid, loc_r, prob_1_norm_r))
            batch_num += 1

    return (pred_info_f, pred_info_r), batch_num
