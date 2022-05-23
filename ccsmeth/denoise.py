import argparse
import time
import os
import sys
import numpy as np
from sklearn import metrics

import random
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from .models import ModelAttRNN
from .dataloader import FeaData
from .dataloader import FeaData2
from .dataloader import clear_linecache

from .utils.constants_torch import use_cuda
from .utils.process_utils import str2bool
from .utils.process_utils import count_line_num
from .utils.process_utils import concat_two_files


# functions ====================================
def random_select_file_rows_s(ori_file, w_file, w_other_file, maxrownum=100000000, header=False):
    """
    split line indexs to two arrays randomly, write the two group of lines into two files,
     and return the arrays
    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    wlf = open(w_other_file, 'w')
    lidxs1, lidxs2 = [], []
    lidx_cnt = 0
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                wlf.write(next(rf))
                lidxs2.append(lidx_cnt)
                lidx_cnt += 1
            chosen_line = next(rf)
            wf.write(chosen_line)
            lidxs1.append(lidx_cnt)
            lidx_cnt += 1
    wf.close()
    wlf.close()
    print('random_select_file_rows_s finished, file1: {}, file2: {}..'.format(len(lidxs1),
                                                                              len(lidxs2)))
    return lidxs1, lidxs2


# for balancing kmer distri in training samples ===
def _count_kmers_of_feafile(feafile):
    kmer_count = {}
    kmers = set()
    with open(feafile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer_count[kmer] = 0
            kmer_count[kmer] += 1
    return kmer_count


# for balancing kmer distri in training samples ===
def _get_kmer2ratio_n_totalline(kmer_count):
    total_cnt = sum(list(kmer_count.values()))
    kmer_ratios = dict()
    for kmer in kmer_count.keys():
        kmer_ratios[kmer] = float(kmer_count[kmer])/total_cnt
    return kmer_ratios, total_cnt


# for balancing kmer distri in training samples ===
def _get_kmer2lines(feafile):
    kmer2lines = {}
    kmers = set()
    with open(feafile, "r") as rf:
        lcnt = 0
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer2lines[kmer] = []
            kmer2lines[kmer].append(lcnt)
            lcnt += 1
    return kmer2lines


# for balancing kmer distri in training samples ===
def _rand_select_by_kmer_ratio(kmer2lines, kmer2ratios, totalline):
    inter_kmers = set(kmer2lines.keys()).intersection(set(kmer2ratios.keys()))
    line_kmer_diff = set(kmer2lines.keys()).difference(set(kmer2ratios.keys()))
    ratio_kmer_diff = set(kmer2ratios.keys()).difference(set(kmer2lines.keys()))
    print("comm kmers: {}, line_kmers_diff: {}, ratio_kmers_diff: {}".format(len(inter_kmers),
                                                                             len(line_kmer_diff),
                                                                             len(ratio_kmer_diff)))
    selected_lines = []
    unselected_lines = []
    unratioed_kmers = line_kmer_diff
    cnts = 0
    for kmer in inter_kmers:
        linenum = int(math.ceil(totalline * kmer2ratios[kmer]))
        lines = kmer2lines[kmer]
        if len(lines) <= linenum:
            selected_lines += lines
            cnts += (linenum - len(lines))
        else:
            seledtmp = random.sample(lines, linenum)
            selected_lines += seledtmp
            unselected_lines += list(set(lines).difference(seledtmp))
    print("for {} common kmers, fill {} samples, "
          "{} samples that can't be filled".format(len(inter_kmers),
                                                   len(selected_lines),
                                                   cnts))
    print("for {} ratio_diff kmers, "
          "{} samples that cant't be filled".format(len(ratio_kmer_diff),
                                                    sum([round(totalline * kmer2ratios[kmer])
                                                         for kmer in ratio_kmer_diff])))
    unfilled_cnt = totalline - len(selected_lines)
    print("totalline: {}, need to fill: {}".format(totalline, unfilled_cnt))
    if unfilled_cnt > 0 and len(unratioed_kmers) > 0:
        minlinenum = int(math.ceil(float(unfilled_cnt)/len(unratioed_kmers)))
        cnts = 0
        for kmer in unratioed_kmers:
            lines = kmer2lines[kmer]
            if len(lines) <= minlinenum:
                selected_lines += lines
                cnts += len(lines)
            else:
                seledtmp = random.sample(lines, minlinenum)
                selected_lines += seledtmp
                cnts += minlinenum
                unselected_lines += list(set(lines).difference(seledtmp))
        print("extract {} samples from {} line_diff kmers".format(cnts, len(unratioed_kmers)))
    unfilled_cnt = totalline - len(selected_lines)
    if unfilled_cnt > 0:
        print("totalline: {}, still need to fill: {}".format(totalline, unfilled_cnt))
        random.shuffle(unselected_lines)
        triplefill_cnt = unfilled_cnt
        if len(unselected_lines) <= unfilled_cnt:
            selected_lines += unselected_lines
            triplefill_cnt = len(unselected_lines)
        else:
            selected_lines += unselected_lines[:unfilled_cnt]
        print("extract {} samples from {} samples not used above".format(triplefill_cnt, len(unselected_lines)))
    selected_lines = sorted(selected_lines)
    selected_lines = [-1] + selected_lines
    return selected_lines


# for balancing kmer distri in training samples ===
def _write_randsel_lines(feafile, wfile, seled_lines):
    wf = open(wfile, 'w')
    with open(feafile) as rf:
        for i in range(1, len(seled_lines)):
            chosen_line = ''
            for j in range(0, seled_lines[i] - seled_lines[i - 1]):
                # print(j)
                chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    print('_write_randsel_lines finished..')


# balance kmer distri in neg_training file as pos_training file
def select_negsamples_asposkmer(pos_file, totalneg_file, seled_neg_file):
    kmer_count = _count_kmers_of_feafile(pos_file)
    kmer2ratio, totalline = _get_kmer2ratio_n_totalline(kmer_count)

    print("{} kmers from kmer2ratio file: {}".format(len(kmer2ratio), pos_file))
    kmer2lines = _get_kmer2lines(totalneg_file)
    sel_lines = _rand_select_by_kmer_ratio(kmer2lines, kmer2ratio, totalline)
    _write_randsel_lines(totalneg_file, seled_neg_file, sel_lines)


# get model type main params
def get_model_type_str(model_type):
    return str(model_type)
# ==============================================


def train_1time(train_file, valid_file, valid_lidxs, args):
    """
    use train_file to train model, then score the probs of the samples in valid_file
    :param train_file:
    :param valid_file:
    :param valid_lidxs:
    :param args:
    :return:
    """
    # ===========
    if args.dl_offsets:
        if args.dl_num_workers > 1:
            raise ValueError("--dl_num_workers should not be >1 when --dl_offsets is True!")
        from .utils.process_utils import count_line_num
        from .dataloader import generate_offsets
        train_linenum = count_line_num(train_file, False)
        train_offsets = generate_offsets(train_file)
        train_dataset = FeaData2(train_file, train_offsets, train_linenum)
    else:
        train_dataset = FeaData(train_file)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.dl_num_workers)

    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            is_qual=str2bool(args.is_qual),
                            is_map=str2bool(args.is_map),
                            is_stds=str2bool(args.is_stds),
                            is_npass=str2bool(args.is_npass),
                            model_type=args.model_type)
    else:
        raise ValueError("--model_type not right!")

    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    elif args.optim_type == "Ranger":
        # use Ranger optimizer
        # refer to https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
        # needs python>=3.6
        try:
            from utils.ranger2020 import Ranger
        except ImportError:
            raise ImportError("please check if ranger2020.py is in utils/ dir!")
        optimizer = Ranger(model.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-5)
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    # Train the model
    total_step = len(train_loader)
    print("train total_step: {}".format(total_step))
    start = time.time()
    model.train()
    for epoch in range(args.epoch_num):
        test_accus = []
        for i, sfeatures in enumerate(train_loader):
            if args.model_type in {"attbigru2s", "attbilstm2s"}:
                _, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                    labels = sfeatures
                if use_cuda:
                    fkmer = fkmer.cuda()
                    fpass = fpass.cuda()
                    fipdm = fipdm.cuda()
                    fipdsd = fipdsd.cuda()
                    fpwm = fpwm.cuda()
                    fpwsd = fpwsd.cuda()
                    fqual = fqual.cuda()
                    fmap = fmap.cuda()

                    rkmer = rkmer.cuda()
                    rpass = rpass.cuda()
                    ripdm = ripdm.cuda()
                    ripdsd = ripdsd.cuda()
                    rpwm = rpwm.cuda()
                    rpwsd = rpwsd.cuda()
                    rqual = rqual.cuda()
                    rmap = rmap.cuda()

                    labels = labels.cuda()
                # Forward pass
                outputs, tlogits = model(fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap,
                                         rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap)
                loss = criterion(outputs, labels)
            else:
                raise ValueError("--model_type not right!")

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                _, tpredicted = torch.max(tlogits.data, 1)

                tlabels = labels
                if use_cuda:
                    tlabels = labels.cpu()
                    tpredicted = tpredicted.cpu()
                i_accuracy = metrics.accuracy_score(tlabels.numpy(), tpredicted)
                i_precision = metrics.precision_score(tlabels.numpy(), tpredicted)
                i_recall = metrics.recall_score(tlabels.numpy(), tpredicted)

                test_accus.append(i_accuracy)

                endtime = time.time()
                print('Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}, '
                      'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                      'Time: {:.2f}s'
                      .format(epoch + 1, args.epoch_num, i + 1, total_step, loss.item(),
                              i_accuracy, i_precision, i_recall, endtime - start))
                sys.stdout.flush()
                start = time.time()
        scheduler.step()
        if np.mean(test_accus) >= 0.95:
            break
    train_dataset.close()
    clear_linecache()

    # valid data
    if args.dl_offsets:
        if args.dl_num_workers > 1:
            raise ValueError("--dl_num_workers should not be >1 when --dl_offsets is True!")
        from .utils.process_utils import count_line_num
        from .dataloader import generate_offsets
        valid_linenum = count_line_num(valid_file, False)
        valid_offsets = generate_offsets(valid_file)
        valid_dataset = FeaData2(valid_file, valid_offsets, valid_linenum)
    else:
        valid_dataset = FeaData(valid_file)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.dl_num_workers)
    total_step = len(valid_loader)
    print("valid total_step: {}".format(total_step))
    model.eval()
    vlosses, vaccus, vprecs, vrecas = [], [], [], []
    lineidx_cnt = 0
    idx2aclogits = {}
    start = time.time()
    for vi, vsfeatures in enumerate(valid_loader):
        if args.model_type in {"attbigru2s", "attbilstm2s"}:
            _, vfkmer, vfpass, vfipdm, vfipdsd, vfpwm, vfpwsd, vfqual, vfmap, \
                vrkmer, vrpass, vripdm, vripdsd, vrpwm, vrpwsd, vrqual, vrmap, \
                vlabels = vsfeatures
            if use_cuda:
                vfkmer = vfkmer.cuda()
                vfpass = vfpass.cuda()
                vfipdm = vfipdm.cuda()
                vfipdsd = vfipdsd.cuda()
                vfpwm = vfpwm.cuda()
                vfpwsd = vfpwsd.cuda()
                vfqual = vfqual.cuda()
                vfmap = vfmap.cuda()

                vrkmer = vrkmer.cuda()
                vrpass = vrpass.cuda()
                vripdm = vripdm.cuda()
                vripdsd = vripdsd.cuda()
                vrpwm = vrpwm.cuda()
                vrpwsd = vrpwsd.cuda()
                vrqual = vrqual.cuda()
                vrmap = vrmap.cuda()

                vlabels = vlabels.cuda()
            # Forward pass
            voutputs, vlogits = model(vfkmer, vfpass, vfipdm, vfipdsd, vfpwm, vfpwsd, vfqual, vfmap,
                                      vrkmer, vrpass, vripdm, vripdsd, vrpwm, vrpwsd, vrqual, vrmap)
            vloss = criterion(voutputs, vlabels)
        else:
            raise ValueError("--model_type not right!")

        _, vpredicted = torch.max(vlogits.data, 1)

        if use_cuda:
            vlabels = vlabels.cpu()
            vpredicted = vpredicted.cpu()
            vlogits = vlogits.cpu()

        for alogit in vlogits.detach().numpy():
            idx2aclogits[valid_lidxs[lineidx_cnt]] = alogit[1]
            lineidx_cnt += 1

        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
        i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

        vaccus.append(i_accuracy)
        vprecs.append(i_precision)
        vrecas.append(i_recall)

        if (vi + 1) % args.step_interval == 0 or (vi + 1) == total_step:
            endtime = time.time()
            print('===Test, Step [{}/{}], ValidLoss: {:.4f}, '
                  'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                  'Time: {:.2f}s'
                  .format(vi+1, total_step, vloss.item(), i_accuracy, i_precision, i_recall, endtime - start))
            sys.stdout.flush()
            start = time.time()
    valid_dataset.close()
    clear_linecache()
    print("===Test, Total Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(np.mean(vaccus),
                                                                                      np.mean(vprecs),
                                                                                      np.mean(vrecas)))
    del model
    return idx2aclogits


def train_rounds(train_file, iterstr, args, modeltype_str):
    """
    repeat rounds of splitting train_file to train_then_valid
    :param train_file:
    :param iterstr:
    :param args:
    :param modeltype_str:
    :return:
    """
    print("\n##########Train Cross Rank##########")
    total_num = count_line_num(train_file, False)
    half_num = total_num // 2
    fname, fext = os.path.splitext(train_file)
    idxs2logtis_all = {}
    for i in range(0, total_num):
        idxs2logtis_all[i] = []

    for i in range(0, args.rounds):
        print("##########Train Cross Rank, Iter {}, Round {}##########".format(iterstr, i+1))
        if train_file == args.train_file:
            train_file1 = fname + "." + modeltype_str + ".half1" + fext
            train_file2 = fname + "." + modeltype_str + ".half2" + fext
        else:
            train_file1 = fname + ".half1" + fext
            train_file2 = fname + ".half2" + fext
        lidxs1, lidxs2 = random_select_file_rows_s(train_file, train_file1, train_file2,
                                                   half_num, False)
        print("##########Train Cross Rank, Iter {}, Round {}, part1##########".format(iterstr, i + 1))
        idxs22logits = train_1time(train_file1, train_file2, lidxs2, args)
        print("##########Train Cross Rank, Iter {}, Round {}, part2##########".format(iterstr, i + 1))
        idxs12logits = train_1time(train_file2, train_file1, lidxs1, args)
        for idx in idxs22logits.keys():
            idxs2logtis_all[idx].append(idxs22logits[idx])
        for idx in idxs12logits.keys():
            idxs2logtis_all[idx].append(idxs12logits[idx])

        os.remove(train_file1)
        os.remove(train_file2)
    print("##########Train Cross Rank, finished!##########")
    sys.stdout.flush()
    return idxs2logtis_all


def clean_samples(train_file, idx2logits, score_cf, is_filter_fn, ori_train_file, modeltype_str):
    # clean train samples ===
    print("\n###### clean the samples ######")
    idx2probs = dict()
    for idx in idx2logits.keys():
        probs = idx2logits[idx]
        meanprob = np.mean(probs)
        stdprob = np.std(probs)
        idx2probs[idx] = [meanprob, stdprob]

    idx2prob_pos, idx2prob_neg = [], []
    with open(train_file, 'r') as rf:
        linecnt = 0
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 1:
                idx2prob_pos.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            else:
                idx2prob_neg.append((linecnt, idx2probs[linecnt][0], idx2probs[linecnt][1]))
            linecnt += 1

    print("There are {} positive, {} negative samples in total;".format(len(idx2prob_pos),
                                                                        len(idx2prob_neg)))

    pos_hc, neg_hc = set(), set()

    idx2prob_pos = sorted(idx2prob_pos, key=lambda x: x[1], reverse=True)
    for idx2prob in idx2prob_pos:
        if idx2prob[1] >= score_cf:
            pos_hc.add(idx2prob[0])
    if is_filter_fn:
        idx2prob_neg = sorted(idx2prob_neg, key=lambda x: x[1])
        for idx2prob in idx2prob_neg:
            if idx2prob[1] < 1 - score_cf:
                neg_hc.add(idx2prob[0])

    left_ratio = float(len(pos_hc)) / len(idx2prob_pos) if len(idx2prob_pos) > 0 else 0
    left_ratio2 = float(len(neg_hc)) / len(idx2prob_neg) if len(idx2prob_neg) > 0 else 0
    print("{} ({}) high quality positive samples left, "
          "{} ({}) high quality negative samples left".format(len(pos_hc),
                                                              round(left_ratio, 6),
                                                              len(neg_hc),
                                                              round(left_ratio2, 6)))

    # re-write train set
    fname, fext = os.path.splitext(train_file)
    if train_file == ori_train_file:
        train_clean_pos_file = fname + "." + modeltype_str + ".pos.cf" + str(score_cf) + fext
    else:
        train_clean_pos_file = fname + ".pos.cf" + str(score_cf) + fext
    wfp = open(train_clean_pos_file, 'w')
    if is_filter_fn:
        if train_file == ori_train_file:
            train_clean_neg_file = fname + "." + modeltype_str + ".neg.cf" + str(score_cf) + fext
        else:
            train_clean_neg_file = fname + ".neg.cf" + str(score_cf) + fext
        wfn = open(train_clean_neg_file, 'w')
    lidx = 0
    with open(train_file, 'r') as rf:
        for line in rf:
            if lidx in pos_hc:
                wfp.write(line)
            elif is_filter_fn and lidx in neg_hc:
                wfn.write(line)
            lidx += 1
    wfp.close()
    if is_filter_fn:
        wfn.close()

    print("###### clean the samples, finished! ######")
    sys.stdout.flush()

    if is_filter_fn:
        left_ratio = (left_ratio + left_ratio2) / 2
        return train_clean_pos_file, left_ratio, train_clean_neg_file
    else:
        return train_clean_pos_file, left_ratio, None


def _get_all_negative_samples(train_file, modeltype_str):
    fname, fext = os.path.splitext(train_file)
    train_neg_file = fname + ".neg_all" + "." + modeltype_str + fext

    wf = open(train_neg_file, "w")
    with open(train_file, 'r') as rf:
        for line in rf:
            words = line.strip().split("\t")
            label = int(words[-1])
            if label == 0:
                wf.write(line)
    wf.close()
    return train_neg_file


def _output_linenumber2probs(wfile, idx2logits):
    wf = open(wfile, "w")
    for idx in sorted(list(idx2logits.keys())):
        wf.write("\t".join([str(idx), str(np.mean(idx2logits[idx]))]) + "\n")
    wf.close()


def denoise(args):
    print("[main]denoise starts..")
    total_start = time.time()
    torch.manual_seed(args.tseed)
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    iterations = args.iterations

    train_file = args.train_file
    modeltype_str = get_model_type_str(args.model_type)
    modeltype_str += "_{}".format(args.job_id)

    # filter neg samples ===
    is_filter_fn = str2bool(args.is_filter_fn)
    if not is_filter_fn:
        train_neg_file = _get_all_negative_samples(train_file, modeltype_str)

    for iter_c in range(iterations):
        print("\n###### cross rank to clean samples, Iter: {} ######".format(iter_c + 1))
        # cross rank
        iterstr = str(iter_c + 1)
        idxs2logtis_all = train_rounds(train_file, iterstr, args, modeltype_str)

        # output probs of 1 iteration
        if iter_c == 0 and args.fst_iter_prob:
            wfile = train_file + ".probs_1stiter.txt"
            _output_linenumber2probs(wfile, idxs2logtis_all)

        train_clean_pos_file, left_ratio, train_clean_neg_file = clean_samples(train_file, idxs2logtis_all,
                                                                               args.score_cf, is_filter_fn,
                                                                               args.train_file, modeltype_str)
        if train_file != args.train_file:
            os.remove(train_file)

        # concat new train_file
        print("\n#####concat denoied file#####")
        pos_num = count_line_num(train_clean_pos_file)
        if pos_num > 0:
            if train_clean_neg_file is None:
                fname, fext = os.path.splitext(train_neg_file)
                train_seled_neg_file = fname + ".r" + str(pos_num) + fext
                select_negsamples_asposkmer(train_clean_pos_file, train_neg_file, train_seled_neg_file)
            else:
                neg_num = count_line_num(train_clean_neg_file)
                fname, fext = os.path.splitext(train_clean_neg_file)
                train_seled_neg_file = fname + ".r" + str(pos_num) + fext
                if pos_num <= neg_num:
                    select_negsamples_asposkmer(train_clean_pos_file, train_clean_neg_file, train_seled_neg_file)
                    os.remove(train_clean_neg_file)
                else:
                    train_seled_neg_file = train_clean_neg_file

            fname, fext = os.path.splitext(args.train_file)
            if is_filter_fn:
                train_file = fname + "." + modeltype_str + ".cf{}".format(args.score_cf) + \
                             ".denoise_fpnp" + str(iter_c + 1) + fext
            else:
                train_file = fname + "." + modeltype_str + ".cf{}".format(args.score_cf) + \
                             ".denoise_fp" + str(iter_c + 1) + fext
            concat_two_files(train_clean_pos_file, train_seled_neg_file, concated_fp=train_file)
            os.remove(train_seled_neg_file)
        else:
            if train_clean_neg_file is not None:
                os.remove(train_clean_neg_file)
            print("WARING: The denoise module denoised all samples in the train_file!!!")
        os.remove(train_clean_pos_file)
        print("#####concat denoied file, finished!#####")

        if left_ratio >= args.kept_ratio or pos_num == 0:
            break

    if os.path.exists(train_neg_file):
        os.remove(train_neg_file)
    total_end = time.time()
    print("###### denoised file for training: {}".format(train_file))
    print("[main]denoise costs {:.2f} seconds".format(total_end - total_start))


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


def main():
    parser = argparse.ArgumentParser("train cross rank, filter false positive samples (and "
                                     "false negative samples).")

    sd_input = parser.add_argument_group("INPUT")
    sd_input.add_argument('--train_file', type=str, required=True, help="file containing (combined positive and "
                                                                        "negative) samples for training. better been "
                                                                        "balanced in kmer level.")
    sd_input.add_argument('--job_id', type=str, required=False, default="djob", help="job uid")

    sd_denoise = parser.add_argument_group("DENOISE")
    sd_denoise.add_argument('--is_filter_fn', type=str, default="yes", required=False,
                            help="is filter false negative samples, 'yes' or 'no', default yes")
    sd_denoise.add_argument('--iterations', type=int, default=10, required=False)
    sd_denoise.add_argument('--rounds', type=int, default=3, required=False)
    sd_denoise.add_argument("--score_cf", type=float, default=0.2,
                            required=False,
                            help="score cutoff to keep high quality (which prob>=score_cf) positive samples. "
                                 "(0, 0.5], default 0.2")
    sd_denoise.add_argument("--kept_ratio", type=float, default=0.99,
                            required=False,
                            help="kept ratio of samples, to end denoise process. default 0.99")
    sd_denoise.add_argument("--fst_iter_prob", action="store_true", default=False,
                            help="if output probs of samples after 1st iteration")

    sd_train = parser.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    sd_train.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s"],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "default: attbigru2s")
    sd_train.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    sd_train.add_argument('--is_npass', type=str, default="yes", required=False,
                          help="if using num_pass features, yes or no, default yes")
    sd_train.add_argument('--is_qual', type=str, default="no", required=False,
                          help="if using base_quality features, yes or no, default no")
    sd_train.add_argument('--is_map', type=str, default="no", required=False,
                          help="if using mapping features, yes or no, default no")
    sd_train.add_argument('--is_stds', type=str, default="no", required=False,
                          help="if using std features, yes or no, default no")
    sd_train.add_argument('--class_num', type=int, default=2, required=False)
    sd_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiRNN model param
    sd_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    sd_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")
    sd_train.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    sd_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size for combined feature")

    sd_training = parser.add_argument_group("TRAINING")
    # model training
    sd_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger"],
                             required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop' "
                                                  "or 'Ranger', default Adam")
    sd_training.add_argument('--batch_size', type=int, default=512, required=False)
    # sd_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
    #                          choices=["StepLR", "ReduceLROnPlateau"],
    #                          help="StepLR or ReduceLROnPlateau, default StepLR")
    sd_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001")
    sd_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    sd_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    # sd_training.add_argument('--lr_patience', type=int, default=0, required=False,
    #                          help="effective in ReduceLROnPlateau. default 0")
    # sd_training.add_argument('--lr_mode_strategy', type=str, default="last", required=False,
    #                          choices=["last", "mean", "max"],
    #                          help="effective in ReduceLROnPlateau. last, mean, or max, default last")
    sd_training.add_argument("--epoch_num", action="store", default=2, type=int,
                             required=False, help="epoch num, default 2")
    sd_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    sd_training.add_argument('--step_interval', type=int, default=500, required=False)
    sd_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")
    sd_training.add_argument('--dl_offsets', action="store_true", default=False, required=False,
                             help="use file offsets loader")

    sd_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    sd_training.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)
    denoise(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime-total_start))


if __name__ == '__main__':
    main()
