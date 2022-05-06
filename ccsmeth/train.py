# -*- coding: utf-8 -*-
import argparse
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR

from .dataloader import FeaData
from .dataloader import clear_linecache

from .models import ModelAttRNN

from .utils.constants_torch import use_cuda
from .utils.process_utils import display_args
from .utils.process_utils import str2bool


def train(args):
    total_start = time.time()
    torch.manual_seed(args.tseed)
    torch.cuda.manual_seed(args.tseed)

    print("[main]train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        train_dataset = FeaData(args.train_file)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

        valid_dataset = FeaData(args.valid_file)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    else:
        raise ValueError("--model_type not right!")

    model_dir = args.model_dir
    model_regex = re.compile(r"" + args.model_type + "\.b\d+_epoch\d+\.ckpt*")
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

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

    if args.init_model is not None:
        print("loading pre-trained model: {}".format(args.init_model))
        para_dict = torch.load(args.init_model) if use_cuda else torch.load(args.init_model,
                                                                            map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)

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
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    curr_best_accuracy_loc = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            if args.model_type in {"attbigru2s", "attbilstm2s"}:
                _, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap, \
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap, \
                    label = sfeatures
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

                    label = label.cuda()
                # Forward pass
                outputs, logits = model(fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fqual, fmap,
                                        rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rqual, rmap)
                loss = criterion(outputs, label)
                tlosses.append(loss.detach().item())
            else:
                raise ValueError("--model_type not right!")

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or i == total_step - 1:
                model.eval()
                with torch.no_grad():
                    vlosses, vlabels_total, vpredicted_total = [], [], []
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
                        vlosses.append(vloss.item())
                        vlabels_total += vlabels.tolist()
                        vpredicted_total += vpredicted.tolist()

                    v_accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
                    v_precision = metrics.precision_score(vlabels_total, vpredicted_total)
                    v_recall = metrics.recall_score(vlabels_total, vpredicted_total)
                    v_meanloss = np.mean(vlosses)
                    if v_accuracy > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = v_accuracy
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type + '.b{}_epoch{}.ckpt'.format(args.seq_len,
                                                                                                epoch + 1))
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                curr_best_accuracy_loc = epoch + 1
                                no_best_model = False

                    time_cost = time.time() - start
                    print('Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; '
                          'ValidLoss: {:.4f}, '
                          'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                          'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                          .format(epoch + 1, args.max_epoch_num, i + 1, total_step, np.mean(tlosses),
                                  v_meanloss, v_accuracy, v_precision, v_recall,
                                  curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()

        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break
    endtime = time.time()
    clear_linecache()
    print("[main]train costs {} seconds, "
          "best accuracy: {} (epoch {})".format(endtime - total_start,
                                                curr_best_accuracy,
                                                curr_best_accuracy_loc))


def main():
    parser = argparse.ArgumentParser("train a model")
    st_input = parser.add_argument_group("INPUT")
    st_input.add_argument('--train_file', type=str, required=True)
    st_input.add_argument('--valid_file', type=str, required=True)

    st_output = parser.add_argument_group("OUTPUT")
    st_output.add_argument('--model_dir', type=str, required=True)

    st_train = parser.add_argument_group("TRAIN")
    # model param
    st_train.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s"],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "default: attbigru2s")
    st_train.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    st_train.add_argument('--is_npass', type=str, default="yes", required=False,
                          help="if using num_pass features, yes or no, default yes")
    st_train.add_argument('--is_qual', type=str, default="no", required=False,
                          help="if using base_quality features, yes or no, default no")
    st_train.add_argument('--is_map', type=str, default="no", required=False,
                          help="if using mapping features, yes or no, default no")
    st_train.add_argument('--is_stds', type=str, default="no", required=False,
                          help="if using std features, yes or no, default no")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiRNN model param
    st_train.add_argument('--n_vocab', type=int, default=16, required=False,
                          help="base_seq vocab_size (15 base kinds from iupac)")
    st_train.add_argument('--n_embed', type=int, default=4, required=False,
                          help="base_seq embedding_size")
    st_train.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    st_train.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size for combined feature")

    # model training
    st_train.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                             "Ranger"],
                          required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop' or 'Ranger', "
                                               "default Adam")
    st_train.add_argument('--batch_size', type=int, default=512, required=False)
    st_train.add_argument('--lr', type=float, default=0.001, required=False)
    st_train.add_argument('--lr_decay', type=float, default=0.1, required=False)
    st_train.add_argument('--lr_decay_step', type=int, default=1, required=False)
    st_train.add_argument("--max_epoch_num", action="store", default=50, type=int,
                          required=False, help="max epoch num, default 50")
    st_train.add_argument("--min_epoch_num", action="store", default=10, type=int,
                          required=False, help="min epoch num, default 10")
    st_train.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_train.add_argument('--tseed', type=int, default=1234,
                          help='random seed for pytorch')
    st_train.add_argument('--step_interval', type=int, default=500, required=False)

    st_train.add_argument('--init_model', type=str, default=None, required=False,
                          help="file path of pre-trained model parameters to load before training")

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
