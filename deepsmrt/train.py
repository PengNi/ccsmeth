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

from dataloader import FeaData
from dataloader import FeaData2
from dataloader import clear_linecache

from models import ModelBiLSTM
from models import ModelResNet18

from utils.constants_torch import use_cuda
from utils.process_utils import display_args
from utils.process_utils import str2bool


def train(args):
    total_start = time.time()
    # torch.manual_seed(args.seed)

    print("[train]start..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")
    if args.model_type in {"bilstm", }:
        train_dataset = FeaData(args.train_file, max_subreads=args.num_subreads)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

        valid_dataset = FeaData(args.valid_file, max_subreads=args.num_subreads)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    elif args.model_type in {"resnet18", }:
        train_dataset = FeaData2(args.train_file, max_subreads=args.num_subreads)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

        valid_dataset = FeaData2(args.valid_file, max_subreads=args.num_subreads)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    else:
        raise ValueError("model_type not right!")

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

    if args.model_type == "bilstm":
        model = ModelBiLSTM(args.seq_len, args.layernum, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            args.num_subreads,
                            is_ccs=str2bool(args.is_ccs), is_stds=str2bool(args.is_stds),
                            is_subreads=str2bool(args.is_subreads),
                            is_kmer=str2bool(args.is_kmer))
    elif args.model_type == "resnet18":
        model = ModelResNet18(args.class_num, args.dropout_rate, args.num_subreads,
                              str2bool(args.is_ccs), str2bool(args.is_stds), str2bool(args.is_subreads))
    else:
        raise ValueError("model_type not right!")

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
    else:
        raise ValueError("optim_type is not right!")
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    print("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            if args.model_type in {"bilstm", }:
                _, kmer, ipd_means, ipd_stds, pw_means, pw_stds, ipd_subs, pw_subs, labels = sfeatures
                if use_cuda:
                    kmer = kmer.cuda()
                    ipd_means = ipd_means.cuda()
                    ipd_stds = ipd_stds.cuda()
                    pw_means = pw_means.cuda()
                    pw_stds = pw_stds.cuda()
                    ipd_subs = ipd_subs.cuda()
                    pw_subs = pw_subs.cuda()
                    labels = labels.cuda()
                # Forward pass
                outputs, logits = model(kmer, ipd_means, ipd_stds, pw_means, pw_stds, ipd_subs, pw_subs)
                loss = criterion(outputs, labels)
                tlosses.append(loss.detach().item())
            elif args.model_type in {"resnet18", }:
                _, _, mats_ccs_mean, mats_ccs_std, mats_subs, labels = sfeatures
                if use_cuda:
                    mats_ccs_mean = mats_ccs_mean.cuda()
                    mats_ccs_std = mats_ccs_std.cuda()
                    mats_subs = mats_subs.cuda()
                    labels = labels.cuda()
                # Forward pass
                outputs, logits = model(mats_ccs_mean, mats_ccs_std, mats_subs)
                loss = criterion(outputs, labels)
                tlosses.append(loss.detach().item())
            else:
                raise ValueError("model_type not right!")

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0:
                model.eval()
                with torch.no_grad():
                    vlosses, vaccus, vprecs, vrecas = [], [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        if args.model_type in {"bilstm", }:
                            _, vkmer, vipd_means, vipd_stds, vpw_means, vpw_stds, \
                                vipd_subs, vpw_subs, vlabels = vsfeatures
                            if use_cuda:
                                vkmer = vkmer.cuda()
                                vipd_means = vipd_means.cuda()
                                vipd_stds = vipd_stds.cuda()
                                vpw_means = vpw_means.cuda()
                                vpw_stds = vpw_stds.cuda()
                                vipd_subs = vipd_subs.cuda()
                                vpw_subs = vpw_subs.cuda()
                                vlabels = vlabels.cuda()
                            voutputs, vlogits = model(vkmer, vipd_means, vipd_stds, vpw_means, vpw_stds,
                                                      vipd_subs, vpw_subs)
                            vloss = criterion(voutputs, vlabels)
                        elif args.model_type in {"resnet18", }:
                            _, _, vmats_ccs_mean, vmats_ccs_std, vmats_subs, vlabels = vsfeatures
                            if use_cuda:
                                vmats_ccs_mean = vmats_ccs_mean.cuda()
                                vmats_ccs_std = vmats_ccs_std.cuda()
                                vmats_subs = vmats_subs.cuda()
                                vlabels = vlabels.cuda()
                            # Forward pass
                            voutputs, vlogits = model(vmats_ccs_mean, vmats_ccs_std, vmats_subs)
                            vloss = criterion(voutputs, vlabels)
                        else:
                            raise ValueError("model_type not right!")

                        _, vpredicted = torch.max(vlogits.data, 1)

                        if use_cuda:
                            vlabels = vlabels.cpu()
                            vpredicted = vpredicted.cpu()
                        i_accuracy = metrics.accuracy_score(vlabels.numpy(), vpredicted)
                        i_precision = metrics.precision_score(vlabels.numpy(), vpredicted)
                        i_recall = metrics.recall_score(vlabels.numpy(), vpredicted)

                        vaccus.append(i_accuracy)
                        vprecs.append(i_precision)
                        vrecas.append(i_recall)
                        vlosses.append(vloss.item())

                    if np.mean(vaccus) > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = np.mean(vaccus)
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.001:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type + '.b{}_epoch{}.ckpt'.format(args.seq_len,
                                                                                                epoch))

                    time_cost = time.time() - start
                    print('Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; '
                          'ValidLoss: {:.4f}, '
                          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, '
                          'curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s'
                          .format(epoch + 1, args.max_epoch_num, i + 1, total_step, np.mean(tlosses),
                                  np.mean(vlosses), np.mean(vaccus), np.mean(vprecs), np.mean(vrecas),
                                  curr_best_accuracy_epoch, time_cost))
                    tlosses = []
                    start = time.time()
                    sys.stdout.flush()
                model.train()
        scheduler.step()
        if curr_best_accuracy_epoch > curr_best_accuracy:
            curr_best_accuracy = curr_best_accuracy_epoch
        else:
            if epoch >= args.min_epoch_num - 1:
                print("best accuracy: {}, early stop!".format(curr_best_accuracy))
                break

    endtime = time.time()
    clear_linecache()
    print("[train]training cost {} seconds".format(endtime - total_start))


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    # model input
    parser.add_argument('--model_type', type=str, default="bilstm",
                        choices=["bilstm", "resnet18"],
                        required=False,
                        help="type of model to use, 'bilstm', 'resnet18',"
                             " default: bilstm")
    parser.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")

    # model param
    parser.add_argument('--is_ccs', type=str, default="yes", required=False,
                        help="if using features at ccs level, yes or no. default yes.")
    parser.add_argument('--is_stds', type=str, default="yes", required=False,
                        help="if using std features at ccs level, yes or no. default yes.")
    parser.add_argument('--is_subreads', type=str, default="yes", required=False,
                        help="if using features at subreads level, yes or no. default yes.")
    parser.add_argument("--num_subreads", type=int, default=5, required=False,
                        help="info of num of subreads to be used for training, default 5")
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiLSTM model param
    parser.add_argument('--is_kmer', type=str, default="yes", required=False,
                        help="if combiniing kmer features at subreads level, yes or no. default yes.")
    parser.add_argument('--layernum', type=int, default=3,
                        required=False, help="lstm layer num, default 3")
    parser.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size for combined feature")
    parser.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    parser.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")

    # ResNet model param
    parser.add_argument('--init_channels', type=int, default=2, required=False,
                        help="resnet18, input channels")

    # # transformer model param
    # parser.add_argument('--layernum_trans', type=int, default=6,
    #                     required=False, help="trans encoder layer num, default 6")
    # parser.add_argument('--d_model', type=int, default=256, required=False)
    # parser.add_argument('--hid_trans', type=int, default=1024, required=False,
    #                     help="transfomer encoder hidden size")
    # parser.add_argument('--n_head', type=int, default=4, required=False)

    # model training
    parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD"],
                        required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop', default Adam")
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument("--max_epoch_num", action="store", default=50, type=int,
                        required=False, help="max epoch num, default 50")
    parser.add_argument("--min_epoch_num", action="store", default=10, type=int,
                        required=False, help="min epoch num, default 10")
    parser.add_argument('--step_interval', type=int, default=500, required=False)
    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)
    # parser.add_argument('--seed', type=int, default=1234,
    #                     help='random seed')

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
