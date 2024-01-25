# -*- coding: utf-8 -*-
import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataloader import FeaData
from .dataloader import FeaData2
from .dataloader import clear_linecache

from .models import ModelAttRNN
from .models import ModelTransEnc
from .models import ModelAttRNN2

from .utils.constants_torch import use_cuda
from .utils.process_utils import display_args
from .utils.process_utils import str2bool

from .utils.logging import mylogger
LOGGER = mylogger(__name__)

device = "cuda" if use_cuda else "cpu"


def train(args):
    total_start = time.time()
    torch.manual_seed(args.tseed)
    if use_cuda:
        torch.cuda.manual_seed(args.tseed)

    LOGGER.info("[main]train starts")
    if use_cuda:
        LOGGER.info("GPU is available!")
    else:
        LOGGER.info("GPU is not available!")

    LOGGER.info("reading data..")
    if args.model_type in {"attbigru2s", "attbilstm2s", "transencoder2s", "attbigru2s2", "attbilstm2s2"}:
        if args.dl_offsets:
            if args.dl_num_workers > 1:
                raise ValueError("--dl_num_workers should not be >1 when --dl_offsets is True!")
            from .utils.process_utils import count_line_num
            from .dataloader import generate_offsets
            train_linenum = count_line_num(args.train_file, False)
            train_offsets = generate_offsets(args.train_file)
            train_dataset = FeaData2(args.train_file, train_offsets, train_linenum)
            valid_linenum = count_line_num(args.valid_file, False)
            valid_offsets = generate_offsets(args.valid_file)
            valid_dataset = FeaData2(args.valid_file, valid_offsets, valid_linenum)
        else:
            train_dataset = FeaData(args.train_file)
            valid_dataset = FeaData(args.valid_file)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.dl_num_workers)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.dl_num_workers)
    else:
        raise ValueError("--model_type not right!")

    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(r"" + args.model_type + "\..*b\d+_epoch\d+\.ckpt*")
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile) is not None:
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    if args.model_type in {"attbigru2s", "attbilstm2s"}:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            is_sn=str2bool(args.is_sn),
                            is_map=str2bool(args.is_map),
                            is_stds=str2bool(args.is_stds),
                            is_npass=str2bool(args.is_npass),
                            model_type=args.model_type,
                            device=device)
    elif args.model_type in {"attbigru2s2", "attbilstm2s2"}:
        model = ModelAttRNN2(args.seq_len, args.layer_rnn, args.class_num,
                             args.dropout_rate, args.hid_rnn,
                             is_sn=str2bool(args.is_sn),
                             is_map=str2bool(args.is_map),
                             is_stds=str2bool(args.is_stds),
                             is_npass=str2bool(args.is_npass),
                             model_type=args.model_type,
                             device=device)
    elif args.model_type in {"transencoder2s"}:
        model = ModelTransEnc(args.seq_len, args.layer_trans, args.class_num,
                              args.dropout_rate, args.d_model, args.nhead, args.dim_ff, 
                              is_npass=str2bool(args.is_npass), is_sn=str2bool(args.is_sn),
                              is_map=str2bool(args.is_map), is_stds=str2bool(args.is_stds), 
                              model_type=args.model_type, device=device)
    else:
        raise ValueError("--model_type not right!")

    if args.init_model is not None:
        LOGGER.info("loading pre-trained model: {}".format(args.init_model))
        para_dict = torch.load(args.init_model, map_location=torch.device('cpu'))
        # para_dict = torch.load(model_path, map_location=torch.device(device))
        model_dict = model.state_dict()
        model_dict.update(para_dict)
        model.load_state_dict(model_dict)
    
    if str2bool(args.use_compile):
        try:
            model = torch.compile(model)
        except:
            raise ImportError('torch.compile does not exist in PyTorch<2.0.')

    if use_cuda:
        # TODO: WARNING not solved when using DataParalle():
        # TODO: /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1266:
        # TODO: UserWarning: RNN module weights are not part of single contiguous chunk of memory.
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.to(device)
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
            from .utils.ranger2020 import Ranger
        except ImportError:
            raise ImportError("please check if ranger2020.py is in utils/ dir!")
        optimizer = Ranger(model.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-5)
    elif args.optim_type == "LookaheadAdam":
        try:
            from .utils.lookahead import LookaheadAdam
        except ImportError:
            raise ImportError("please check if lookahead.py is in utils/ dir!")
        optimizer = LookaheadAdam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("--optim_type is not right!")

    if args.lr_scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay,
                                      patience=args.lr_patience, verbose=True)
    else:
        raise ValueError("--lr_scheduler is not right!")

    # Train the model
    total_step = len(train_loader)
    LOGGER.info("total_step: {}".format(total_step))
    curr_best_accuracy = 0
    curr_best_accuracy_loc = 0
    curr_best_accuracy_epoches = []
    model.train()
    for epoch in range(args.max_epoch_num):
        curr_best_accuracy_epoch = 0
        accuracies_per_epoch = []
        no_best_model = True
        tlosses = []
        start = time.time()
        for i, sfeatures in enumerate(train_loader):
            if args.model_type in {"attbigru2s", "attbilstm2s", "transencoder2s", "attbigru2s2", "attbilstm2s2"}:
                _, fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fsn, fmap, \
                    rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rsn, rmap, \
                    labels = sfeatures
                if use_cuda:
                    fkmer = fkmer.to(device)
                    fpass = fpass.to(device)
                    fipdm = fipdm.to(device)
                    fipdsd = fipdsd.to(device)
                    fpwm = fpwm.to(device)
                    fpwsd = fpwsd.to(device)
                    fsn = fsn.to(device)
                    fmap = fmap.to(device)

                    rkmer = rkmer.to(device)
                    rpass = rpass.to(device)
                    ripdm = ripdm.to(device)
                    ripdsd = ripdsd.to(device)
                    rpwm = rpwm.to(device)
                    rpwsd = rpwsd.to(device)
                    rsn = rsn.to(device)
                    rmap = rmap.to(device)

                    labels = labels.to(device)
                # Forward pass
                outputs, logits = model(fkmer, fpass, fipdm, fipdsd, fpwm, fpwsd, fsn, fmap,
                                        rkmer, rpass, ripdm, ripdsd, rpwm, rpwsd, rsn, rmap)
                loss = criterion(outputs, labels)
                tlosses.append(loss.detach().item())
            else:
                raise ValueError("--model_type is not right!")

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.step_interval == 0 or (i + 1) == total_step:
                model.eval()
                with torch.no_grad():
                    vlosses, vlabels_total, vpredicted_total = [], [], []
                    for vi, vsfeatures in enumerate(valid_loader):
                        if args.model_type in {"attbigru2s", "attbilstm2s", "transencoder2s", "attbigru2s2", "attbilstm2s2"}:
                            _, vfkmer, vfpass, vfipdm, vfipdsd, vfpwm, vfpwsd, vfsn, vfmap, \
                                vrkmer, vrpass, vripdm, vripdsd, vrpwm, vrpwsd, vrsn, vrmap, \
                                vlabels = vsfeatures
                            if use_cuda:
                                vfkmer = vfkmer.to(device)
                                vfpass = vfpass.to(device)
                                vfipdm = vfipdm.to(device)
                                vfipdsd = vfipdsd.to(device)
                                vfpwm = vfpwm.to(device)
                                vfpwsd = vfpwsd.to(device)
                                vfsn = vfsn.to(device)
                                vfmap = vfmap.to(device)

                                vrkmer = vrkmer.to(device)
                                vrpass = vrpass.to(device)
                                vripdm = vripdm.to(device)
                                vripdsd = vripdsd.to(device)
                                vrpwm = vrpwm.to(device)
                                vrpwsd = vrpwsd.to(device)
                                vrsn = vrsn.to(device)
                                vrmap = vrmap.to(device)

                                vlabels = vlabels.to(device)
                            # Forward pass
                            voutputs, vlogits = model(vfkmer, vfpass, vfipdm, vfipdsd, vfpwm,
                                                      vfpwsd, vfsn, vfmap,
                                                      vrkmer, vrpass, vripdm, vripdsd, vrpwm,
                                                      vrpwsd, vrsn, vrmap)
                            vloss = criterion(voutputs, vlabels)
                        else:
                            raise ValueError("--model_type is not right!")

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
                    accuracies_per_epoch.append(v_accuracy)
                    if v_accuracy > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = v_accuracy
                        if curr_best_accuracy_epoch > curr_best_accuracy - 0.0002:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type +
                                       '.b{}_epoch{}.ckpt'.format(args.seq_len, epoch + 1))
                            if curr_best_accuracy_epoch > curr_best_accuracy:
                                curr_best_accuracy = curr_best_accuracy_epoch
                                curr_best_accuracy_loc = epoch + 1
                                no_best_model = False

                        if len(curr_best_accuracy_epoches) > 0 and curr_best_accuracy_epoch > \
                                curr_best_accuracy_epoches[-1]:
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type +
                                       '.betterthanlast.b{}_epoch{}.ckpt'.format(args.seq_len,
                                                                                 epoch + 1))

                    time_cost = time.time() - start
                    try:
                        last_lr = scheduler.get_last_lr()
                        LOGGER.info('Epoch [{}/{}], Step [{}/{}]; LR: {:.4e}; TrainLoss: {:.4f}; '
                                    'ValidLoss: {:.4f}, '
                                    'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                    'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                    .format(epoch + 1, args.max_epoch_num, i + 1, total_step, last_lr,
                                            np.mean(tlosses), v_meanloss, v_accuracy, v_precision, v_recall,
                                            curr_best_accuracy_epoch, curr_best_accuracy, time_cost))
                    except Exception:
                        LOGGER.info('Epoch [{}/{}], Step [{}/{}]; TrainLoss: {:.4f}; '
                                    'ValidLoss: {:.4f}, '
                                    'Acc: {:.4f}, Prec: {:.4f}, Reca: {:.4f}, '
                                    'CurrE_best_acc: {:.4f}, Best_acc: {:.4f}; Time: {:.2f}s'
                                    .format(epoch + 1, args.max_epoch_num, i + 1, total_step,
                                            np.mean(tlosses), v_meanloss, v_accuracy, v_precision, v_recall,
                                            curr_best_accuracy_epoch, curr_best_accuracy, time_cost))

                    tlosses = []
                    start = time.time()
                    # sys.stdout.flush()
                model.train()

        if args.lr_scheduler == "ReduceLROnPlateau":
            if args.lr_mode_strategy == "mean":
                reduce_metric = np.mean(accuracies_per_epoch)
            elif args.lr_mode_strategy == "last":
                reduce_metric = accuracies_per_epoch[-1]
            elif args.lr_mode_strategy == "max":
                reduce_metric = np.max(accuracies_per_epoch)
            else:
                raise ValueError("--lr_mode_strategy is not right!")
            scheduler.step(reduce_metric)
        else:
            scheduler.step()

        curr_best_accuracy_epoches.append(curr_best_accuracy_epoch)
        if no_best_model and epoch >= args.min_epoch_num - 1:
            LOGGER.info("early stop!")
            break
    endtime = time.time()
    clear_linecache()
    if args.dl_offsets:
        train_dataset.close()
        valid_dataset.close()
    LOGGER.info("[main]train costs {:.1f} seconds, "
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

    st_train = parser.add_argument_group("TRAIN MODEL_HYPER")
    # model param
    st_train.add_argument('--model_type', type=str, default="attbigru2s",
                          choices=["attbilstm2s", "attbigru2s", "transencoder2s", 
                                   "attbilstm2s2", "attbigru2s2",],
                          required=False,
                          help="type of model to use, 'attbilstm2s', 'attbigru2s', "
                               "'transencoder2s', 'attbilstm2s2', 'attbigru2s2', "
                               "default: attbigru2s")
    st_train.add_argument('--seq_len', type=int, default=21, required=False,
                          help="len of kmer. default 21")
    st_train.add_argument('--is_npass', type=str, default="yes", required=False,
                          help="if using num_pass features, yes or no, default yes")
    st_train.add_argument('--is_sn', type=str, default="no", required=False,
                          help="if using signal-to-noise-ratio features, yes or no, default no")
    st_train.add_argument('--is_map', type=str, default="no", required=False,
                          help="if using mapping features, yes or no, default no")
    st_train.add_argument('--is_stds', type=str, default="no", required=False,
                          help="if using std features, yes or no, default no")
    st_train.add_argument('--class_num', type=int, default=2, required=False)
    st_train.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    
    st_trainb = parser.add_argument_group("TRAIN MODEL_HYPER RNN")
    # BiRNN model param
    st_trainb.add_argument('--layer_rnn', type=int, default=3,
                          required=False, help="BiRNN layer num, default 3")
    st_trainb.add_argument('--hid_rnn', type=int, default=256, required=False,
                          help="BiRNN hidden_size, default 256")
    
    st_traint = parser.add_argument_group("TRAIN MODEL_HYPER TRANSFORMER")
    # Transformer model param
    st_traint.add_argument('--layer_trans', type=int, default=6, required=False,
                          help="TransformerEncoder nlayers, default 6")
    st_traint.add_argument('--nhead', type=int, default=4, required=False,
                          help="TransformerEncoder nhead, default 4")
    st_traint.add_argument('--d_model', type=int, default=256, required=False, 
                          help="TransformerEncoder input feature numbers, default 256")
    st_traint.add_argument('--dim_ff', type=int, default=512, required=False,
                          help="TransformerEncoder dim_feedforward, default 512")

    st_training = parser.add_argument_group("TRAINING")
    # model training
    st_training.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD",
                                                                                "Ranger", "LookaheadAdam"],
                             required=False, help="type of optimizer to use, 'Adam', 'SGD', 'RMSprop', "
                                                  "'Ranger' or 'LookaheadAdam', default Adam")
    st_training.add_argument('--batch_size', type=int, default=512, required=False)
    st_training.add_argument('--lr_scheduler', type=str, default='StepLR', required=False,
                             choices=["StepLR", "ReduceLROnPlateau"],
                             help="StepLR or ReduceLROnPlateau, default StepLR")
    st_training.add_argument('--lr', type=float, default=0.001, required=False,
                             help="default 0.001")
    st_training.add_argument('--lr_decay', type=float, default=0.1, required=False,
                             help="default 0.1")
    st_training.add_argument('--lr_decay_step', type=int, default=1, required=False,
                             help="effective in StepLR. default 1")
    st_training.add_argument('--lr_patience', type=int, default=0, required=False,
                             help="effective in ReduceLROnPlateau. default 0")
    st_training.add_argument('--lr_mode_strategy', type=str, default="last", required=False,
                             choices=["last", "mean", "max"],
                             help="effective in ReduceLROnPlateau. last, mean, or max, default last")
    st_training.add_argument("--max_epoch_num", action="store", default=50, type=int,
                             required=False, help="max epoch num, default 50")
    st_training.add_argument("--min_epoch_num", action="store", default=10, type=int,
                             required=False, help="min epoch num, default 10")
    st_training.add_argument('--pos_weight', type=float, default=1.0, required=False)
    st_training.add_argument('--step_interval', type=int, default=500, required=False)
    st_training.add_argument('--dl_num_workers', type=int, default=0, required=False,
                             help="default 0")
    st_training.add_argument('--dl_offsets', action="store_true", default=False, required=False,
                             help="use file offsets loader")

    st_training.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    st_training.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')
    st_training.add_argument('--use_compile', type=str, default="no", required=False,
                             help="if using torch.compile, yes or no, default no ('yes' only works in pytorch>=2.0)")

    args = parser.parse_args()

    display_args(args)
    train(args)


if __name__ == '__main__':
    main()
