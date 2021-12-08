import torch
from models import ModelRNN
from models import ModelAttRNN
from models import ModelAttRNN2s
from models import ModelResNet18
from models import ModelTransEncoder
import argparse
import os


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    p_call = argparse.ArgumentParser("unzip model_ckpt saved by torch 1.6+, for lower version torch use")
    p_call.add_argument("--model_file", type=str, required=True, help="model path")

    p_call.add_argument('--model_type', type=str, default="attbigru",
                        choices=["attbilstm", "attbigru", "bilstm", "bigru",
                                 "transencoder", "attbigru2s",
                                 "resnet18"],
                        required=False,
                        help="type of model to use, 'attbilstm', 'attbigru', "
                             "'bilstm', 'bigru', 'transencoder', 'resnet18', 'attbigru2s',"
                             "default: attbigru")
    p_call.add_argument('--seq_len', type=int, default=21, required=False,
                        help="len of kmer. default 21")
    p_call.add_argument('--is_stds', type=str, default="yes", required=False,
                        help="if using std features at ccs level, yes or no. default yes.")
    p_call.add_argument('--class_num', type=int, default=2, required=False)
    p_call.add_argument('--dropout_rate', type=float, default=0.5, required=False)

    # BiRNN/transformerencoder model param
    p_call.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    p_call.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")

    # BiRNN model param
    p_call.add_argument('--layer_rnn', type=int, default=3,
                        required=False, help="BiRNN layer num, default 3")
    p_call.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiRNN hidden_size for combined feature")

    # transformerencoder model param
    p_call.add_argument('--layer_tfe', type=int, default=6,
                        required=False, help="transformer encoder layer num, default 6")
    p_call.add_argument('--d_model_tfe', type=int, default=256,
                        required=False, help="the number of expected features in the "
                                             "transformer encoder/decoder inputs")
    p_call.add_argument('--nhead_tfe', type=int, default=4,
                        required=False, help="the number of heads in the multiheadattention models")
    p_call.add_argument('--nhid_tfe', type=int, default=512,
                        required=False, help="the dimension of the feedforward network model")

    args = p_call.parse_args()

    if args.model_type in {"bilstm", "bigru", }:
        model = ModelRNN(args.seq_len, args.layer_rnn, args.class_num,
                         args.dropout_rate, args.hid_rnn,
                         args.n_vocab, args.n_embed,
                         is_stds=str2bool(args.is_stds),
                         model_type=args.model_type)
    elif args.model_type in {"attbilstm", "attbigru", }:
        model = ModelAttRNN(args.seq_len, args.layer_rnn, args.class_num,
                            args.dropout_rate, args.hid_rnn,
                            args.n_vocab, args.n_embed,
                            is_stds=str2bool(args.is_stds),
                            model_type=args.model_type)
    elif args.model_type in {"attbigru2s", }:
        model = ModelAttRNN2s(args.seq_len, args.layer_rnn, args.class_num,
                              args.dropout_rate, args.hid_rnn,
                              args.n_vocab, args.n_embed,
                              is_stds=str2bool(args.is_stds),
                              model_type=args.model_type)
    elif args.model_type in {"transencoder", }:
        model = ModelTransEncoder(args.seq_len, args.layer_tfe, args.class_num,
                                  args.dropout_rate, args.d_model_tfe, args.nhead_tfe, args.nhid_tfe,
                                  args.n_vocab, args.n_embed,
                                  is_stds=str2bool(args.is_stds),
                                  model_type=args.model_type)
    elif args.model_type == "resnet18":
        model = ModelResNet18(args.class_num, args.dropout_rate, str2bool(args.is_stds))
    else:
        raise ValueError("model_type not right!")

    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    fname, fext = os.path.splitext(args.model_file)
    mode_unzip_file = fname + ".unzip" + fext
    torch.save(model.state_dict(), mode_unzip_file, _use_new_zipfile_serialization=False)
