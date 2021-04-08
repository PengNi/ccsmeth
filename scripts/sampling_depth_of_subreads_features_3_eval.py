import os
import argparse
from subprocess import Popen,PIPE


curr_dir = os.path.dirname(__file__)
# model_path = "/public/home/hpc174601028/data/hkmodel_pnas/02_DATA/mytest.M03_W03.3/" \
#              "my.train.model_bilstm_ccsstd_M030W030_zscore_depth1_kmerbaled/bilstm.b21_epoch14.ckpt"
model_path = "/public/home/hpc174601028/data/hkmodel_pnas/02_DATA/mytest.M03_W03.3/" \
             "my.train.model_bilstm_ccsstd_M030W030_zscore_depth10_samp_depth5_kmerbaled/bilstm.b21_epoch14.ckpt"


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def _step1_sampling_features(infile, depth):
    script_path = curr_dir + "/sampling_depth_of_subreads_features_2.py"
    fname, fext = os.path.splitext(infile)
    wfile = fname + ".depth_" + str(depth) + fext
    cmd = " ".join(["python", script_path, "--input", infile, "--output", wfile,
                    "--depth", str(depth)])

    stdinfo, returncode = run_cmd(cmd)
    return wfile


def _step2_call_mods(infile):
    script_path = os.path.dirname(curr_dir) + "/deepsmrt/call_modifications.py"
    fname, fext = os.path.splitext(infile)
    wfile = fname + ".call_mods" + fext
    cmd = " ".join(["CUDA_VISIBLE_DEVICES=0", "python", script_path, "--input_path", infile, "--model_path",
                    model_path, "--result_file", wfile,
                    "--nproc_gpu 6 --is_ccs yes --is_stds yes --is_subreads no"])
    stdinfo, returncode = run_cmd(cmd)
    return wfile


def _step3_comb_mods(infile):
    script_path = curr_dir + "/eval_deepsmrt_comb_two_strands_of_ccs.py"
    fname, fext = os.path.splitext(infile)
    wfile = fname + ".fb_comb" + fext
    cmd = " ".join(["python", script_path, "--result_fp", infile])
    stdinfo, returncode = run_cmd(cmd)
    return wfile


def _step4_eval_result(res_m, res_um, depth=1, is_fbcomb=False, outdir="."):
    script_path = curr_dir + "/eval_deepsmrt_in_readlevel.py"
    model_name = model_path.split("/")[-2].split(".")[-1]
    if is_fbcomb:
        wfile = outdir + "/my.evel_sampling." + model_name + ".depth_sampling_" + str(depth) + ".fb_comb.accinfo.txt"
    else:
        wfile = outdir + "/my.evel_sampling." + model_name + ".depth_sampling_" + str(depth) + ".accinfo.txt"
    cmd = " ".join(["python", script_path, "--methylated", res_m, "--unmethylated", res_um,
                    "--result_file", wfile,
                    "--depth_cf 1 --prob_cf 0.0 --prob_cf 0.1 --prob_cf 0.2 --prob_cf 0.4"])
    stdinfo, returncode = run_cmd(cmd)
    return wfile


def main():
    parser = argparse.ArgumentParser("extract features with info of all subreads, and only keep reads with "
                                     "subread_depth >= 30, step2")
    parser.add_argument("--inputm", type=str, required=True,
                        help="")
    parser.add_argument("--inputum", type=str, required=True,
                        help="")
    parser.add_argument("--depth", action="append", required=False,
                        help="append mode, depth")
    parser.add_argument("--res_dir", type=str, required=True, help="")

    args = parser.parse_args()

    res_dir = args.res_dir
    if os.path.exists(res_dir):
        pass
    else:
        os.makedirs(res_dir)

    for depth in args.depth:
        fea_m = _step1_sampling_features(args.inputm, int(depth))
        fea_um = _step1_sampling_features(args.inputum, int(depth))

        callmods_m = _step2_call_mods(fea_m)
        callmods_um = _step2_call_mods(fea_um)

        callmodsfb_m = _step3_comb_mods(callmods_m)
        callmodsfb_um = _step3_comb_mods(callmods_um)

        eval_file = _step4_eval_result(callmods_m, callmods_um, int(depth), False, res_dir)
        evalfb_file = _step4_eval_result(callmodsfb_m, callmodsfb_um, int(depth), True, res_dir)

        os.remove(fea_m)
        os.remove(fea_um)
        os.remove(callmods_m)
        os.remove(callmods_um)
        os.remove(callmodsfb_m)
        os.remove(callmodsfb_um)


if __name__ == '__main__':
    main()
