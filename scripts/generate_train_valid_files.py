import argparse
import os
import numpy as np
import gc
from subprocess import Popen, PIPE


# =================================================================
def count_line_num(sl_filepath, fheader=False):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    print('done count the lines of file {}'.format(sl_filepath))
    return count


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        print('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    print('done writing features info to {}'.format(wfilepath))


def concat_two_files(file1, file2, concated_fp, shuffle_lines_num=2000000,
                     lines_num=1000000000000, isheader=False):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, False)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, False)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()
    print('done concating files to: {}'.format(concated_fp))
# =================================================================


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posfile", type=str, required=True, help="file containing positive samples")
    parser.add_argument("--negfile", type=str, required=True, help="file containing negative samples")
    parser.add_argument("--wprefix", type=str, required=True, help="file path prefix")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="ratio of samples used for train, else "
                                                                       "for validation")

    args = parser.parse_args()
    posfile = args.posfile
    negfile = args.negfile
    tratio = args.train_ratio
    wprefix = args.wprefix.rstrip("/")

    combined_file = wprefix + ".comb.txt"
    concat_two_files(posfile, negfile, combined_file, 5000000, isheader=False)

    line_cnt = count_line_num(combined_file, False)
    line_cnt_t = int(round(line_cnt * tratio))
    line_cnt_v = line_cnt - line_cnt_t
    trainfile = wprefix + ".train_" + str(round(tratio, 2)) + "_" + str(round(line_cnt_t/1000)) + "k.txt"
    validfile = wprefix + ".valid_" + str(round(1 - tratio, 2)) + "_" + str(round(line_cnt_v/1000)) + "k.txt"
    _, rcode_t = run_cmd(" ".join(["head", "-n", str(line_cnt_t), combined_file,  ">", trainfile]))
    # sed -ne':a;$p;N;line_cnt_v+1,$D;ba' A.txt > B.txt
    _, rcode_v = run_cmd(" ".join(["sed -ne':a;$p;N;"+str(line_cnt_v+1)+",$D;ba'",
                                   combined_file,  ">", validfile]))
    if rcode_t or rcode_v:
        print("generate files failed..")
    else:
        print("generate files succeed..")
    os.remove(combined_file)


if __name__ == '__main__':
    main()
