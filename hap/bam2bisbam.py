"""
refer to bam2bis from phase module of NanoMethPhase
"""
import os
import sys
import argparse
import re
import pysam

import gzip
import bz2
import tabix
from tqdm import tqdm
from subprocess import Popen, PIPE
import multiprocessing as mp
from itertools import repeat
from collections import defaultdict
import warnings


def run_cmd(args_list):
    proc = Popen(args_list, shell=True, stdout=PIPE, stderr=PIPE)
    stdinfo = proc.communicate()
    # print(stdinfo)
    return stdinfo, proc.returncode


def openfile(file):
    '''
    Opens a file
    '''
    if file.endswith('.gz'):
        opened_file = gzip.open(file,'rt')
    elif file.endswith('bz') or file.endswith('bz2'):
        opened_file = bz2.open(file,'rt')
    else:
        opened_file = open(file,'rt')
    return opened_file


def methcall2bed(readlist,
                 callthresh=0,
                 readID_index=3,
                 prob0_index=5,
                 prob1_index=6):
    read_list = list()
    for read in readlist:
        methylated_sites = []
        unmethylated_sites = []
        prob_methylated = []
        prob_unmethylated = []
        for line in read:
            line = line.split('\t')
            strand = line[2]
            cpg_pos = int(line[1])
            if strand == "-":
                cpg_pos = cpg_pos - 1  # to make it like nanopolish
            read_id = line[readID_index]
            chrom = line[0]
            deltaprob = float(line[prob1_index]) - float(line[prob0_index])
            # Skipping ambiguous call in methyl call file
            if abs(deltaprob) < callthresh:
                continue
            if deltaprob > 0:
                methylated_sites.append(cpg_pos)
                prob_methylated.append(str(deltaprob))
            else:
                unmethylated_sites.append(cpg_pos)
                prob_unmethylated.append(str(deltaprob))
        all_positions = sorted(methylated_sites + unmethylated_sites)
        if all_positions:
            if not methylated_sites:
                methylated_sites.append('NA')
                prob_methylated.append('NA')
            if not unmethylated_sites:
                unmethylated_sites.append('NA')
                prob_unmethylated.append('NA')
            append_info = (chrom,
                           str(all_positions[0]),
                           str(all_positions[-1] + 1),
                           strand, read_id,
                           ','.join(prob_methylated),
                           ','.join(prob_unmethylated),
                           ','.join(map(str, methylated_sites)),
                           ','.join(map(str, unmethylated_sites)))
            read_list.append(append_info)

    return read_list


def methy_sitetsv2readbed(methycallfile, readbedfile, threads=10, chunk_size=100, callthresh=0,
                          isont=False):
    MethylCallfile = os.path.abspath(methycallfile)
    chunk = chunk_size

    if isont:
        readID_index = 4
        start_index = 1
        strand_index = 2
        prob0_index = 6
        prob1_index = 7
    else:
        readID_index = 3
        start_index = 1
        strand_index = 2
        prob0_index = 5
        prob1_index = 6

    meth = openfile(MethylCallfile)
    # next(meth)  # To skip the header
    prev_info= next(meth).rstrip().split('\t')
    prev_readID= prev_info[readID_index]
    prev_start= int(prev_info[start_index])
    prev_strand= prev_info[strand_index]
    all_lines = 1
    for line in meth:
        all_lines += 1
    meth.close()
    meth = openfile(MethylCallfile)
    # next(meth)  # To skip the header
    feedlist = []
    chunklist = []
    readlist= []
    tqdm_add= 0

    wf = open(readbedfile, "w")
    with tqdm(total=all_lines,
              desc="MethylCallProcessor: ", bar_format=
              "{l_bar}{bar} [ Estimated time left: {remaining} ]") as pbar:
        for line in meth:
            tqdm_add += 1
            line = line.rstrip()
            line_info= line.split('\t')
            start= int(line_info[start_index])
            if (line_info[readID_index] == prev_readID and
                line_info[strand_index] == prev_strand and
                abs(start -  prev_start) < 300000):
                prev_readID = line_info[readID_index]
                prev_strand = line_info[strand_index]
                prev_start= start
                readlist.append(line)
            else:
                chunklist.append(readlist)
                readlist = []
                readlist.append(line)
                prev_readID = line_info[readID_index]
                prev_strand = line_info[strand_index]
                prev_start= start
            if len(chunklist) == chunk:
                feedlist.append(chunklist)
                chunklist = []
            if len(feedlist) == threads:
                p = mp.Pool(threads)
                results = p.starmap(methcall2bed,
                                    list(zip(feedlist,
                                             repeat(float(callthresh)),
                                             repeat(int(readID_index)),
                                             repeat(int(prob0_index)),
                                             repeat(int(prob1_index)))))
                p.close()
                p.join()
                for result in results:
                    if result is not None:
                        for processed_line in result:
                            wf.write('\t'.join(processed_line)+'\n')
                feedlist = []
                pbar.update(tqdm_add)
                tqdm_add= 0
        else:
            chunklist.append(readlist)
            feedlist.append(chunklist)
            p = mp.Pool(len(feedlist))
            results = p.starmap(methcall2bed,
                                list(zip(feedlist,
                                         repeat(float(callthresh)),
                                         repeat(int(readID_index)),
                                         repeat(int(prob0_index)),
                                         repeat(int(prob1_index)))))
            p.close()
            p.join()
            for result in results:
                if result is not None:
                    for processed_line in result:
                        wf.write('\t'.join(processed_line)+'\n')
            feedlist = []
            pbar.update(tqdm_add)
    meth.close()
    wf.close()
    return readbedfile


def zip_readbed(readbedfile, tmpdir="/home/nipeng", threads=10):
    cmd = "sort -T{tmp} --parallel={nproc} -k1,1 -k2,2n -k3,3n {bed}" \
          "| bgzip -@ {nproc} > {bed}.gz && tabix -p bed {bed}.gz".format(tmp=tmpdir,
                                                                          nproc=threads,
                                                                          bed=readbedfile)
    stdinfo, returncode = run_cmd(cmd)
    return returncode, readbedfile+".gz"


def read2bis(read_sam_list):
    """
    This function converts a read based on information in processed
    MethylCallFile to a bisulfite format read for nice visualization by
    IGV.
    """
    motif = 'CG'
    sam_list = read_sam_list[2:]
    ref_seq = sam_list[-5]
    strand = read_sam_list[1]
    HP = read_sam_list[0]
    all_methylated = sam_list[-2]
    all_unmethylated = sam_list[-1]
    all_tags= sam_list[-3]
    all_sites = [(j.start()) for j in re.finditer(motif, ref_seq)]
    ref_seq = list(ref_seq)
    if strand == '-' and motif == 'CG':
        offset = 1
        ambigbase = 'N'
        unmodified = 'A'
    else:
        offset = 0
        ambigbase = 'N'
        unmodified = 'T'
    for site in all_sites:
        if site not in all_methylated:
            if site+offset < len(ref_seq):
                ref_seq[site+offset] = ambigbase

    for site in all_unmethylated:
        if site+offset < len(ref_seq):
            ref_seq[site+offset] = unmodified
    ref_seq = ''.join(ref_seq)
    return [HP]+sam_list[0:-5]+[ref_seq]+[sam_list[-4]]+[all_tags]


def alignmentwriter(result,
                    output):
    '''
    Writes the results of converting reads to bisulfite format
    to a bam file
    '''
    (HP, read_id, flag, true_ref_name ,
     start, mp_quality, cigar, RNEXT ,
     PNEXT, TLEN, ref_seq, QUAL, all_tags) = result
    out_samRead = pysam.AlignedSegment(output.header)
    out_samRead.query_name = read_id
    out_samRead.cigarstring = str(len(ref_seq))+'M'
    out_samRead.query_sequence = ref_seq
    out_samRead.flag = flag
    out_samRead.reference_name = true_ref_name
    out_samRead.reference_start = start
    out_samRead.mapping_quality = mp_quality
    if HP != 'NON':
        all_tags= [(HP[0:2], int(HP[-1]),"i")]+all_tags
    if len(all_tags) >= 1:
        out_samRead.set_tags(all_tags)
    output.write(out_samRead)


def openalignment(alignment_file,
                  window):
    '''
    Opens a bam/sam file and creates bam iterator
    '''
    bam = pysam.AlignmentFile(alignment_file, 'rb')
    if window is not None:
        window_chrom = window.split(':')[0]
        if len(window.split(':')) == 2:
            window_margin= window.split(':')[1].split('-')
            if len(window_margin) == 2:
                window_start = int(window_margin[0])
                window_end = int(window_margin[1])
                bamiter = bam.fetch(window_chrom, window_start, window_end)
                count= bam.count(window_chrom, window_start, window_end)
            else:
                window_start = int(window_margin[0])
                bamiter = bam.fetch(window_chrom, window_start)
                count= bam.count(window_chrom, window_start)
        else:
            try:
                bamiter = bam.fetch(window_chrom)
                count= bam.count(window_chrom)
            except:
                count= 0
                bamiter= ""
    else:
        bamiter = bam.fetch(until_eof=True)
        count = 0
    return bamiter, bam, count


def bam_info_extractor(read,
                       reference,
                       fasta):
    if read.is_reverse:
        strand = "-"
    else:
        strand = "+"
    read_id = read.query_name
    start = read.reference_start
    end = read.reference_end
    true_ref_name = read.reference_name
    rnext= read.next_reference_name
    pnext= read.next_reference_start
    tlen= read.template_length
    cigar = read.cigartuples
    base_qualities = read.query_qualities
    flag = read.flag
    if read.query_sequence:
        read_seq = read.query_sequence
    ref_seq = ""
    ref_len = ""
    if reference is not None:
        try:
            ref_seq = fasta.fetch(reference=true_ref_name,
                                  start=start,
                                  end=end)
        except:
            warnings.warn("Reference genome sequence was not found "
                          "for this read: {} at this cordinates {}:{}-{}. "
                          "Skipping the read".format(read_id,
                                                     true_ref_name,
                                                     start,
                                                     end))
    if ((read_seq and cigar and base_qualities) and
    (cigar != "*" or cigar is not None) and
                                      base_qualities is not None):
        read_seq = read_seq.upper()
        read_len = read.query_alignment_length
        ref_seq= ref_seq.upper()
        ref_len= len(ref_seq)
        all_tags= read.get_tags(with_value_type=True)
        return (true_ref_name, strand, flag, read_id, read_seq ,
                read_len, cigar, rnext, pnext, tlen, base_qualities ,
                start, end, ref_seq, ref_len, all_tags)
    else:
        warnings.warn("{} does not have a read sequence,CIGAR"
                      ", or base quality information. "
                      "Skipping the read".format(read_id))


def bam2bis(haped_bam, refpath, methycall1, methycall2, outprefix, threads, isont, region,
            chunk=100):
    bam_file = os.path.abspath(haped_bam)
    reference = os.path.abspath(refpath)

    fasta = pysam.FastaFile(reference)

    bam = pysam.AlignmentFile(bam_file, 'rb')
    chrom_list = "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10," \
                 "chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20," \
                 "chr21,chr22,chrX,chrY".split(",")
    if region is None:
        outHP12BisSam = pysam.AlignmentFile(outprefix + ".hp1bis.bam",
                                            "wb", template=bam)
        outHP22BisSam = pysam.AlignmentFile(outprefix + ".hp2bis.bam",
                                            "wb", template=bam)
    else:
        words = region.split(":")
        cchrom = words[0]
        crange = words[1].split("-")
        cstart = int(crange[0])
        cend = int(crange[1])
        outHP12BisSam = pysam.AlignmentFile(outprefix + ".hp1bis_{}_{}.bam".format(cchrom, cstart),
                                            "wb", template=bam)
        outHP22BisSam = pysam.AlignmentFile(outprefix + ".hp2bis_{}_{}.bam".format(cchrom, cstart),
                                            "wb", template=bam)
        chrom_list = [region, ]

    if not os.path.isfile(os.path.abspath(methycall1) + ".tbi"):
        raise Exception("Could not find index file for methylation call file1.")
    tb1 = tabix.open(os.path.abspath(methycall1))
    if not os.path.isfile(os.path.abspath(methycall2) + ".tbi"):
        raise Exception("Could not find index file for methylation call file2.")
    tb2 = tabix.open(os.path.abspath(methycall2))

    all_read = highq_read = h1_bam2bis = h2_bam2bis = 0
    for chrom in chrom_list:
        read_sam_list = list()
        bamiter, bam, count = openalignment(bam_file, chrom)
        description = "Processing reads from {}: ".format(chrom)
        with tqdm(total=count,
                  desc=description) as pbar:
            for read in bamiter:
                pbar.update(1)
                all_read += 1
                mp_quality = read.mapping_quality

                if (read.is_unmapped or
                        read.is_secondary or read.is_supplementary or
                        read.is_qcfail or read.is_duplicate):
                    continue

                highq_read += 1
                (true_ref_name, strand, flag, read_id,
                 read_seq, read_len, cigar, rnext, pnext, tlen,
                 base_qualities, start, end, ref_seq, ref_len,
                 all_tags) = bam_info_extractor(read,
                                                reference,
                                                fasta)

                records1 = None
                records2 = None
                try:
                    records1 = tb1.query(true_ref_name, start, end)
                except:
                    warnings.warn("{}:{}-{} does not exist in the "
                                  "MethylCallFile1."
                                  "Skipping it".format(read_id,
                                                       start,
                                                       end))
                try:
                    records2 = tb2.query(true_ref_name, start, end)
                except:
                    warnings.warn("{}:{}-{} does not exist in the "
                                  "MethylCallFile2."
                                  "Skipping it".format(read_id,
                                                       start,
                                                       end))
                cnt_record1 = cnt_record2 = 0
                if records1 is not None:
                    methylated_sites = list()
                    unmethylated_sites = list()
                    llr_methylated = list()
                    llr_unmethylated = list()
                    for record in records1:
                        record_readid = record[4] + "/ccs" if not isont else record[4]
                        if read_id == record_readid and ((not isont) or strand == record[3]):
                            cnt_record1 += 1
                            if record[7] != 'NA':
                                methylated_sites += map(int,
                                                        record[7].split(','))
                                llr_methylated += map(float,
                                                      record[5].split(','))
                            if record[8] != 'NA':
                                unmethylated_sites += map(int,
                                                          record[8].split(','))
                                llr_unmethylated += map(float,
                                                        record[6].split(','))
                    if cnt_record1 > 0:
                        methylcall_dict = dict()
                        for i, j in zip(methylated_sites +
                                        unmethylated_sites,
                                        llr_methylated +
                                        llr_unmethylated):
                            if (i >= start and i <= end):
                                if i not in methylcall_dict:
                                    methylcall_dict[i] = [record[0],
                                                          i,
                                                          i + 1,
                                                          strand,
                                                          read_id, j]
                                elif abs(j) > abs(methylcall_dict[i][-1]):
                                    methylcall_dict[i] = [record[0],
                                                          i,
                                                          i + 1,
                                                          strand,
                                                          read_id, j]
                        read_sam_list.append(['HP1', strand, read_id,
                                              flag, true_ref_name, start,
                                              mp_quality, ref_len, rnext,
                                              pnext, tlen, ref_seq, '*',
                                              all_tags,
                                              [i - start for i in methylcall_dict.keys()
                                               if methylcall_dict[i][-1] > 0],
                                              [i - start for i in methylcall_dict.keys()
                                               if methylcall_dict[i][-1] <= 0]])
                if records2 is not None:
                    methylated_sites = list()
                    unmethylated_sites = list()
                    llr_methylated = list()
                    llr_unmethylated = list()
                    for record in records2:
                        record_readid = record[4] + "/ccs" if not isont else record[4]
                        if read_id == record_readid and ((not isont) or strand == record[3]):
                            cnt_record2 += 1
                            if record[7] != 'NA':
                                methylated_sites += map(int,
                                                        record[7].split(','))
                                llr_methylated += map(float,
                                                      record[5].split(','))
                            if record[8] != 'NA':
                                unmethylated_sites += map(int,
                                                          record[8].split(','))
                                llr_unmethylated += map(float,
                                                        record[6].split(','))
                    if cnt_record2 > 0:
                        methylcall_dict = dict()
                        for i, j in zip(methylated_sites +
                                        unmethylated_sites,
                                        llr_methylated +
                                        llr_unmethylated):
                            if (i >= start and i <= end):
                                if i not in methylcall_dict:
                                    methylcall_dict[i] = [record[0],
                                                          i,
                                                          i + 1,
                                                          strand,
                                                          read_id, j]
                                elif abs(j) > abs(methylcall_dict[i][-1]):
                                    methylcall_dict[i] = [record[0],
                                                          i,
                                                          i + 1,
                                                          strand,
                                                          read_id, j]
                        read_sam_list.append(['HP2', strand, read_id,
                                              flag, true_ref_name, start,
                                              mp_quality, ref_len, rnext,
                                              pnext, tlen, ref_seq, '*',
                                              all_tags,
                                              [i - start for i in methylcall_dict.keys()
                                               if methylcall_dict[i][-1] > 0],
                                              [i - start for i in methylcall_dict.keys()
                                               if methylcall_dict[i][-1] <= 0]])
                if cnt_record1 > 0 and cnt_record2 > 0:
                    raise ValueError("???")
                elif cnt_record1 > 0:
                    h1_bam2bis += 1
                elif cnt_record2 > 0:
                    h2_bam2bis += 1
                if len(read_sam_list) == (threads * chunk):
                    p = mp.Pool(threads)
                    results = p.map(read2bis, read_sam_list)
                    p.close()
                    p.join()
                    for result in results:
                        if result is not None:
                            if result[0] == "HP1":
                                alignmentwriter(result, outHP12BisSam)
                            else:
                                alignmentwriter(result, outHP22BisSam)
                    read_sam_list = list()
            else:
                if read_sam_list:
                    p = mp.Pool(threads)
                    results = p.map(read2bis, read_sam_list)
                    p.close()
                    p.join()
                    for result in results:
                        if result is not None:
                            if result[0] == "HP1":
                                alignmentwriter(result, outHP12BisSam)
                            else:
                                alignmentwriter(result, outHP22BisSam)
    outHP12BisSam.close()
    outHP22BisSam.close()
    sys.stderr.write("Job Finished.\n"
                     "all reads: {}, "
                     "highquality_read: {}, "
                     "hp1 reads: {}, "
                     "hp2 reads: {}".format(all_read, highq_read, h1_bam2bis, h2_bam2bis))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haped_bam", type=str, required=True, help="")
    parser.add_argument("--haped_tsv1", type=str, required=True, help="call_mods.hp1.tsv")
    parser.add_argument("--haped_tsv2", type=str, required=True, help="call_mods.hp2.tsv")
    parser.add_argument("--ref", type=str, required=True, help="ref")
    parser.add_argument("--outprefix", type=str, required=True, help="outprefix")
    parser.add_argument("--threads", type=int, default=20, required=False,
                        help="threads")
    parser.add_argument("--tmpdir", type=str, default="/home/nipeng", required=False,
                        help="tmpdir")
    parser.add_argument("--ont", action="store_true", default=False,
                        required=False, help="")
    parser.add_argument("--region", type=str, default=None, required=False,
                        help="chr1:0-111")

    args = parser.parse_args()

    readbedgz1 = args.haped_tsv1 + ".read.bed.gz"
    if not os.path.exists(args.haped_tsv1 + ".read.bed.gz.tbi"):
        readbedfile1 = methy_sitetsv2readbed(args.haped_tsv1, args.haped_tsv1 + ".read.bed", args.threads,
                                             100, 0, args.ont)
        returncode, readbedgz1 = zip_readbed(readbedfile1, args.tmpdir, args.threads)
        if not returncode:
            os.remove(readbedfile1)
        else:
            raise FileNotFoundError("gz file1 not generated!!")
    readbedgz2 = args.haped_tsv2 + ".read.bed.gz"
    if not os.path.exists(args.haped_tsv2 + ".read.bed.gz.tbi"):
        readbedfile2 = methy_sitetsv2readbed(args.haped_tsv2, args.haped_tsv2 + ".read.bed", args.threads,
                                             100, 0, args.ont)
        returncode, readbedgz2 = zip_readbed(readbedfile2, args.tmpdir, args.threads)
        if not returncode:
            os.remove(readbedfile2)
        else:
            raise FileNotFoundError("gz file2 not generated!!")
    bam2bis(args.haped_bam, args.ref, readbedgz1, readbedgz2, args.outprefix, args.threads,
            args.ont, args.region)


if __name__ == '__main__':
    main()
