#!/usr/bin/env python
# coding: utf-8

import os
import time
import math
import warnings
import multiprocessing
from multiprocessing import Process, Queue, Lock
from multiprocessing.sharedctypes import Value, Array
import numpy as np
import pandas as pd
import pysam

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def decode_flag(flag):
    """
    Decodes standard SAM flags into specific framework category signals.
    """
    signal = {4: 0, 0: 1, 16: 2, 2048: 3, 2064: 4}
    return signal[flag] if flag in signal else 0

def c_pos(cigar, refstart):
    """
    Calculates mapping boundaries and clip coordinates from raw CIGAR strings.
    """
    number = ''
    numlist = {str(i) for i in range(10)}
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    
    for c in cigar:
        if c in numlist:
            number += c
        else:
            if not number:
                continue
            number = int(number)
            if readstart is False and c in ['M', 'I', '=', 'X']:
                readstart = readloc
            if readstart != False and c in ['H', 'S']:
                readend = readloc
                refend = refloc
                break

            if c in ['M', 'I', 'S', '=', 'X']:
                readloc += number
            if c in ['M', 'D', 'N', '=', 'X']:
                refloc += number
            number = ''
            
    if readend is False:
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend 

def splitread(chr_name, bamfile):
    """
    Extracts split-alignment features from secondary/supplementary structural tags.
    """
    dada = []
    for read in bamfile.fetch(chr_name):
        if read.has_tag('SA'):
            code = decode_flag(read.flag)
            rawsalist = read.get_tag('SA').split(';')
            for sa in rawsalist[:-1]:
                sainfo = sa.split(',')
                tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
                if tmpcontig != chr_name:
                    continue
                if (strand == '-' and (code % 2) == 0) or (strand == '+' and (code % 2) == 1):
                    refstart_1 = read.reference_start
                    refend_1 = read.reference_end
                    readstart_1 = read.query_alignment_start
                    readend_1 = read.query_alignment_end
                    refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                    
                    a = readend_1 - readstart_2
                    b = refend_1 - refstart_2
                    if abs(b - a) < 30:
                        continue
                    if abs(b) < 2000:
                        if (b - a) > 50 and abs(b - a) < 200000:
                            dada.extend([refstart_2, refend_1])

    data = pd.value_counts(dada)
    return data

def process_read(read, ref_pos, loci_clip_sm, loci_clip_ms, ins_count):
    """
    Processes a single read to parse concrete CIGAR operators for genomic features.
    """
    aligned_length = read.reference_length
    if aligned_length is None:
        aligned_length = 0
        
    if (read.mapping_quality >= 0) and (aligned_length >= 0):
        cigar = np.array(read.cigartuples)
        if cigar is None or len(cigar) == 0:
            return ref_pos, loci_clip_sm, loci_clip_ms, ins_count

        ref_pos.extend(read.get_reference_positions())
        ref_pos_start = read.reference_start + 1 
        
        for i in range(cigar.shape[0]):
            op, length = cigar[i, 0], cigar[i, 1]
            if op in [0, 7, 8, 2]:  # M, =, X, D
                ref_pos_start += length
            elif op == 1:  # I
                if length >= 20:
                    ins_count.append(ref_pos_start)

        if cigar[0, 0] == 4:  # Soft clip at the start
            loci_clip_sm.append(read.reference_start + 1)
        if cigar[-1, 0] == 4:  # Soft clip at the end
            loci_clip_ms.append(read.reference_end)

    return ref_pos, loci_clip_sm, loci_clip_ms, ins_count

def caculate_feature(start, end, ref_pos, loci_clip_sm, loci_clip_ms, ins_count):
    """
    Generates binned frequency distributions of alignment operators across the genomic window.
    """
    ref_pos = np.array(ref_pos) + 1
    if len(ref_pos) == 0:
        ref_pos = np.array([0])
    ref_pos = np.bincount(ref_pos, minlength=int(end + 1))[int(start):int(end)]
    
    if len(loci_clip_sm) == 0:
        loci_clip_sm = np.array([0])
    if len(loci_clip_ms) == 0:
        loci_clip_ms = np.array([0])
        
    loci_clip_sm = np.array(loci_clip_sm)
    loci_clip_ms = np.array(loci_clip_ms)

    loci_clip_sm = np.bincount(loci_clip_sm, minlength=int(end + 1))[int(start):int(end)]
    loci_clip_ms = np.bincount(loci_clip_ms, minlength=int(end + 1))[int(start):int(end)]
    
    if len(ins_count) == 0:
        ins_count = np.array([0])  
    ins_count = np.bincount(ins_count, minlength=int(end + 1))[int(start):int(end)]

    return ref_pos, loci_clip_sm, loci_clip_ms, ins_count

def feature_extraction_long(bamfile, chro, start, end, ref_pos, loci_clip_sm, loci_clip_ms, ins_count):
    """
    Extracts positional alignment metrics within the targeted chromosome coordinates.
    """
    for read in bamfile.fetch(chro, start, end):
        ref_pos, loci_clip_sm, loci_clip_ms, ins_count = process_read(read, ref_pos, loci_clip_sm, loci_clip_ms, ins_count)     
    tmp_ref, tmp_loci_clip_sm, tmp_loci_clip_ms, tmp_ins_count = caculate_feature(start, end, ref_pos, loci_clip_sm, loci_clip_ms, ins_count)
    return tmp_ref, tmp_loci_clip_sm, tmp_loci_clip_ms, tmp_ins_count

def labeldata(vcfpath, contig, start, end, window_size, index):
    """
    Parses structural variant gold-standard labels from VCF configurations.
    """
    goldl = []
    if 'chr' in contig:
        contig = contig[3:]
        
    for rec in pysam.VariantFile(vcfpath).fetch():
        if rec.contig != contig:
            continue            
        if rec.info.get('SVTYPE') == 'INS':
            goldl.append([rec.start, rec.stop, rec.stop - rec.start, 1])
            
    if len(goldl) == 0:
        return np.zeros((len(index), 1), dtype=np.float32)
        
    goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')

    y = [] 
    for rec in index:
        if (((goldl[:, 1:2] > rec) & (goldl[:, :1] < (rec + window_size))).sum() != 0):
            y.append((((goldl[:, 1:2] >= rec) & (goldl[:, :1] <= (rec + window_size))) * goldl[:, 3:]).sum())
        else:
            y.append(0)
    return (np.array(y) > 0).astype('float32')

def fun(a):
    """
    Normalizes multi-dimensional genomic matrix representations.
    """
    oshape = a.shape
    a = a.reshape(-1, 5).astype('float32')
    a -= a.mean(axis=0)
    a /= (np.sqrt(a.var(axis=0)) + 1e-10)
    return a.reshape(oshape)

def compute(converage_long, loci_clip_long_sm, loci_clip_long_ms, split_read, loci_ins_long, start, end):
    """
    Concatenates individual parallel channels into structural identity tensors.
    """
    s_e = np.arange(start, end)
    converage_long = converage_long.reshape(-1, 1)
    loci_clip_long_sm = loci_clip_long_sm.reshape(-1, 1)
    loci_clip_long_ms = loci_clip_long_ms.reshape(-1, 1)
    loci_ins_long = loci_ins_long.reshape(-1, 1)
    loci_ins_split = split_read.reindex(index=s_e).fillna(value=0).values.reshape(-1, 1)
    infor = np.concatenate([converage_long, loci_clip_long_sm, loci_clip_long_ms, loci_ins_long, loci_ins_split], axis=1)
    return infor

def create_data_long(bamfile_long_path, outputpath, contig, vcf_file):
    """
    Executes raw processing pipeline to convert BAM inputs into structured array blocks.
    """
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    time_st = time.time()
    bamfile_long = pysam.AlignmentFile(bamfile_long_path, 'rb', threads=20)
    contig2length = {}
    window = 200
    
    if len(contig) == 0:
        contig = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contig.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contig = np.array(contig).astype(str)
        for count in range(len(bamfile_long.get_index_statistics())):
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
            
    for ww in contig:
        chr_name_long = ww
        chr_length = contig2length[ww]
        ider = math.ceil(chr_length / 10000000)
        start = 0
        end = 10000000
        s = 0
        print(f"chr_name_long = {chr_name_long}, chr_length = {chr_length}, ider = {ider}")
        time_q = time.time()
        split_read = splitread(chr_name_long, bamfile_long)
        
        for n in range(ider):
            time_s = time.time()
            x_data = []
            index = []
            print(f"chr {chr_name_long} start: {start} end: {end} {n+1} / {ider}")
            
            ref_pos = []
            loci_clip_sm = []
            loci_clip_ms = []
            ins_count = []
            
            loci_cover_long, loci_clip_long_sm, loci_clip_long_ms, loci_ins_long = feature_extraction_long(
                bamfile_long, chr_name_long, start, end, ref_pos, loci_clip_sm, loci_clip_ms, ins_count
            )
            
            if len(loci_cover_long) == 0:
                start += 10000000
                end += 10000000 
                continue
                
            xx = compute(loci_cover_long, loci_clip_long_sm, loci_clip_long_ms, split_read, loci_ins_long, start, end)
            xx = xx.reshape(-1, 1000)        

            for k in range(len(xx)):
                if xx[k].any() != 0:
                    x_data.append(xx[k])
                    index.append(s)
                s += window
                
            x_data = np.array(x_data)
            index = np.array(index)
            print(f"x_data.shape = {x_data.shape}, index.shape = {index.shape}")
            
            start += 10000000
            end += 10000000 
            
            if len(x_data) == 0:
                continue
                
            x_data = fun(x_data)
            y_label = labeldata(vcf_file, chr_name_long, start, end, 200, index)
            y_label = y_label.reshape(-1, 1)
            print(f"x_data.shape = {x_data.shape}, y_label.shape = {y_label.shape}")

            x_data = np.concatenate([x_data, y_label], axis=1)
            print(f"final x_data.shape = {x_data.shape}")
            
            if 'chr' in chr_name_long:
                filename_data = os.path.join(outputpath, f"{chr_name_long}_{start-10000000}_{end-10000000}.npy")
                filename_index = os.path.join(outputpath, f"{chr_name_long}_{start-10000000}_{end-10000000}_index.npy")
            else:
                filename_data = os.path.join(outputpath, f"chr{chr_name_long}_{start-10000000}_{end-10000000}.npy")
                filename_index = os.path.join(outputpath, f"chr{chr_name_long}_{start-10000000}_{end-10000000}_index.npy")
                
            np.save(filename_data, x_data)
            np.save(filename_index, index)

            time_e = time.time()
            print(f"Step duration: {time_e - time_s}")
        print(f"Contig runtime: {time.time() - time_q}") 
    print(f"Total runtime: {time.time() - time_st}")

def create_features_multi_threading(bam_file, output_path, contigs_list, max_worker, vcf_file):
    """
    Manages multi-processed infrastructure for distributed sequence transformation.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    bamfile_long = pysam.AlignmentFile(bam_file, 'rb', threads=20)
    contig2length = {}
    
    if len(contigs_list) == 0:
        contigs_list = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contigs_list.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contigs_list = np.array(contigs_list).astype(str)
        
    count = 0
    while count < len(contigs_list):
        if len(multiprocessing.active_children()) < int(max_worker): 
            j = contigs_list[count]
            p = Process(target=create_data_long, args=(bam_file, output_path, [j], vcf_file))
            p.start()
            count += 1
        else:
            time.sleep(2)
