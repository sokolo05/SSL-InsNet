#!/usr/bin/env python
# coding: utf-8
import sys
import pysam
from predict_model import predict_funtion
from generate_feature import create_features_multi_threading

def get_all_contigs(bam_path):
    """Dynamically fetch all contig names from the BAM file header if none are provided."""
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return [stat.contig for stat in bam.get_index_statistics()]

if len(sys.argv) < 2:
    mode = "help"
else:
    mode = sys.argv[1]

debug = 0
print(f'len of sys.argv = {len(sys.argv)}')

if mode == 'generate_feature':
    if len(sys.argv) not in [5, 6, 7]:
        debug = 1
    else:
        print(f'Produce data')
        if len(sys.argv) == 7:
            bam_file, output_path, contigs_list, max_worker, vcf_file = sys.argv[2], sys.argv[3], [str(contig) for contig in eval(sys.argv[4])], int(sys.argv[5]), sys.argv[6]
        elif len(sys.argv) == 6:
            bam_file, output_path, contigs_list, max_worker, vcf_file = sys.argv[2], sys.argv[3], [], int(sys.argv[4]), sys.argv[5]
        elif len(sys.argv) == 5:
            bam_file, output_path, contigs_list, max_worker, vcf_file = sys.argv[2], sys.argv[3], [], 5, sys.argv[4]

        print(f'bam_file = {bam_file}')
        print(f'output_path = {output_path}')
        print(f'max_worker set to {max_worker}')

        if not contigs_list:
            print(f'All chromosomes within bamfile will be used')
            contigs_list = get_all_contigs(bam_file)
        
        print(f'Following chromosomes will be used: {contigs_list}')
        create_features_multi_threading(bam_file=bam_file, output_path=output_path, contigs_list=contigs_list, max_worker=max_worker, vcf_file=vcf_file)
        print(f'\nGenerate feature completed\n')
        
elif mode == 'call_insertion':
    if len(sys.argv) not in [10, 11]:
        debug = 1
    else:
        print(f'testing')
        if len(sys.argv) == 11:
            gpu_name, save_length, timesteps, ins_predict_weight, data_path, bam_file, out_vcf_file, contigs, support = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], [str(contig) for contig in eval(sys.argv[9])], sys.argv[10]
        else:
            gpu_name, save_length, timesteps, ins_predict_weight, data_path, bam_file, out_vcf_file, contigs, support = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], [], sys.argv[9]
        
        print(f'bam_file path = {bam_file}')
        print(f'ins_predict_weight path = {ins_predict_weight}')
        print(f'data_path = {data_path}')
        print(f'out_vcf_file = {out_vcf_file}')
        
        if not contigs:
            print(f'All chromosomes within bamfile will be used')
            contigs = get_all_contigs(bam_file)
            
        print(f'Following chromosomes will be used: {contigs}')
        predict_funtion(gpu_name, save_length, timesteps, ins_predict_weight, data_path, bam_file, out_vcf_file, contigs, support)
        print(f'\nCompleted, Result saved in current folder\n')

else:
    debug = 1

if debug == 1:
    script_name = sys.argv[0] if sys.argv[0] else "SSL_InsNet.py"
    print(f'\nUsage:')
    print(f'Produce data for call insertion:')
    print(f'python {script_name} generate_feature bam_file output_path contigs_list(default:[](all chromosomes)) max_worker vcf_file')
    print(f'Call insertion:')
    print(f'python {script_name} call_insertion gpu_name save_length timesteps ins_predict_weight data_path bam_file out_vcf_file contigs(default:[](all chromosomes)) support')

# Examples for reference:
# python SSL_InsNet.py generate_feature bam_file.bam ./chr12-13 [12,13] 5 ./HG002_SVs_Tier1_v0.6.vcf.gz
# python SSL_InsNet.py call_insertion '1,2' 10000000 100 ./ins_predict_weight.pth ./00.dataset/HG002_PB_70x_RG_HP10XtrioRTG/ ./HG002_PB_10x_RG_HP10XtrioRTG.bam ./out_vcf_file.vcf [12,13,14,15,16,17,18,19,20,21,22] 5
