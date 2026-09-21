# SSL-InsNet
a Sequence Structured Labeling Framework for Large-Scale Genomic Insertion Detection Leveraging Time-Distributed Dual Transformer

<img width="1119" height="421" alt="SSL-InsNet Architecture" src="https://anonymous.4open.science/r/SSL-InsNet-5D41/imgs/main_plot_new.png" />

<div align="center">
  
**Overall workflow and network architecture of SSL-InsNet for large-scale genomic insertion variant calling.**

</div>

# 📖 Overview

**SSL-InsNet** is a novel sequence structured labeling framework engineered for large-scale genomic insertion detection in third-generation long-read sequencing data. The framework reformulates insertion calling as a structured tagging task, integrating a Multi-level Spatial Perception (MSP) module and a Time-distributed Transformer (T-Trans) within a dual-transformer architecture. By incorporating an advanced confidence-gated fusion layer, the network effectively suppresses alignment artifacts to deliver highly precise, chromosome-wide structural variant profiles.

# ✨ Key Features

⏳ **Sequence Structured Labeling**: Reformulates large-scale genomic insertion detection into an optimized sequence structured tagging task across continuous long-read tracks.

🧬 **Multi-level Spatial Perception (MSP)**: Integrates a **Spatial Transformer (S-Trans)** with deep convolutions via a hierarchical attention mechanism to simultaneously resolve micro-scale breakpoint motifs and regional genomic contexts.

🤝 **Confidence-gated Feature Integration (CFI)**: Implements an elite adaptive **Feature Fusion** module that leverages learnable descriptors to fuse heterogeneous spatial features, effectively suppressing sequencer background noise and alignment artifacts.

⏳ **Time-distributed Transformer (T-Trans)**: Tracks macro-scale long-distance sequence dependencies along the temporal axis across consecutive segments, ensuring global structural consistency for ultra-long variants under standard GPU footprints.

🛡️ **Neighborhood-Weighted Focal Loss (NWFL)**: Utilizes a custom structural optimization loss function to supervise continuous position tags and drastically refine boundary alignment precision.

# 🚀 Quick Start

## Prerequisites

- Python 3.9.1
- CUDA 12.4+
- PyTorch 2.5
- 11+ GB VRAM recommended

## Installation

### Create a virtual environment
conda create -n ssl_insnet python=3.9.1 -y
conda activate ssl_insnet

### Install core dependencies
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install pandas==2.2.3 numpy==1.26.4

### Install specialized bioinformatics packages
pip install pysam==0.23.0

## Essential Bioinformatics Dependencies

| Package | Purpose | Official Repository |
|---------|---------|---------------------|
| ![pysam](https://img.shields.io/badge/pysam-0.23.0-14a2b8?logo=python&logoColor=white) | BAM/CRAM file processing and header extraction | [GitHub](https://github.com/pysam-developers/pysam) |
| ![pandas](https://img.shields.io/badge/pandas-2.2.3-14a2b8?logo=pandas&logoColor=white) | Window feature metrics management | [GitHub](https://github.com/pandas-dev/pandas) |
| ![numpy](https://img.shields.io/badge/numpy-1.26.4-14a2b8?logo=numpy&logoColor=white) | Vectorized genomic array serialization | [GitHub](https://github.com/numpy/numpy) |

# 📁 Workflow Pipeline

## 1. Produce Data for Call SV (`generate_feature`)

Extract alignment signatures and base matrices into continuous serialized chunks across target chromosomes.
```
python  SSL_InsNet.py generate_feature bam_file output_path contigs_list(default:[](all chromosomes)) max_worker vcf_file
```
bam_file: the path of the alignment file about the reference and the long read set;

output_path: a folder which is used to store generated features data;

contigs_list: the list of contig to preform detection. (default: [], all contig are used);

max_worker: the number of threads to use;

vcf_file: the gold standard file for standard data.
```
eg: python  SSL_InsNet.py generate_feature ./HG002_PB_5x_RG_HP10XtrioRTG.bam ./features_dir [12,13] 5 ./HG002_SVs_Tier1_v0.6.vcf.gz
```

## 2. Call Insertion (`call_insertion`)

Stream window blocks through the Time-Distributed network to run sequence decoding and merge insertion candidates.
```
python  SSL_InsNet.py call_insertion gpu_name save_length timesteps ins_predict_weight data_path bam_file out_vcf_file contigs support
```

gpu_name: num of the GPU to use (e.g., '0' or '1,2');

save_length: the feature file spans across nucleotide base sequence lengths;

timesteps: time step of time-distributed network;

ins_predict_weight: path of insert predict weight file (.pth);

data_path: a folder for storing evaluation feature files;

bam_file: path of the alignment file about the reference and the long read set;

out_vcf_file: the path of output vcf file;

contigs: the list of contig to preform detection. (default: [], all contig are used);

support: min support reads.
```
eg: python SSL_InsNet.py call_insertion '1,2' 10000000 100 ./ins_predict_weight.pth ./features_dir /home/laicx/00.dataset/HG002_PB_10x_RG_HP10XtrioRTG.bam ./out_vcf_file.vcf [12,13] 5
```

# 📊 Tested Data

## HG002 CLR data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam


## HG002 ONT data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam


## HG002 CCS data
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb/alignment/HG002.Sequel.15kb.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
