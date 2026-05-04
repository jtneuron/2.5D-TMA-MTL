# 2.5D-TMA-MTL

A task-driven multi-task learning framework for 2.5D Tissue Microarrays in Gastric Cancer

## Overview

This repository implements the framework from our v2 paper, which captures multi-layer tumor heterogeneity and integrates multiple tasks including:
- Biomarker prediction (PD-L1, Claudin18.2)
- Cancer subtype classification (Lauren subtype)
- Survival analysis (Overall Survival)

The framework includes:
- **Task-Aware Intra-Slide Fusion Module** with SSP and TCDR
- **Shared-Adaptive Representation Module** for cross-task knowledge exchange
- Compatible with multiple MIL backbones and pathology foundation models

## Installation

```bash
git clone https://github.com/username/2.5D-TMA-MTL.git
cd 2.5D-TMA-MTL
pip install -r requirements.txt
