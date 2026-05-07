# DI-Bench

Official repository for **DI-Bench** (Disaster Intelligence Benchmark).

## Overview

DI-Bench is a multimodal disaster-scene benchmark for evaluating:

- Scene semantic alignment
- Scene understanding and reasoning
- Decision and action planning

The benchmark covers multi-source remote sensing and ground-view inputs, with task categories such as retrieval, alignment, damage assessment, population estimation, and route planning.

## Benchmark Structure

DI-Bench organizes tasks into three levels:

- **Level 1: Scene Semantic Alignment**
- **Level 2: Scene Understanding and Reasoning**
- **Level 3: Decision and Action Planning**

## Dataset

The dataset is organized by `scene_*` folders, each with a `questions.json` file and referenced media assets.

## Quick Start

Detailed setup and running instructions will be provided in the code release.

## TODO

- [ ] Release training/inference code
- [ ] Release full data preparation pipeline
- [ ] Release standardized evaluation scripts and configs
- [ ] Add model zoo and baseline checkpoints
- [ ] Add reproducibility guide

## Citation

If you find DI-Bench useful, please cite our paper (BibTeX will be added after publication).
