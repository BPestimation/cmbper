# Continuous Monitoring of Blood Pressure with Evidential Regression

PyTorch implementation of "Continuous Monitoring of Blood Pressure with Evidential Regression"

## Abstract
Photoplethysmogram (PPG) signal-based blood
pressure (BP) estimation is a promising candidate
for modern BP measurements, as PPG signals
can be easily obtained from wearable devices in
a non-invasive manner, allowing quick BP measurement. However, the performance of existing
machine learning-based BP measuring methods
still fall behind some BP measurement guidelines
and most of them provide only point estimates
of systolic blood pressure (SBP) and diastolic
blood pressure (DBP). In this paper, we present a
cutting-edge method which is capable of continuously monitoring BP from the PPG signal and
satisfies healthcare criteria such as the Association for the Advancement of Medical Instrumentation (AAMI) and the British Hypertension Society (BHS) standards. Furthermore, the proposed
method provides the reliability of the predicted
BP by estimating its uncertainty to help diagnose
medical condition based on the model prediction.
Experiments on the MIMIC II database verify
the state-of-the-art performance of the proposed
method under several metrics and its ability to
accurately represent uncertainty in prediction.


## Requirements

- Python 3.7.0
- numpy 1.19.2
- PyTorch 1.7.1
- tensorboardX 2.1
- matplotlib 3.3.4
- sklearn
- h5py 2.10.0


## Pre-processed data download

Click [here](https://drive.google.com/file/d/1x8zBX8LqgOqr3aINe-YEGotaEyU1oPk_/view?usp=sharing) to download the pre-processed data used in the experiments.

## Pretrained model download

Click [here](https://drive.google.com/file/d/1TCO_41hynrPL12sB3DYoVbfoCWC3eFuO/view?usp=sharing) to download the pretrained models.

## Examples

### 1. Model training
In our experiments, we compare 6 different models (Model 1, Model 2, ... , Model6).

Each model can be adequately trained by following these steps:

#### 1-1) 1st stage - Training

To train Model 1 & Model 5, command:

`python train.py  --loss evi --eta 0.0 --zeta 0.1 --bsz 512 --lr 5e-4 --max_itr 500000`

To train Model 2 & Model 6, command:

`python train.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --max_itr 500000`

To train Model 3, command:

`python train.py  --loss L1 --eta 0.0 --bsz 512 --lr 5e-4 --max_itr 500000`

To train Model 4, command:

`python train.py  --loss L1 --eta 1.0 --bsz 512 --lr 5e-4 --max_itr 500000`

Model 1 & Model 2 are selected based on the evidential loss.

Model 3 ~ Model 6 are selected based on the mean absolute error (MAE) loss in the 1st stage and undergo further post-processing.


#### 1-2) 2nd stage - Post-processing

To train Model 3, command:

`python postprocess.py  --loss evi --eta 0.0 --zeta 0.1 --bsz 512 --lr 5e-4 --load_type best_MAE --post_lr 2e-2 --post_itr 1000`

To train Model 4, command:

`python postprocess.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --load_type best_MAE --post_lr 2e-2 --post_itr 1000`

To train Model 5, command:

`python postprocess.py  --loss L1 --eta 0.0 --zeta 1.5 --bsz 512 --lr 5e-4 --load_type best_MAE --post_lr 5e-4 --post_itr 50000`

To train Model 6, command:

`python postprocess.py  --loss L1 --eta 1.0 --zeta 1.5 --bsz 512 --lr 5e-4 --load_type best_MAE --post_lr 5e-4 --post_itr 50000`

Model 3 ~ Model 6 are selected based on the evidential loss in the 2nd stage.

#### 2. Evaluation on Blood Pressure Measurement

Ex) To evaluate Model 6, command:

`python BP_prediction.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --postprocess --load_type best_loss`

`BP_prediction.py` provides evaluation results for BP prediction, the BHS standard, and the AAMI standard.


#### 3. Uncertainty Visualization

Ex) To visualize the uncertainty using Model 6, command:

`python viz_uncertainty.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --postprocess --load_type best_loss`


#### 4. Evaluation on Blood Pressure Measurement with High-Reliability Samples

Ex) To evaluate Model 6 using high-reliability samples, command:

`python BP_prediction_subset.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --postprocess --load_type best_loss --subset_ratio 0.8`

#### 5. Hypertension Classification

Ex) To classify hypertension using Model 6, command:

`python BP_classification.py  --loss evi --eta 1.0 --zeta 0.1 --bsz 512 --lr 5e-4 --postprocess --load_type best_loss --subset_ratio 0.8`


#### (Optional) Data Preparation

Click [here](https://drive.google.com/file/d/1GtsQgPP_gEdeTJHws_O9zL8BviUBWfEO/view) to download to well-distributed BP signals 50mmHg and 200mmHg provided by the authors of [PPG2ABP](https://arxiv.org/abs/2005.01669).

Then, the pre-processed data used in our experiment can be obtained by executing `data_handling.py` as follows:

`python data_handling.py  --data_path DB/data.hdf5 --out_dir datasets`

The refined version of MIMIC II can be found at [here](https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation) and you can proceed pre-processing steps with it from scratch.

## Reference

- PPG2ABP : https://github.com/nibtehaz/PPG2ABP
- DEMUCS : https://github.com/facebookresearch/demucs
- Real Time Speech Enhancement in the Waveform Domain : https://github.com/facebookresearch/denoiser
- Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time: https://ieeexplore.ieee.org/document/7168806