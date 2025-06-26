# Estimation of Battery State of Charge (SoC) in Battery Management System Using Reservoir Spiking Neural Network (RSNN)

This repository contains the implementation and evaluation of a Reservoir Spiking Neural Network (RSNN) for battery State of Charge (SoC) estimation, as part of a final year project (FYP/PSM). The RSNN approach is benchmarked against a conventional Feedforward Neural Network (FFNN) using the NASA lithium-ion battery dataset.

---

## 🔍 Project Overview

This work explores biologically inspired spiking neural networks for time-series modeling of battery SoC. Using various spike encoding strategies, spike trains were generated and processed through an RSNN reservoir implemented in the BindsNET framework. The RSNN results were compared with an FFNN baseline in terms of prediction accuracy and computational efficiency.

> **Note**: As the RSNN was simulated on a standard laptop using BindsNET, performance metrics like training time and memory usage are not reflective of deployment on neuromorphic hardware. The implementation serves as a proof of concept.

---

## 🗂 Repository Structure

### 🔋 `battery_data/`

* Original NASA lithium-ion battery dataset (no modifications made).

### 🧠 `bindsnet/`

* Custom Python library containing `poisson_normalized` spike encoding function.

### 📈 Plotting & Visualization

* `SoC_graph.py`: Generates SoC vs time plots for all cycles (figures not included).
* `spike_visualizer.py`: Visualizes spike trains used in the report (screenshots were taken manually; output is not saved).

### 🧪 RSNN Implementations

* `spikingSoC_v3_poisson_psutil.py`: RSNN with **Poisson encoding** (output in `SoC_Pred_Results_poisson_normalized/`)
* `spikingSoC_v3_fixed_rate.py`: RSNN with **Fixed-rate encoding** (output in `SoC_Pred_Results_fixed_rate/`)
* `spikingSoC_v3_population.py`: RSNN with **Population encoding** (output in `SoC_Pred_Results_population2/`)
* `spikingSoC_v3_ttf.py`: RSNN with **Time-to-First-Spike encoding** *(not functional)*
* `SoC_Spiking_fixed_population.py`: Experimental RSNN using fixed and population encoding *(not in thesis; output in `Population_test/`)*
* `spikingSoC_v3.py`: Deprecated RSNN version (outdated)
* `spikingSoH.py`: RSNN-based State of Health (SoH) estimator from senior project (reference only)

### 🔁 FFNN Baseline

* `ann.ipynb`: Jupyter notebook for FFNN benchmark

  * Outputs: `ann_soc_estimation_batch1/`, `ann_soc_estimation_batch32/`

### 📦 Dependencies

* `requirements.txt`: List of Python libraries used in this project.

---

## 📊 Results Summary

| Model | Encoding                     | RMSE (Test)  | Final MSE (Epoch 20) | Training Time      | Peak RAM Usage             |
| ----- | ---------------------------- | ------------ | -------------------- | ------------------ | -------------------------- |
| FFNN  | -                            | ✓ Comparable | ✓ Comparable         | ✅ Fast             | ✅ Low                      |
| RSNN  | Poisson / Fixed / Population | ✓ Comparable | ✓ Comparable         | ❌ Slow (simulated) | ❌ High (software overhead) |

* **Prediction Accuracy**: RSNN and FFNN both achieved competitive RMSE and MSE.
* **Efficiency**: FFNN outperformed in training time and memory due to batch processing and optimized frameworks.
* **RSNN Limitation**: Simulated in software (BindsNET) on CPU; not representative of real-time, event-driven neuromorphic systems.

---

## 🧠 Potential for Future Work

Deploying the RSNN model on neuromorphic hardware (e.g., Intel Loihi 2) would unlock its low-power, event-driven advantages, making it viable for real-time embedded battery management in electric vehicles, renewable energy systems, and edge-AI devices.

---

## 📚 Citation

**Project Title**: *Estimation of Battery State of Charge (SoC) in Battery Management System Using Reservoir Spiking Neural Network (RSNN)*
**Submitted as**: Final Year Project (PSM)

