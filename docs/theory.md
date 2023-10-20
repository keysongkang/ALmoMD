# Theoretical Background

Active-learning machine-operated molecular dynamics (ALmoMD) is a Python code package designed for the effective training of machine learned interatomic potential (MLIP) through active learning based on uncertainty evaluation. It also facilitates the implementation of molecular dynamics (MD) using trained MLIPs with uncertainty evaluation.

## Uncertainty evaluation

In ALmoMD, __uncertainty__ refers to the prediction uncertainty of a group of trained models, which can be obtained through three distinct training methods [1]. First, each model can be trained with the same number of different training data (Subsampling). Second, each model can be trained with different random initializations of the machine learning model but the same training data (Deep Ensemble). Lastly, each model can be trained using different machine learning techniques. In all cases, these models provide a range of different predictions, and their standard deviation indicates the degree of uncertainty, which is used in active learning. ALmoMD combines Subsampling and Deep Ensemble methods to determine this uncertainty.

<br>
<figure style="text-align:center;">
  <img src="fig_uncert.png" alt="Uncertainty evaluation" width="800"/>
  <figcaption>Figure 1. (Left) Various models can be trained using the Subsampling and Deep Ensemble methods. (Right) The disparity in their predictions can be quantified by calculating their standard deviation, which serves as a measure of uncertainty.</figcaption>
</figure>
<br>

We note that there are significant concerns regarding the use of uncertainty in global phase predictions, as raised by both the Zipoli group [1] and Scheffler group [2]. However, in ALmoMD, we utilize uncertainty to qualitatively identify cases when the models go beyond their trained domain, enabling us to determine where additional model training is necessary. For example, if we exclusively train the CuI models with pure crystals, we observe uncertainty spikes when defects are created or when the model departs from its trained domain.

<br>
<figure style="text-align:center;">
  <img src="fig_al.png" alt="Active learning scheme" width="800"/>
  <figcaption>Figure 2. (Left) New data (light blue and purple points) are obtained from molecular dynamics using MLIP. The error bars indicate their uncertainty. Since the purple points exhibit high uncertainty, they are selected for the next round of training data, and their energies are subsequently corrected through DFT calculations. (Right) Retraining with additional samples from the active learning process will yield a more reliable MLIP, which can be iteratively improved.</figcaption>
</figure>
<br>



# References
[1] L. Kahle and F. Zipoli, _Phys. Rev. E_ __105__, 015311 (2022).
[2] S. Lu, L.M. Ghringhelli, C. Carbogno, J. Wang, and M. Scheffler, arXiv:2309.00195 (2023).

# Contents
- [Back to Home](READMD.md)
- [Installation Guide](installation.md)
- [User Manuals](documentation.md)
- [Tutorials](tutorial.md)