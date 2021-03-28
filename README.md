# Code for On the Information Bottleneck Theory of Deep Learning
Last update:Oct.22.2020

Make sure to have all the requirements installed before running the code. 

* `SaveActivation_nuceli.ipynb` trains the data on 'stage1_train' folder and saves (in a data directory) activations when run on test set('stage1_test') inputs (as well as weight norms, &c.) for each epoch.

* `Unet_nuceli_ComputeMI_coarsening.ipynb` loads the data files, computes MI values using spatial coarsening methods. It plots Information planes and U-shaped information planes from the data created using 'SaveActivation_nuceli.ipynb'.

* `Unet_nuceli_ComputeMI_k_mean_clustering.ipynb` loads the data files, computes MI values using spatial coarsening methods. It plots Information planes and U-shaped information planes from the data created using 'SaveActivation_nuceli.ipynb'.

*'production_slurm_1.sh' used for running 'SaveActivation_nuceli.py' code in cedar.

*'production_slurm_2.sh' used for running 'Unet_nuceli_ComputeMI_coarsening.py' or `Unet_nuceli_ComputeMI_k_mean_clustering.py` code in cedar.

***********Quick method for running the code***********

1. Run 'SaveActivation_nuceli.ipynb' with appropriate epochs. (Warning! Saving data may take up a huge amount of storage. Make sure to have enough space or recommend running it in the other machine.)

2. Choose either `Unet_nuceli_ComputeMI_coarsening.ipynb` or `Unet_nuceli_ComputeMI_k_mean_clustering.ipynb` to compute mutual information and for plotting information plane.

