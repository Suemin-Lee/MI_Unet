# **Codes for "Information flow through U-Nets"**

This repository contains the code for mutual information measurements on U-Net model using the nuclei image dataset.


###### **Installations**

Make sure to have all the requirements installed before running the code. 

* `SaveActivation_nuceli.ipynb` trains the data on 'stage1_train' folder and saves (in a data directory) activations when run on test set('stage1_test') inputs (as well as weight norms, &c.) for each epoch.

* `Unet_nuceli_ComputeMI_coarsening.ipynb` loads the data files, computes MI values using spatial coarsening methods. It plots Information planes and U-shaped information planes from the data created using  `SaveActivation_nuceli.ipynb`.

* `Unet_nuceli_ComputeMI_k_mean_clustering.ipynb` loads the data files, computes MI values using spatial coarsening methods. It plots Information planes and U-shaped information planes from the data created using `SaveActivation_nuceli.ipynb`.


## **Quick method for running the code**

1. First, run `SaveActivation_nuceli.ipynb` with appropriate epochs. (Warning! Saving data may take up a huge amount of storage. Make sure to have enough space or recommend running it on the other machine. Here, it is currently set as 3, make sure to change the epochs to see the full results. )

2. Choose either `Unet_nuceli_ComputeMI_coarsening.ipynb` or `Unet_nuceli_ComputeMI_k_mean_clustering.ipynb` to compute mutual information and for plotting information plane.



## **Reference code and dataset**

The original code from : https://github.com/artemyk/ibsgd 

The dataset from : https://www.kaggle.com/c/data-science-bowl-2018/data


## **Citation**

If you find this code useful, please cite the following paper:

S. Lee and I. V. Bajic, “Information flow through U-Nets,” Proc. IEEE International Symposium on Biomedical Imaging (ISBI), Apr. 2021.

 
```
BibTeX:
@inproceedings{Lee_Bajic_2021,
  author={S. Lee and I. V. Baji'\{c}},
  booktitle={Proc. IEEE International Symposium on Biomedical Imaging (ISBI)},
  title={Information flow through {U-Nets}},
  year={2021},
  month={Apr.},
  address={Nice, France}
}
```


