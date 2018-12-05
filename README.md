# Overview 
This code implement the DARTS paper in Tensorflow
>[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)  
>Hanxiao Liu, Karen Simonyan, Yiming Yang.  
>arXiv:1806.09055.  

## Architecture Search 
To carry out architecture search using 2nd-order approximation, run
```
python cnn/train_search.py
```
NOTE:
* Only implement the Cifar10 experiments
* Bacht size is set to only 16 (64 in the pytorch Ver) for the OOM issue using one single 1080 Ti GPU
* Only implement the 2nd-order approximation, but you can simply modify the code if you want 1st-order one
* Didn't implement the PATH DROP operation on test code for the unclear motivation  

Here shows the details training pocess
<p align="center">
  <img src="images/train_loss.png" alt="darts" width="70%">
  <img src="images/valid_loss.png" alt="darts" width="70%">
</p>  
<p align="center">
Fig. The Train loss and Valid loss of the searching process
</p>  

The train loss is decreased steadly during the searching pocess, which is more stable than the RL based method. Note that the valid loss refer to the 2nd-order loss for architecture params.  
<p align="center">
  <img src="images/normal.png" alt="darts" width="70%">
  </p> 
  <p align="center">
  Fig. The final normal cell  
  </p>  
  <p align="center">
  <img src="images/reduction.png" alt="darts" width="70%">  
  <p align="center">  
  Fig. The final reduction cell  
  </p>  
</p>  

## Architecture evaluation  
Still working hard to implement the test code ...  
## Acknowledgement
* This implementation is based on the original Torch implementation [quark0/darts](https://github.com/quark0/darts)  
