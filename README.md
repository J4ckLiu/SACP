# Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification

## How to Run

Calculating non-conformity scores with SACP

```
python main.py --model sstn --data_name ip --alpha 0.05 --base_score APS
```

with the following arguments:

- model: the name of the model, including ```cnn1d, cnn3d, hybrid, sstn.```


- data_name: the name of dataset, including ```ip, pu, sa.```
- alpha: the user-specified error rate.
- base_score: the standard non-coformity score, incuding ```APS, RAPS, SAPS.```


<pre>
@article{liu2024spatial,
  title={Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification},
  author={Liu, Kangdao and Sun, Tianhao and Zeng, Hao and Zhang, Yongshan and Pun, Chi-Man and Vong, Chi-Man},
  journal={arXiv preprint arXiv:2409.01236},
  year={2024}
}
</pre>




