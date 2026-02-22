Make sure jax, torch compiled with cuda 12

```
conda activate jax
```

Run to cluster all deepmath traces into different difficulty levels and upload to HF :

```
python process_deepmath_traces.py --hf_key 
```

Train latent ode : 

```
CUDA_VISIBLE_DEVICES=0 python train_latent_ode.py