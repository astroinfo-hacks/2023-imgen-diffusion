# Generative modeling of astro images with diffusion models

## Rationale

The goal of this hack is to use Diffusion Models to build a simulations-driven prior that can be used to denoise and deconvolve galaxy images from JWST.

We would follow these steps:
  1. Gather simulated galaxy images from the Illustris TNG hydrodynamical simulations to act as a prior on what perfect observations would look like:  
![image](https://github.com/astroinfo-hacks/2023-imgen-diffusion/assets/861591/ad38c62f-2b60-4ab7-8233-a98faadadfbe)
![image](https://github.com/astroinfo-hacks/2023-imgen-diffusion/assets/861591/c721d6d8-d209-4dd6-ba8a-b9de245d8332)

  2. Train a Score-Based Denoising Diffusion Model on these images. This is a type of generative models which can learn to generate images similar to the input training data. These diffusion models work by learning a mapping from pure noise to images in the training set:
![image](https://github.com/astroinfo-hacks/2023-imgen-diffusion/assets/861591/a1469a0a-5fad-4255-911b-24bc80393a9a)

  4. Use the knownledge about galaxy morphology collected by the model to solve the inverse problem of reconstructing a high resolution, PSF-deconvolved, and noise-free image of observed JWST images similar to these:
![image](https://github.com/astroinfo-hacks/2023-imgen-diffusion/assets/861591/1440306f-5174-4d22-9bd2-af5ee6ccb6fb)

In more details, we will be using a forward model of the observed images, which takes into account the noise and and PSF properties of the image, and solve a Bayesian inverse problem with the Diffusion Model as our prior, to sample posterior images, compatible with the observations and still likely under the prior. 


You can watch the [hack presentation][telecon] by Marc during the telecon.

[telecon]: https://u-paris.zoom.us/rec/share/ibQAB_HcRwoRFxrmne3RtWUnGp3xH_bqsS9oOG0vMHZEPJidfSASYsXzR_MzNCM.0GfrQ39bReZsAScg

## Dataset

 We will be using galaxy images from Illustris TNG, prepared with a radiative transfer code. The observations we will try to denoise and deconvolve are galaxy images from JWST.

## References

  - A nice introduction to diffusion models: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  - An example of paper using diffusion models as priors for inverse models:https://arxiv.org/abs/2201.05561
