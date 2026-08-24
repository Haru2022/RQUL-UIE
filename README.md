# RQUL-UIE

This is the official repo of RQUL-UIE contains the inference script and local pipeline code. The pre-trained model can be loaded by the huggingface model card "HaruCloud9/RQUL-UIE", which is already preset in the test.py.

## Create the environment

```bash
conda env create -f environment.yml
conda activate RQUL-UIE

pip install -r requirements.txt
```

If you prefer to create the environment manually, use `conda create -n RQUL-UIE python=3.10.16` and then install the pinned packages from `requirements.txt`.


## Test

```bash
python test.py
```

## Train

the training code will be released when the manuscript is accepted.
