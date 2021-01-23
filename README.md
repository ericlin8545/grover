# Grover
##### UPDATE, Sept 17 2019. We got into NeurIPS (camera ready coming soon!) and we've made Grover-Mega publicly available without you needing to fill out the form. You can download it using [download_model.py](download_model.py).

(aka, code for [Defending Against Neural Fake News](https://arxiv.org/abs/1905.12616))

Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. 

Visit our project page at [rowanzellers.com/grover](https://rowanzellers.com/grover), [the AI2 online demo](https://grover.allenai.org), or read the full paper at [arxiv.org/abs/1905.12616](https://arxiv.org/abs/1905.12616). 

![teaser](https://i.imgur.com/VAGFpBe.png "teaser")

## What's in this repo?

We are releasing the following:
* Code for the Grover generator (in [lm/](lm/)). This involves training the model as a language model across fields.
* Code for the Grover discriminator in [discrimination/](discrimination/). Without much changing, you can run Grover as a discriminator to detect Neural Fake News.
* Code for generating from a Grover model, in [sample/](sample/).
* Code for making your own RealNews dataset in [realnews/](realnews/).
* Model checkpoints freely available online for *all* of the Grover models. For using the RealNews dataset for research, please [submit this form](https://docs.google.com/forms/d/1LMAUeUtHNPXO9koyAIlDpvyKsLSYlrBj3rYhC30a7Ak) and message me on [contact me on Twitter](https://twitter.com/rown) or [through email](https://scr.im/rowan). You will need to use a valid account that has google cloud enabled, otherwise, I won't be able to give you access 😢

Scroll down 👇 for some easy-to-use instructions for setting up Grover to generate news articles.

## Setting up your environment

*NOTE*: If you just care about making your own RealNews dataset, you will need to set up your environment separately just for that, using an AWS machine (see [realnews/](realnews/).)

There are a few ways you can run Grover:
* **Generation mode (inference)**. This requires a GPU because I wasn't able to get top-p sampling, or caching of transformer hidden states, to work on a TPU.
* **LM Validation mode (perplexity)**. This could be run on a GPU or a TPU, but I've only tested this with TPU inference.
* **LM Training mode**. This requires a large TPU pod.
* **Discrimination mode (training)**. This requires a TPU pod.
* **Discrimination mode (inference)**. This could be run on a GPU or a TPU, but I've only tested this with TPU inference.

**NOTE**: You might be able to get things to work using different hardware. However, it might be a lot of work engineering wise and I don't recommend it if possible. Please don't contact me with requests like this, as there's not much help I can give you.

I used Python3.6 for everything. Usually I set it up using the following commands:
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/conda && \
     rm ~/miniconda.sh && \
     ~/conda/bin/conda install -y python=3.6
```
Then `pip install -r requirements-gpu.txt` if you're installing on a GPU, or `pip install requirements-tpu.txt` for TPU.

Misc notes/tips:
* If you have a lot of projects on your machine, you might want to use an anaconda environment to handle them all. Use `conda create -n grover python=3.6` to create an environment named `grover`. To enter the environment use `source activate grover`. To leave use `source deactivate`.
* I'm using tensorflow `1.13.1` which requires Cuda `10.0`. You'll need to install that from the nvidia website. I usually install it into `/usr/local/cuda-10.0/`, so you will need to run `export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64` so tensorflow knows where to find it. 
* I always have my pythonpath as the root directory. While in the `grover` directory, run `export PYTHONPATH=$(pwd)` to set it.

## Quickstart: setting up Grover for generation!

1. Set up your environment. Here's the easy way, assuming anaconda is installed: `conda create -y -n grover python=3.6 && source activate grover && pip install -r requirements-gpu.txt`
2. Download the model using `python download_model.py base`
3. Now generate: `PYTHONPATH=$(pwd) python sample/contextual_generate.py -model_config_fn lm/configs/base.json -model_ckpt models/base/model.ckpt -metadata_fn sample/april2019_set_mini.jsonl -out_fn april2019_set_mini_out.jsonl`

Congrats! You can view the generations, conditioned on the domain/headline/date/authors, in `april2019_set_mini_out.jsonl`.

## FAQ: What's the deal with the release of Grover?

Our core position is that [it is important to release possibly-dangerous models to researchers](https://thegradient.pub/why-we-released-grover/). At the same time, we believe Grover-Mega isn't particularly useful to anyone who isn't doing research in this area, particularly as [we have an online web demo available](https://grover.allenai.org/) and the model is computationally expensive. We previously were a bit stricter and limited initial use of Grover-Mega to researchers. Now that several months have passed since we put the paper on arxiv, and since several other large-scale language models have been publicly released, we figured that there is little harm in fully releasing Grover-Mega.

### Bibtex

```
@inproceedings{zellers2019grover,
    title={Defending Against Neural Fake News},
    author={Zellers, Rowan and Holtzman, Ari and Rashkin, Hannah and Bisk, Yonatan and Farhadi, Ali and Roesner, Franziska and Choi, Yejin},
    booktitle={Advances in Neural Information Processing Systems 32},
    year={2019}
}
```

## Tutorial for Capstone Project

### Basic Setup

Please go through the following steps to do the basic setup of the capstone project.

1. Use the following anaconda command to create a new conda environment called ```capstone ``` and enter the created environment.

   ```conda create -y -n capstone python=3.6 && source activate capstone```

2. Git clone this repository with the following command and will get a `grover` directory.

   ```git clone https://github.com/ericlin8545/grover.git```

3. Get into the cloned `/grover` and execute the following commands.
   
   ```pip install -r requirements-gpu.txt && conda install -c anaconda cudnn```

4. Execute the following commands to download the discrimination model of Grover from Google Cloud Storage.

   ```bash
   gsutil cp gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.data-00000-of-00001 output/
   gsutil cp gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.index output/
   gsutil cp gs://grover-models/discrimination/generator=medium~discriminator=grover~discsize=medium~dataset=p=0.96/model.ckpt-1562.meta output/
   ```

   **Note:** If you haven't installed Gsutil in your machine or workstation, you need to install it before running the above commands. For installing Gsutil, please follow the [tutorial](https://cloud.google.com/storage/docs/gsutil_install#linux).

### Setup the used GPU for the attack program

Before running the attack program, please use ```nvidia-smi``` to check which GPU is not occupied now, and setup the **CUDA_VISIBLE_DEVICES** to use the GPU.

 For example, if you found that GPU 1 is free, use the `export CUDA_VISIBLE_DEVICES=1` to use GPU 1.

### Run the Attack Program

Use the following command to run the attack program

```bash
PYTHONPATH=$(pwd) python discrimination/attackGrover.py --input_data ./input_data/Machine_Label_Examples.txt --output_dir output/ --predict_test true --config_file ./lm/configs/large.json --recipe PWWS > output.txt
```

For the parameters, here is the explanation:

- **inpput_data:** the file containing the fake news that is going to be attacked, and `Machine_Label_Examples.txt` is a file  that has 500 fake news that is labeld as machine by Grover.
- **output_dir:** the directory that saves the numericla attacking results, not the text attacking result.
- **predict_test:** predict the data with test label or not, since all of our test samples are labeled as test, so we need to turn it on to do the Grover discrimination.
- **config_file:** the config file for input data, for here we used large format.
- **recipe:** the attacking algorithm, you can replace `PWWS` with `BAE` or `BERTAttack`.

And the attacking result will be standard outputed, so I just printed it to a file called `output.txt` in here. Feel free to change the printed file name.