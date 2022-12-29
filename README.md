# DecT

Source code for Decoder Tuning

## Installation

Our code is based on [OpenPrompt](https://github.com/thunlp/OpenPrompt), please install OpenPrompt

```bash
pip install openprompt
```

This will also check other dependencies like Transformers and PyTorch.

## Download Datasets

Download the 10 datasets with the following scripts

```bash
cd datasets
bash download_datasets.sh
cd ..
```

## Run DecT

Then you can run DecT by running `run_dect.py`, for example

```bash
python src/run_dect.py \
	--model roberta \
	--model_name_or_path roberta-large \
	--shot 1 \
	--dataset sst2 \
	--proto_dim 128 \
	--model_logits_weight 1 \
```

In `run_dect.py` we provide instructions for each argument. To reproduce the results in paper, please run the following combinations

```bash
python src/run_dect.py \
	--shot [1, 4, 16] \
	--dataset [sst2, imdb, yelp, agnews, dbpedia, yahoo, rte, snli, mnli-m, mnli-mm, fewnerd] \
	--seed [0, 1, 2, 3, 4] \
```
