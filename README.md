# HNCSE
A PyTorch implementation for our paper "HNCSE: Advancing Sentence Embeddings through Hybrid Contrastive Learning with Hard Negatives".

## Abstract

Unsupervised sentence representation learning remains a pivotal challenge in modern Natural Language Processing (NLP). Recently, contrastive learning techniques have risen to prominence in addressing this, exhibiting notable success in capturing textual semantics. Many such models prioritize optimization using negative samples. In domains like computer vision, hard negatives—samples near class boundaries and thus more difficult to distinguish—have been shown to enhance representation learning. Yet, adapting hard negatives for contrastive sentence learning is intricate due to the complex syntactic and semantic intricacies of text. To tackle this, we present BlendCSE, a novel contrastive model that extends the leading SimCSE approach. BlendCSE's hallmark is its strategic use of hard negatives to boost positive instance learning. This doesn't just involve identifying hard negatives but also creating additional ones and intensifying the embedding process for a deeper semantic grasp. Empirical tests on semantic text similarity and transfer task datasets confirm BlendCSE's superiority.

## Requirement

* Python==3.8.13
* Torch==1.12.1
* Numpy==1.19.5
* Transformer==4.2.1

### Train

```shell
bash run_unsup_example.sh
```

### Evaluation

```shell
MODEL_PATH=result/BlendCSE_base_bert
python3 simcse_to_huggingface.py --path=$MODEL_PATH
python evaluation.py --model_name_or_path $MODEL_PATH --pooler cls_before_pooler --task_set full --mode test
```

### Model

See our model in huggingface:
