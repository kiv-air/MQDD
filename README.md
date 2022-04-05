# MQDD - Multimodal Question Duplicity Detection

This repository publishes trained models and other supporting materials for the paper 
[MQDD – Pre-training of Multimodal Question Duplicity Detection for Software Engineering Domain](https://arxiv.org/abs/2203.14093). For more information, see the paper.

The Stack Overflow Datasets (SOD) and Stack Overflow Duplicity Dataset (SODD) presented in the paper can be obtained from our [Stack Overflow Dataset repository](https://github.com/kiv-air/StackOverflowDataset).

## Available Models

### 1) Pre-trained MQDD model

First model that we make publicly available is the MQDD model, which is based on a Longformer architecture and is pre-trained on 218.5M training examples. The model was trained using MLM training objective acompanied with our novel Same Post (SP) and Question Answer (QA) learning objectives targetting specifically the duplicate detection task. 

The model can be obtained from our [HuggingFace repository](https://huggingface.co/UWB-AIR/MQDD-pretrained) and can be loaded using the following source code snippet:

```Python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-pretrained")
model = AutoModel.from_pretrained("UWB-AIR/MQDD-pretrained")
```

Besides, our model is also available at our [GoogleDrive folder](https://drive.google.com/drive/folders/1rSlb_wgb1kGP0ciDSWmpKSR9GDT0-273?usp=sharing).

### 2) Fine-tuned MQDD model (duplicates)

In addition to the pre-trained MQDD, we also release its fine-tuned version on duplicate detection task. The model's architectture follows the architecture of a two-tower model as depicted in the figure below:

![Two-tower model architecture](img/architecture.png)

A self-standing encoder without a duplicate detection head can be obtained from our [HuggingFace repository](https://huggingface.co/UWB-AIR/MQDD-duplicates). Such a model can be used for building search systems based, for example, on [Faiss](https://github.com/facebookresearch/faiss) library  model can be then loaded using the following source code:

```Python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-duplicates")
model = AutoModel.from_pretrained("UWB-AIR/MQDD-duplicates")
```

A checkpoint of a full two-tower model can than be obtained from our [GoogleDrive folder](https://drive.google.com/drive/folders/1CYiqF2GJ2fSQzx_oM4-X_IhpObi4af5Q?usp=sharing). To load the model, one needs to use the model's implementation from `models/MQDD_model.py` in our repository. To construct the model and load it's checkpoint, use the following source code:

```Python
from MQDD_model import ClsHeadModelMQDD

model = ClsHeadModelMQDD("UWB-AIR/MQDD-duplicates")
ckpt = torch.load("model.pt",  map_location="cpu")
model.load_state_dict(ckpt["model_state"])
```

### 3) Fine-tuned MQDD model (code search)

Besides our MQDD model for duplicates, we release the checkpoints of our MQDD model fine-tuned on the code search task. The model architecture employs the [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) from the `transformers` library. The checkpoints for all the programming languages are available in our [GoogleDrive folder](https://drive.google.com/drive/folders/1vdPN_W4FnkLSEAk0ldpmY36KCpBF_Nuz?usp=sharing). To load the model, use the following source code:

```Python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("UWB-AIR/MQDD-pretrained")
model = AutoModelForSequenceClassification.from_pretrained("UWB-AIR/MQDD-pretrained")

ckpt = torch.load("model.pt",  map_location="cpu")
model.load_state_dict(ckpt["model_state"])
```

### 4) Fine-tuned CodeBERT models
In addition to our MQDD models, we release fine-tuned [CodeBERT](https://github.com/microsoft/CodeBERT) models for both code search and duplicate detection task. For detailed instruction see the subsequent sections.

#### Duplicates

Checkpoints of the CodeBERT fine-tuned for the duplicity detection tasks can be downloaded from our [GoogleDrive folder](https://drive.google.com/drive/folders/1kN9EuEIFwX-U4CUOqg0EWKbUYDSVUHDE?usp=sharing). THe model can then be constructed and restored using the following source code:

```Python
from CodeBERT_model import ClsHeadModelCodeBERT

model = ClsHeadModelCodeBERT()
ckpt = torch.load("model.pt",  map_location="cpu")
model.load_state_dict(ckpt["model_state"])
```

#### Code Search

## Licence
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. http://creativecommons.org/licenses/by-nc-sa/4.0/

## How should I cite the MQDD? 
For now, please cite [the Arxiv paper](https://arxiv.org/abs/2203.14093):
```
@misc{https://doi.org/10.48550/arxiv.2203.14093,
  doi = {10.48550/ARXIV.2203.14093},
  url = {https://arxiv.org/abs/2203.14093},
  author = {Pašek, Jan and Sido, Jakub and Konopík, Miloslav and Pražák, Ondřej},
  title = {MQDD -- Pre-training of Multimodal Question Duplicity Detection for Software Engineering Domain},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```
