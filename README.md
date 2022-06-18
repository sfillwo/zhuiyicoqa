## Source

This repo contains code publicly accessible from CodaLab -- [CoQA RoBERTa](https://worksheets.codalab.org/bundles/0x5cd8f55f02514cc2930865a1628a86c7) from user zhuiyi_betty

#### Usage
Docker image Requirement:
```
pytorch/pytorch     1.1.0-cuda10.0-cudnn7.5-devel   1be771bff6d1        3 months ago        6.94GB
```

Install fairseq:
```
cd fairseq
pip install --editable .
```

Install regex, spacy
```
pip install regex
pip install spacy
python -m spacy download en
```

Train:
```
export CUDA_VISIBLE_DEVICES=7
export PYTHONIOENCODING=utf-8
python main.py \
       --data_folder=/root/CoQA/ \
       --bert_dir=../roberta.large/ \
       --train_batch_size=48 \
       --predict_batch_size=100 \
       --gradient_accumulation_steps=6 \
       --learning_rate=4.5e-5 \
       --num_train_epochs=2 \
       --max_seq_length=512 \
       --doc_stride=128 \
       --output_dir=./experiments/011/ \
       --do_train \
       --warmup_proportion=0.06 \
       --fp16 \
       --lr_layer_decay_rate=0.9 \
```

Predict:
Do not modify bert_dir, modify the output_dir to the model you wanna load and use to do the inference.
```
export CUDA_VISIBLE_DEVICES=7
export PYTHONIOENCODING=utf-8
python main.py \
       --data_folder=/root/CoQA/ \
       --bert_dir=../roberta.large/ \
       --train_batch_size=48 \
       --predict_batch_size=100 \
       --gradient_accumulation_steps=6 \
       --learning_rate=4.5e-5 \
       --num_train_epochs=2 \
       --max_seq_length=512 \
       --doc_stride=128 \
       --output_dir=./experiments/011/ \
       --do_predict \
       --warmup_proportion=0.06 \
       --fp16 \
       --lr_layer_decay_rate=0.9 \
```