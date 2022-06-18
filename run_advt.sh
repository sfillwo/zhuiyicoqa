dir=../models/206
dir=${dir}/
export PYTHONIOENCODING=utf-8
python main_advtrain.py \
       --data_folder=./data/ \
       --train_batch_size=48 \
       --predict_batch_size=100 \
       --learning_rate=5e-5 \
       --lr_layer_decay_rate=0.9 \
       --num_train_epochs=2 \
       --max_seq_length=512 \
       --gradient_accumulation_steps=6 \
       --doc_stride=128 \
       --output_dir=${dir} \
       --bert_dir=./roberta_large/ \
       --warmup_proportion=0.06 \
       --lr_layer_decay_rate=0.9 \
       --do_predict \
       --fp16 \
    