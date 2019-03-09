cd text_classification

cd a01_FastText

python p6_fastTextB_train_multilabel.py \
--embed_size 300 \
--cache_file_h5py ../../data/data.h5 \
--cache_file_pickle ../../data/vocab_label.pik \
--learning_rate 0.01 \
--decay_steps 10000 \
--decay_rate 0.5 \
# --use_embedding 1 
