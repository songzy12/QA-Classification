cd ../text_classification/a01_FastText

python -u p6_fastTextB_train_multilabel.py --embed_size 300 --cache_file_h5py ../../data/data.h5 --cache_file_pickle ../../data/vocab_label.pik --learning_rate 0.01 --decay_steps 10000 --decay_rate 0.5 2>&1 | tee ../../log/FastText_"$(date +%F)".log
