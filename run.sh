cd text_classification

cd a01_FastText
python p6_fastTextB_train_multilabel.py --embed_size 300 --cache_file_h5py ../../data/data.h5 --cache_file_pickle ../../data/vocab_label.pik
