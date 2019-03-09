cd ../text_classification/a02_TextCNN

python -u p7_TextCNN_train.py --embed_size 300 --cache_file_h5py ../../data/data.h5 --cache_file_pickle ../../data/vocab_label.pik --learning_rate 0.01 --decay_steps 10000 --decay_rate 0.5 --num_epochs 25 2>&1 | tee ../../log/TextCNN_"$(date +%F)".log
