cd ../text_classification/a03_TextRNN

python -u p8_TextRNN_train.py --embed_size 300 --cache_file_h5py ../../data/data.h5 --cache_file_pickle ../../data/vocab_label.pik --learning_rate 0.01 --decay_steps 10000 --decay_rate 0.5 --num_epochs 25 --batch_size 128 2>&1 | tee ../../log/TextRNN_"$(date +%F)".log
