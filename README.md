代码位置：`taoli:/home/songzy/Xiaomu-Classification`

这个是用来比较不同模型在我们标注的数据集上表现的代码。

## code

- `data/`: 除标注之外的所有数据文件（因为觉得标注比较重要
  - `label/`: 手工标注的数据
  - `svm/`: 从标注文件生成的符合 sklearn 输入格式的数据
- `model/`: 保存的训练好的模型文件
- `sklearn_baseline`: 一些非神经网络的模型实现，如 svm, xgboost, lightgbm 等
- `text_classification`: fork 的某个 repo, 但是原 repo 无法直接运行，进行了一些修改。

## sklearn_baseline

运行命令可见 `sklearn_baseline/run.sh`.

```
python baseline.py 2>&1 | tee ../log/baseline_$(date +%F).log
```

上面的命令会将各个模型的 performance 打印在 console 里，

并通过 tee 重定向至 log 文件。

## text_classification

可以参见 `script/` 里的几个运行脚本。

目前应该是只修改了 FastText, TextCNN, TextRNN 这几个 model. 

之后可以根据自己的需要来使得原 repo 里更多的 model 跑起来。