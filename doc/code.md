路径：`/home/songzy/mu_qa/question_classifier`

## 文件结构

* bak/: 纯粹是标注数据的备份，为了防止自己手抖。
* cnn-text-classification-tf/: 修改过的 Github 上的 CNN
* data/: 除标注之外的所有数据文件（因为觉得标注比较重要
* doc/: 只有一个振寰给的 es console 的使用方式说明
* grid_search.py: sklearn 中用于参数搜索时的一些代码
* label/: 标注数据
* model/: 保存的训练好的模型文件
* ngram_svm.py: svm 分类器的代码
* ngram_xgboost.py: xgboost 分类器的代码
* reference/: 修改之前的原 Github 文件（当时是为了防止自己手抖改坏
* run.sh: 一个完整的流程
* src/:  放了一些陈旧的代码
* svm_train/, svm_test/, svm_train_test/: 分别是从标注文件生成的符合 sklearn 输入格式的数据
* util_data.py: 标注数据时用到的一些辅助函数
* util.py: 一些从前有用但是现在没用的函数
* xgb_nn/: xgboost 和 neural network 的分类器

总结一下，现在有用的代码文件：

* ngram_svm.py: 线上的 SVM 分类器
* cnn-text-classification-tf/: CNN 分类器
* xgb_nn/: XGBOOST 分类器

这几个不同的分类器目前用的 feature 不太一样。

现在有用的数据文件：

* label/: 标好的数据
* svm_train, svm_test, svm_train_test: 直接用于训练或测试的数据
* model/: 训练后生成的模型

## 运行

### SVM

在 ngram_svm.py 的最后几行，`train()` 调用之后会在上面提到的 model/ 下生成模型文件。（这样方便保存下来以后直接调用）`grid = load_model()` 后就把 model/ 保存的文件重新加载回内存。`predict(grid, question)` 直接返回一个标签，`predict_proba` 会返回各个标签的概率分布。`code.interact()` 这句是用来进入交互式命令行的（调试时用）

## CNN

`data_helpers.py`: 把数据改成适合输入的格式。（目前已经有数据）

训练和测试：

```
python train.py
python test.py
```

（训练和测试的具体结构我还是需要再看一看

## XGB

直接运行 `python XGB.py`, 其实这个还不算完全写好。