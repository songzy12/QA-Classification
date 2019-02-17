from collections import defaultdict, Counter
import io
import json
import os


def merge_label_file(path_input, path_output):
    m = {}

    try:
        with open(path_output) as f:
            m.update(json.loads(f.read()))
    except:
        pass

    for file_ in path_input:
        with io.open("../data/label/%s" % file_, encoding='utf8') as f:
            m.update(json.loads(f.read()))

    print(path_output, len(m))
    with io.open(path_output, 'w', encoding='utf8') as f:
        f.write(json.dumps(m, ensure_ascii=False, indent=4))


def generate_dataset(path_input='label/train_test.json', path_output='svm_train_test'):
    with io.open(path_input, encoding='utf8') as f:
        m = json.loads(f.read())
        m_ = defaultdict(list)

        for k, v in m.items():
            m_[v].append(k)

        for k, vs in m_.items():
            for i, v in enumerate(vs):
                filename = "%s/%s/%s.txt" % (path_output, str(k), str(i))
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except:
                        raise
                with io.open(filename, "w", encoding='utf8') as f:
                    f.write(v)


if __name__ == '__main__':
    label_files = ['label_course.json',
                   'label_common.json',
                   'label_2017-03-09_2017-04-14.json',
                   'label_2017-04-14_2017-05-10.json',
                   'label_2017-05-10_2017-06-09.json',
                   'label_2017-06-05_2017-07-02.json',
                   'label_2017-07-03_2017-08-06.json',
                   'label_2017-08-07_2017-09-03.json',
                   'label_2017-09-04_2017-10-01.json',
                   'label_2017-10-02_2017-11-05.json',
                   'label_2017-11-06_2017-12-03.json',
                   'label_2017-12-04_2017-12-31.json',
                   ]

    for label_file in label_files:
        print(label_file)
        with io.open("../data/label/%s" % label_file, encoding='utf8') as f:
            content = json.loads(f.read())
            m = defaultdict(list)
            for k, v in content.items():
                m[v].append(k)
            for k, v in m.items():
                print(k, len(v))

    merge_label_file(
        path_input=label_files[:-1], path_output='../data/svm/train.json')
    merge_label_file(
        path_input=label_files[-1:], path_output='../data/svm/test.json')

    generate_dataset('../data/svm/train.json', '../data/svm/train')
    generate_dataset('../data/svm/test.json', '../data/svm/test')
