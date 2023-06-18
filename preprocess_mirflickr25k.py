import os
import shutil
import imageio

import numpy as np
from tqdm import tqdm

mirflickr25k_data_dir = './data'

def get_mirflickr_tag_map():
    tag2id = dict()
    id2tag = dict()

    with open(mirflickr25k_data_dir + '/mirflickr25k/doc/common_tags.txt',encoding='UTF-8') as f:
        tag_names = [l.split()[0] for l in f.readlines()]
    for i, name in enumerate(tag_names):
        tag2id[name] = i
        id2tag[i] = name

    print(f'text dimension: {len(tag2id)}')

    return tag2id, id2tag


def get_mirflickr_label_map():
    label2id = dict()
    id2label = dict()

    for root, _, filenames in os.walk(mirflickr25k_data_dir + '/mirflickr25k_annotations_v080/remove_r1'):
        for i, filename in enumerate(filenames):
            label = filename[:-4]
            label2id[label] = i
            id2label[i] = label

    print(f'label dimension: {len(label2id)}')

    return label2id, id2label


def remove_r1_and_readme():
    origin_path = mirflickr25k_data_dir + '/mirflickr25k_annotations_v080'
    processed_path = mirflickr25k_data_dir + '/mirflickr25k_annotations_v080/remove_r1'

    os.makedirs(processed_path, exist_ok=True)

    # copy & filter files
    for root, directory, filenames in os.walk(origin_path):
        for file in filenames:
            if '_r1' in file or file == 'README.txt':
                continue

            origin = os.path.join(origin_path, file)
            shutil.copy2(origin, processed_path)


def process_tags():
    tag2id, id2tag = get_mirflickr_tag_map()
    text_dim = len(tag2id)

    for root, _, filenames in os.walk(mirflickr25k_data_dir + '/mirflickr25k/meta/tags'):
        print(1)
        #词袋模型（Bag-of-Words model，BOW）BoW(Bag of Words)词袋模型最初被用在文本分类中，将文档表示成特征矢量。
        # 它的基本思想是假定对于一个文本，忽略其词序和语法、句法，仅仅将其看做是一些词汇的集合，而文本中的每个词汇都是独立的。
        texts = np.zeros((len(filenames), text_dim), dtype=np.float32)# BOW
        # x[4:-4]表示的是 数字，tag 的数字
        filenames.sort(key=lambda x: int(x[4:-4]))

        for i, filename in enumerate(filenames):
            file = os.path.join(root, filename)
            with open(file,encoding='UTF-8') as f:
                #把一张图片的tag全部读出，存入Lines，它是一个list
                lines = f.readlines()
                #l.split() 是将字符串 l 按照空格进行分割，并返回分割后的字符串列表
                tags = list(map(lambda l: l.split()[0], lines))
                for tag in filter(lambda x: x in tag2id.keys(), tags):
                    index = tag2id[tag]
                    texts[i][index] = 1
    print(f'Saving texts(tags) with shape: {texts.shape}')
    np.save(mirflickr25k_data_dir + '/texts.npy', texts)


def process_labels():
    label2id, id2label = get_mirflickr_label_map()
    label_dim = len(label2id)

    label_map = dict()
    for root, _, files in os.walk(mirflickr25k_data_dir + '/mirflickr25k_annotations_v080/remove_r1'):
        for file in files:
            with open(os.path.join(root, file)) as f:
                label_map[file[:-4]] = set(map(int, f.readlines()))

    def get_label(number):
        for l, s in label_map.items():
            if number in s:
                return l

    total_num = 25000
    labels = np.zeros((total_num, label_dim), dtype=np.float32)  # one-hot labels
    for i in range(1, total_num + 1):
        label = get_label(i)
        if label not in label2id.keys():
            continue

        index = label2id[label]
        labels[i - 1][index] = 1  # start from 0 instead of 1

    print(f'Saving labels with shape: {labels.shape}')
    np.save(mirflickr25k_data_dir + '/labels.npy', labels)


def process_images():
    image_folder = mirflickr25k_data_dir + '/mirflickr25k'
    images = []

    filenames = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    #就是按照序号大小排列   2:-4 其实就是 im1.jpg   1就是2  而-4 就是 .
    filenames.sort(key=lambda x: int(x[2:-4]))
    for filename in tqdm(filenames, desc='Image loading'):
        image = imageio.imread(os.path.join(image_folder, filename), pilmode='RGB')
        images.append(np.asarray(image))

    images = np.array(images, dtype=object)

    print(f'Saving images with shape: {images.shape}')
    np.save(mirflickr25k_data_dir + '/images.npy', images, allow_pickle=True)


def clean():
    texts = np.load(f'{mirflickr25k_data_dir}/texts.npy')
    labels = np.load(f'{mirflickr25k_data_dir}/labels.npy')
    images = np.load(f'{mirflickr25k_data_dir}/images.npy', allow_pickle=True)

    # Remove elements that without text or label (text/label vector is all-zero vector)
    delete_indexes = [i for (i, (t, l)) in enumerate(zip(texts, labels)) if not any(t) or not any(l)]

    print(f'Data left: {len(texts) - len(delete_indexes)}')

    # Get (20015, ...)
    texts = np.delete(texts, delete_indexes, axis=0)
    labels = np.delete(labels, delete_indexes, axis=0)
    images = np.delete(images, delete_indexes, axis=0)

    np.save(f'{mirflickr25k_data_dir}/texts.npy', texts)
    np.save(f'{mirflickr25k_data_dir}/labels.npy', labels)
    np.save(f'{mirflickr25k_data_dir}/images.npy', images, allow_pickle=True)

    print('Data clean done.')


def main():
    process_labels()


if __name__ == '__main__':
    main()
