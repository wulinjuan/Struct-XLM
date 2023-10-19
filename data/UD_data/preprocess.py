import csv
import os
import random

pairs_label = []


def generate_lines_for_sent(lines):
    '''Yields batches of lines describing a sentence in conllx.

    Args:
      lines: Each line of a conllx file.
    Yields:
      a list of lines describing a single sentence in conllx.
    '''
    buf = []
    id = ''
    for line in lines:
        if line.startswith('#'):
            if line.startswith('# sent_id = '):
                id = line.strip().replace("# sent_id = ", "")
            continue
        if not line.strip():
            if buf:
                yield id, buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield id, buf


def load_conll_dataset(filepath):
    """Reads in a conllx file; generates Observation objects

    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset

    Returns:
      A list of Observations
    """
    sentence_trees = []

    lines = (x for x in open(filepath, 'r', encoding='utf-8'))
    for id, buf in generate_lines_for_sent(lines):
        if id == "":
            continue
        conllx_lines = []
        for line in buf:
            if "-" in line.strip().split("\t")[0] and "_	_	_	_" in line or "." in line.strip().split("\t")[0]:
                continue
            word = line.strip().split('\t')[1]
            tag = line.strip().split('\t')[3]
            tree_next_idx = line.strip().split('\t')[6]
            # punc = line.strip().split('\t')[4]
            conllx_lines.append((word, tag, tree_next_idx))
        sentence_trees.append((id, conllx_lines))
    return sentence_trees


def tree_to_actor(sentence_tree):
    pun = ["《", "》", "〈", "〉", "、", "“"]
    actor = []
    sentence = []
    i = 0
    tail = -1
    while i < len(sentence_tree):
        info = sentence_tree[i]
        word = info[0]
        sentence.append(word)

        tag = info[1]
        if tag == 'PUNCT' and word not in pun:
            actor.append(2)
            tail = -1
            i += 1
            continue

        next_idx = int(info[2])
        if next_idx == 0 and i < len(sentence_tree) - 1:
            if sentence_tree[i + 1][1] == 'PUNCT' and sentence_tree[i + 1][0] not in pun:
                if i - 1 >= 0:
                    actor[i - 1] = 0
            actor.append(1)
            tail = i
            i += 1
            continue
        if next_idx < i:
            actor.append(1)
            tail = i
            i += 1
            continue
        if next_idx == i:
            if i - 1 >= 0:
                actor[i - 1] = 0
            actor.append(1)
            tail = i
            i += 1
            continue
        actor.append(0)
        i += 1
        while i < next_idx:
            word1 = sentence_tree[i][0]
            sentence.append(word1)

            tag1 = sentence_tree[i][1]
            if tag1 == 'PUNCT' and word1 not in pun:
                if i - 1 >= 0:
                    actor[i - 1] = 1
                actor.append(2)
                tail = -1
                i += 1
                break
            if i == next_idx - 1:
                next_idx1 = int(sentence_tree[i][2])
                if next_idx1 - 1 == tail:
                    actor[tail] = 0
                actor.append(1)
                tail = i
            else:
                actor.append(0)
            i += 1

    return sentence, actor


def en_other(en_sentence_trees, other_sentence_trees, writer, writer_dev=None, writer_test=None):
    i = 0
    j = 0
    count = 0
    if writer_dev is not None:
        cont_dev = 0
        cont_test = 0
    while j < len(other_sentence_trees) and i < len(en_sentence_trees):
        en_id, en_tree = en_sentence_trees[i]
        other_id, other_tree = other_sentence_trees[j]
        en_id = en_id.split("_")[-1]
        other_id = other_id.split("_")[-1]
        if en_id != other_id:
            i += 1
            continue
        if en_id not in pairs_label:
            pairs_label.append(en_id)
            en_id = len(pairs_label) - 1
        else:
            en_id = pairs_label.index(en_id)
        i += 1
        j += 1
        en_sentenc, en_actor = tree_to_actor(en_tree)
        if 20 > len(en_sentenc) or len(en_sentenc) > 100:
            continue
        other_sentenc, other_actor = tree_to_actor(other_tree)
        if writer_dev is not None and random.randint(1, 1000) < 300:
            cont_dev += 1
            writer_dev.writerow([en_id, en_sentenc, other_sentenc, en_actor, other_actor])
            if cont_dev == 20:
                writer_dev = None
        elif writer_test is not None and random.randint(1, 1000) < 300:
            cont_test += 1
            writer_test.writerow([en_id, en_sentenc, other_sentenc, en_actor, other_actor])
            if cont_test == 20:
                writer_test = None
        else:
            writer.writerow([en_id, en_sentenc, other_sentenc, en_actor, other_actor])
        count += 1
    print(count)


if __name__ == '__main__':
    train_path = './train.csv'
    dev_path = './dev.csv'
    test_path = './test.csv'

    train_file = open(train_path, 'w', encoding='utf-8')
    dev_file = open(dev_path, 'w', encoding='utf-8')
    test_file = open(test_path, 'w', encoding='utf-8')

    writer_train = csv.writer(train_file)
    writer_dev = csv.writer(dev_file)
    writer_test = csv.writer(test_file)

    path = './'
    path_list = os.listdir(path)
    for filename in path_list:
        if "." in filename:
            continue
        file_path = os.path.join(path, filename)
        data_file_list = os.listdir(file_path)
        en_sentences_train = []
        other_sentences_train = []
        en_sentences_dev = []
        other_sentences_dev = []
        en_sentences_test = []
        other_sentences_test = []

        for data_file_name in data_file_list:
            if "English" in data_file_name:
                en_file_path = os.path.join(file_path, data_file_name)
                en_list = os.listdir(en_file_path)
                for en_file in en_list:
                    if '.conllu' in en_file:
                        if 'train' in en_file:
                            filepath = os.path.join(en_file_path, en_file)
                            en_sentences_train = load_conll_dataset(filepath)
                        elif 'dev' in en_file:
                            filepath = os.path.join(en_file_path, en_file)
                            en_sentences_dev = load_conll_dataset(filepath)
                        elif 'test' in en_file:
                            filepath = os.path.join(en_file_path, en_file)
                            en_sentences_test = load_conll_dataset(filepath)
                break
        for data_file_name in data_file_list:
            filepath1 = None
            filepath2 = None
            filepath3 = None
            if "English" not in data_file_name and filename != "PUD":
                other_file_path = os.path.join(file_path, data_file_name)
                other_list = os.listdir(other_file_path)
                for other_file in other_list:
                    if '.conllu' in other_file:
                        if 'train' in other_file:
                            filepath1 = os.path.join(other_file_path, other_file)
                            other_sentences_train = load_conll_dataset(filepath1)
                        elif 'dev' in other_file:
                            filepath2 = os.path.join(other_file_path, other_file)
                            other_sentences_dev = load_conll_dataset(filepath2)
                        elif 'test' in other_file:
                            filepath3 = os.path.join(other_file_path, other_file)
                            other_sentences_test = load_conll_dataset(filepath3)

                if en_sentences_train and other_sentences_train:
                    en_other(en_sentences_train, other_sentences_train, writer_train)
                    print(filepath1)
                if en_sentences_dev and other_sentences_dev:
                    en_other(en_sentences_dev, other_sentences_dev, writer_dev)
                    print(filepath2)
                if en_sentences_test and other_sentences_test:
                    en_other(en_sentences_test, other_sentences_test, writer_test)
                    print(filepath3)
            elif "English" not in data_file_name and filename == "PUD":
                other_file_path = os.path.join(file_path, data_file_name)
                other_list = os.listdir(other_file_path)
                for other_file in other_list:
                    if '.conllu' in other_file:
                        filepath = os.path.join(other_file_path, other_file)
                        other_sentences_test = load_conll_dataset(filepath)
                if en_sentences_test and other_sentences_test:
                    en_other(en_sentences_test, other_sentences_test, writer_train, writer_dev, writer_test)
                    print(filepath)
        print(filename, ' finish!!!')
    print("label num is ", len(pairs_label))
    # label num is  13765

    # filepath = "zh_gsdsimp-ud-test.conllu"
    # sentence_trees = load_conll_dataset(filepath)
    # for id, sentence_tree in sentence_trees:
    #     sentenc, actor = tree_to_actor(sentence_tree)
