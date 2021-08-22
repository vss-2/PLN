import os
import pickle
import numpy as np
import pandas as pd
import nltk
import spacy
import tensorflow as tf
from pandas import DataFrame
from keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM as LSTM
from nltk.chunk import tree2conlltags, ne_chunk
from nltk import word_tokenize
from subprocess import call

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# spacy_nlp = spacy.load("en_core_web_sm")

if 'input' not in os.listdir(os.getcwd()):
    call(['bash', 'init.sh'])

def main():
    DATA_DIR = "./input/atis"

    def load_ds(fname='atis.train.pkl'):
        with open(fname, 'rb') as stream:
            ds, dicts = pickle.load(stream)
        return ds, dicts


    train_ds, dicts = load_ds(os.path.join(DATA_DIR, 'atis.train.pkl'))
    test_ds, dicts = load_ds(os.path.join(DATA_DIR, 'atis.test.pkl'))

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
    query, slots, intent = map(
        train_ds.get, ['query', 'slot_labels', 'intent_labels'])

    labels = set()

    for i in range(len(query)):
        # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
        #                                 ' '.join(map(i2t.get, query[i]))))
        for j in range(len(query[i])):
            # print('{:>33} {:>40}'.format(i2t[query[i][j]],
            #                              i2s[slots[i][j]]  ))
            labels.add(i2s[slots[i][j]])
        # print('*'*100)

    query, slots, intent = map(
        test_ds.get, ['query', 'slot_labels', 'intent_labels'])

    for i in range(len(query)):
        # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
        #                                 ' '.join(map(i2t.get, query[i]))))
        for j in range(len(query[i])):
            # print('{:>33} {:>40}'.format(i2t[query[i][j]],
            #                              i2s[slots[i][j]]  ))
            labels.add(i2s[slots[i][j]])
        # print('*'*100)

    query, slots, intent = map(
        test_ds.get, ['query', 'slot_labels', 'intent_labels'])
    cidades = []
    aeroportos = []
    for i in range(len(query)-800):

        # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
        # ' '.join(map(i2t.get, query[i]))))
        for j in range(len(query[i])):
            if i2t[query[i][j]] not in ['BOS', 'EOS']:
                if str(i2s[slots[i][j]]).endswith('city_name'):
                    cidades.append(i2t[query[i][j]])
                if str(i2s[slots[i][j]]).endswith('airport_name'):
                    aeroportos.append(i2t[query[i][j]])
                # print(i2s[slots[i][j]], end=' ')
            # print('{:>33} {:>40}'.format(i2t[query[i][j]],
                    #  i2s[slots[i][j]]  ))
        # print('\n', '*'*100)
    # cidades = set(cidades)
    # print(cidades)

    frases = [list(dicts['token_ids'].keys())[i]
            for t in train_ds['query'] for i in t]
    c = 10
    x = 0
    # Altere o C e descomente para printar C frases
    # print('{} Frases:\n'.format(c))
    # for f in frases:
    #     if f == 'EOS':
    #         print('\n', 'Intent: ', list(dicts['intent_ids'].keys())[
    #             train_ds['intent_labels'][x][0]], '\n', sep='')
    #         x = x + 1
    #         if x == c:
    #             break
    #     elif f == 'BOS':
    #         pass
    #     else:
    #         print(f, sep='', end=' ')

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    txt = ''
    x = 0
    frases = [list(dicts['token_ids'].keys())[i]
            for t in train_ds['query'] for i in t]
    for f in frases:
        if f == 'EOS':
            y_train.append(list(dicts['intent_ids'].keys())[
                        train_ds['intent_labels'][x][0]])
            x = x + 1
            X_train.append(txt.strip())
        elif f == 'BOS':
            txt = ''
            pass
        else:
            txt = str(txt) + ' ' + str(f)

    txt = ''
    x = 0
    frases = [list(dicts['token_ids'].keys())[i]
            for t in test_ds['query'] for i in t]
    for f in frases:
        if f == 'EOS':
            y_test.append(list(dicts['intent_ids'].keys())
                        [test_ds['intent_labels'][x][0]])
            x = x + 1
            X_test.append(txt.strip())
        elif f == 'BOS':
            txt = ''
            pass
        else:
            txt = str(txt) + ' ' + str(f)

    max_length = max([len(x) for x in X_train + X_test])
    vocab_size = len(dicts['token_ids'])

    encoder = TextVectorization(max_tokens=vocab_size)
    encoder.adapt(X_train+y_train)
    vocab_size = encoder.vocabulary_size()

    def get_intents(Y_train, num_intents):
        count = 0
        intent_map = {}
        intents = [""]*num_intents
        for idx, intent in enumerate(Y_train):
            if intent not in intent_map.keys():
                intents[count] = intent
                intent_map[intent] = count
                count += 1
        return (intents, intent_map)


    def encode_intents(Y_train, intent_map):
        intent_list = []
        for idx, intent in enumerate(Y_train):
            intent_list.append([0]*len(intent_map))
            intent_list[idx][intent_map[intent]] = 1
        return intent_list


    num_intents = len(dicts['intent_ids'])
    intents, intent_map = get_intents(
        list(dicts['intent_ids'].keys()), len(set(list(dicts['intent_ids'].keys()))))
    y_train = encode_intents(y_train, intent_map)
    y_test = encode_intents(y_test, intent_map)

    y_train = pad_sequences(y_train, maxlen=num_intents, padding='pre')
    y_test = pad_sequences(y_test, maxlen=num_intents, padding='pre')

    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=vocab_size)

    X_train = DataFrame(X_train)
    X_test = DataFrame(X_test)
    encoder.adapt(X_train.values)

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    running_model = Sequential([
        encoder,
        Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64
        ),
        Bidirectional(LSTM(64)),
        Dense(num_intents, activation='softmax')
    ])

    running_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy', f1_m, precision_m, recall_m])
    print('Tamanho Xtrain e ytrain:', len(X_train), len(y_train))

    # model.fit(X_train, y_train, epochs=20)
    running_model.fit(X_train, y_train, epochs=50)

    loss, accuracy, x, y, z = running_model.evaluate(X_train, y_train, verbose=1)
    print('Training Accuracy is {}'.format(accuracy*100, x*100, y*100,z*100))

    loss, accuracy, x, y, z = running_model.evaluate(X_test, y_test, verbose=1)
    print('Test Accuracy is {}'.format(accuracy*100, x*100, y*100,z*100))

    def file2Examples(f):
        '''
        Read data files and return input/output pairs
        '''

        examples = []

        try:
            example = [[], []]

            for line in f.values:

                input_output_split = line.split()

                if len(input_output_split) == 4:
                    example[0].append(input_output_split[0])
                    example[1].append(input_output_split[-1])
                    # labels.add(input_output_split[-1])

                elif len(input_output_split) == 0:
                    examples.append(example)
                    example = [[], []]
                else:
                    example = [[], []]

            f.close()

        except:
            # print('Erro', line)
            pass

        return examples


    file2Examples(X_train)
    file2Examples(X_test)

    all_text = " ".join([" ".join(x[0]) for x in X_train.values])
    vocab = sorted(set(all_text))

    char2idx = {u: i+1 for i, u in enumerate(vocab)}
    char2idx2 = {u.upper(): i+1 for i, u in enumerate(vocab)}
    char2idx = {**char2idx2, **char2idx}
    idx2char = np.array(vocab)
    label2idx = {u: i+1 for i, u in enumerate(labels)}
    idx2label = np.array(labels)

    print(label2idx)
    print(idx2char)
    print(idx2label)
    print(char2idx)

    queryTrain, _void, __void = map(
        train_ds.get, ['query', 'slot_labels', 'intent_labels'])
    queryTest, _void, __void = map(
        test_ds.get, ['query', 'slot_labels', 'intent_labels'])

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
    query, slots, intent = map(
        train_ds.get, ['query', 'slot_labels', 'intent_labels'])

    labels = set()

    # Descomente se quiser ver frases
    # for i in range(len(query)):
    #     # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
    #     #                                 ' '.join(map(i2t.get, query[i]))))
    #     for j in range(len(query[i])):
    #         # print('{:>33} {:>40}'.format(i2t[query[i][j]],
    #         #                              i2s[slots[i][j]]  ))
    #         labels.add(i2s[slots[i][j]])
    #     # print('*'*100)

    query, slots, intent = map(
        train_ds.get, ['query', 'slot_labels', 'intent_labels'])

    # for i in range(len(query)):
    #     # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
    #     #                                 ' '.join(map(i2t.get, query[i]))))
    #     for j in range(len(query[i])):
    #         # print('{:>33} {:>40}'.format(i2t[query[i][j]],
    #         #                              i2s[slots[i][j]]  ))
    #         labels.add(i2s[slots[i][j]])
    #     # print('*'*100)

    palavras_train = dict()
    palavras_test = dict()

    test, train = [], []

    for i in range(len(query)):
        k, x = [], []
        for j in range(len(query[i])):
            labels.add(i2s[slots[i][j]])
            if i2t[query[i][j]] not in palavras_train.keys():
                palavras_train.update({i2t[query[i][j]]: set()})
                palavras_train[i2t[query[i][j]]].add(i2s[slots[i][j]])
            else:
                palavras_train[i2t[query[i][j]]].add(i2s[slots[i][j]])
            x.append(i2s[slots[i][j]])
            k.append(i2t[query[i][j]])
        train.append([k, x])

    query, slots, intent = map(
        test_ds.get, ['query', 'slot_labels', 'intent_labels'])
    for i in range(len(query)):
        k, x = [], []
        for j in range(len(query[i])):
            labels.add(i2s[slots[i][j]])

            if i2t[query[i][j]] not in palavras_test.keys():
                palavras_test.update({i2t[query[i][j]]: set()})
                palavras_test[i2t[query[i][j]]].add(i2s[slots[i][j]])
            else:
                palavras_test[i2t[query[i][j]]].add(i2s[slots[i][j]])
            x.append(i2s[slots[i][j]])
            k.append(i2t[query[i][j]])
        test.append([k, x])

    def split_char_labels(eg):
        tokens = eg[0]
        labels = eg[1]
        input_chars = []
        output_char_labels = []
        for t, l in zip(tokens, labels):
            input_chars.extend([char for char in t])
            input_chars.extend(' ')
            output_char_labels.extend([l] * len(t))
            output_char_labels.extend('O')

        a = [char2idx[x] for x in input_chars[:-1]]
        return [a]
    # [[frase], [resultado]]


    train_formatado = [split_char_labels(eg) for eg in train]
    test_formatado = [split_char_labels(eg) for eg in test]
    # valido_formatado = [split_char_labels(eg) for eg in train]
    print(len(train_formatado))
    print(len(test_formatado))

    def gen_train_series():

        for eg in train_formatted:
            yield eg[0], eg[1]

    def gen_valid_series():

        for eg in valid_formatted:
            yield eg[0],eg[1]

    def gen_test_series():

        for eg in test_formatted:
            yield eg[0], eg[1]
    

    series = tf.data.Dataset.from_generator(gen_train_series, output_types=(
        tf.int32, tf.int32), output_shapes=((None, None)))
    series_valid = tf.data.Dataset.from_generator(gen_valid_series, output_types=(
        tf.int32, tf.int32), output_shapes=((None, None)))
    series_test = tf.data.Dataset.from_generator(gen_test_series, output_types=(
        tf.int32, tf.int32), output_shapes=((None, None)))

    BATCH_SIZE = 128
    BUFFER_SIZE = 1000

    
    ds_series_batch = series.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)
    ds_series_batch_valid = series_valid.padded_batch(
        BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)
    ds_series_batch_test = series_test.padded_batch(
        BATCH_SIZE, padded_shapes=([None], [None]), drop_remainder=True)

    vocab_size = len(vocab)+1

    embedding_dim = 256
    rnn_units = 1024
    label_size = len(labels)

    # model.summary()
    return running_model, intent_map

predictions, intents_to_predict = None, None
predictions, intents_to_predict = main()

def entities_extract(frase):
    nnp = nltk.pos_tag(word_tokenize(frase))
    # Padrão: 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser("""
                        VP: {<ADJ_SIM><V_PRS>}
                        VP: {<ADJ_INO><V.*>}
                        VP: {<V_PRS><N_SING><V_SUB>}
                        NP: {<N_SING><ADJ.*><N_SING>}
                        NP: {<N.*><PRO>}
                        VP: {<N_SING><V_.*>}
                        VP: {<V.*>+}
                        NP: {<ADJ.*>?<N.*>+ <ADJ.*>?}
                        DNP: {<DET><NP>}
                        PP: {<ADJ_CMPR><P>}
                        PP: {<ADJ_SIM><P>}
                        PP: {<P><N_SING>}
                        PP: {<P>*}
                        DDNP: {<NP><DNP>}
                        NPP: {<PP><NP>+}
                        """)
    cs = cp.parse(nnp)
    iob_tagged = tree2conlltags(cs)
    print(iob_tagged, '\n\n\n')
    ne_tree = ne_chunk(nltk.pos_tag(word_tokenize(frase)))
    print(ne_tree)

def testeNLP(frase: str = 'I would like to find a flight from Houston to Dallas'):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(frase)
    
    result = predictions.predict([frase])
    result = list(result[0])
    intents = list(intents_to_predict.values())
    intention = list(intents_to_predict)[intents[result.index(max(result))]]
    print('Extracted intention: ', intention)
    
    print(*[(X.text, X.label_) for X in doc.ents])
    entidades = [(X.text, X.label_) for X in doc.ents]
    entidades.append([intention, 'intent'])
    return entidades

