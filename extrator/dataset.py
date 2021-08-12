import os
import numpy as np
import pandas as pd

import pickle
from subprocess import call

if 'input' not in os.listdir(os.getcwd()):
    call(['bash', 'init.sh'])

DATA_DIR="./input/atis"

def load_ds(fname='atis.train.pkl'):
    with open(fname, 'rb') as stream:
        ds, dicts = pickle.load(stream)
    print('Done  loading: ', fname)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds,dicts

train_ds, dicts = load_ds(os.path.join(DATA_DIR,'atis.train.pkl'))
test_ds, dicts  = load_ds(os.path.join(DATA_DIR,'atis.test.pkl'))

t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
query, slots, intent =  map(train_ds.get, ['query', 'slot_labels', 'intent_labels'])

labels = set()

for i in range(len(query)):
    # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
    #                                 ' '.join(map(i2t.get, query[i]))))
    for j in range(len(query[i])):
        # print('{:>33} {:>40}'.format(i2t[query[i][j]],
        #                              i2s[slots[i][j]]  ))
        labels.add(i2s[slots[i][j]])
    # print('*'*100)

query, slots, intent =  map(test_ds.get, ['query', 'slot_labels', 'intent_labels'])

for i in range(len(query)):
    # print('{:4d}:{:>15}: {}'.format(i, i2in[intent[i][0]],
    #                                 ' '.join(map(i2t.get, query[i]))))
    for j in range(len(query[i])):
        # print('{:>33} {:>40}'.format(i2t[query[i][j]],
        #                              i2s[slots[i][j]]  ))
        labels.add(i2s[slots[i][j]])
    # print('*'*100)

### PRINTANDO DO TESTE
query, slots, intent =  map(test_ds.get, ['query', 'slot_labels', 'intent_labels'])
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

# print('Token: ', dicts['token_ids'])
# print('Slot: ', dicts['slot_ids'])
# print('Intents:\n', *[i for i in dicts['intent_ids']], sep='\n')
frases = [list(dicts['token_ids'].keys())[i] for t in train_ds['query'] for i in t]
c = 10
x = 0
print('{} Frases:\n'.format(c))
for f in frases:
  if f == 'EOS':
    print('\n', 'Intent: ', list(dicts['intent_ids'].keys())[train_ds['intent_labels'][x][0]], '\n', sep='')
    x = x + 1
    if x == c:
      break
  elif f == 'BOS':
    pass
  else:
    print(f, sep='', end=' ')