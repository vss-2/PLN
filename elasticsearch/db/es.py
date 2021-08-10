from elasticsearch import Elasticsearch
from database import fetchCidades

def main():
    IP, PORT = '127.0.0.1', '8090'
    es = Elasticsearch([IP], port = PORT)
    for cidade in fetchCidades():
        c = dict({'City name': cidade})
        res = es.index(index='CITIES', id=1, body=c)
        print('Insert: ', res['status'])
    res = es.get(index='CITIES', id=1)
    print('Results: ', res['_source'])

main()
