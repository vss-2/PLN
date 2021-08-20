from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['127.0.0.1'],
    port=9200
)

search = 'Lucas'

res = es.search(index="python-elasticsearch", body={"query": {"match": {"nome": ""+search+""}}})
print("\nA busca retornou: {} hits".format(res['hits']['total']['value']))

for hit in res['hits']['hits']:
    id = hit['_id']
    print(id)
    print("Nome: {}, skills: {}".format(hit["_source"]["nome"], hit["_source"]["skills"]))

    if id == "2":

        res = es.delete(index="python-elasticsearch", id="2")
        print('status: {}'.format(str(res['result'])))
