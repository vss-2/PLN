from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['127.0.0.1'],
    port=9200
)

search = 'What is the maximum capacity of the Zeppelin-Staaken R.VI?'


def searchCuriosity(search):
    res = es.search(index="python-pln-elasticsearch", body={"query": {"match": {"question": "" + search + ""}}})
    return "\n{}".format(res['hits']['hits'][0]["_source"]["answer"])


print(searchCuriosity(search))
