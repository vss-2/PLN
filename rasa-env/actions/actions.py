from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['127.0.0.1'],
    port=9200
)

class ActionAirfareFind(Action):
    
    def name(self) -> Text:
        return 'find_flight'
    
    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Achei o voo para você")
        return []

class ActionCuriosity(Action):
    
    def name(self) -> Text:
        return 'curiosity'
    
    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        message = tracker.latest_message.get('text')
        result = searchCuriosity(message)

        dispatcher.utter_message(text=result)

        return []

    def searchCuriosity(search):
        res = es.search(index="python-pln-elasticsearch", body={"query": {"match": {"question": "" + search + ""}}})
        return "\n{}".format(res['hits']['hits'][0]["_source"]["answer"])