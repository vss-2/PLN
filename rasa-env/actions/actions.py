from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from datetime import datetime
import requests
import json

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

    def searchCuriosity(search):
        res = es.search(index="python-pln-elasticsearch", body={"query": {"match": {"question": "" + search + ""}}})
        return "\n{}".format(res['hits']['hits'][0]["_source"]["answer"])
    
    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        message = tracker.latest_message.get('text')
        result = ActionCuriosity.searchCuriosity(message)

        dispatcher.utter_message(text=result)

        return []


def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
   return {
       "from_city": [
           self.from_entity(entity="city", role="from_city"),
       ],
       "destination_city": [
           self.from_entity(entity="city", role="destination_city"),
       ]
   }

class SubmitFlightForm(Action):
    
    def name(self) -> Text:
        return 'submit_flight_form'

    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        request_url = "http://localhost:8080/flights"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        data = {
            "source": tracker.get_slot("from_city"),
            "destiny": tracker.get_slot("destination_city"),
            "departure_time": tracker.get_slot("time")
        }

        try:
            response = requests.get(request_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
        
        response = json.loads(response.content)

        for flight in response:
            source = flight["source"]
            destiny = flight["destiny"]
            date = datetime.fromisoformat(flight["departure_time"]).strftime("%d/%m/%Y")
            time = datetime.fromisoformat(flight["departure_time"]).strftime("%H:%M")
            dispatcher.utter_message("- departure-city: {}, destination-city: {}, date: {}, time: {}".format(source, destiny, date, time))
        

        return [SlotSet("from_city", None), SlotSet("destination_city", None), SlotSet("time", None)]


class SubmitGroundForm(Action):
    
    def name(self) -> Text:
        return 'submit_ground_form'

    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        request_url = "http://localhost:8080/grounds"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "city": tracker.get_slot("ground_city")
        }

        try:
            response = requests.get(request_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
        
        response = json.loads(response.content)

        for ground in response:
            dispatcher.utter_message("- {}".format(ground))
        
        return [SlotSet("ground_city", None)]
