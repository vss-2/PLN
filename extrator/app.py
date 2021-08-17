import json
import requests
from flask import Flask, request, jsonify
from NER import testeNLP

app = Flask(__name__)

@app.route('/', methods=["POST"])
def processarMensagem():
    data = request.json
    # print("Mensagem recebida: " + format(data))
    output = dict({'entities': [], 'intent': []})
    for t in testeNLP(data['input']):
        if t[1] != 'intent':
            output['entities'].append(t[0])
        else:
            output['intent'].append(t[0])
    return jsonify(output)