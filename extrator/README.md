## Extrator e Classificador

### Como utilizar (no bash)
```bash
# Ative a virtual-enviroment
source ./bin/activate
# Execute o flask (em outro terminal)
flask run
# Envie requisições POST para http://127.0.0.1:5000/ no formato exemplo:
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"input": "I would like to find informations about flights from New York to San Francisco"}' \
  http://127.0.0.1:5000/
# Seu output será:
{"entities":["New York","San Francisco"],"intent":["flight"]}

```

