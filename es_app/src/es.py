import json
from flask import Flask
from flask import request
from flask import Response
from schema import Schema, And
from elasticsearch import Elasticsearch

app = Flask(__name__)

MAX = 2147483647

es = Elasticsearch("http://localhost:9200")  # type: Elasticsearch


@app.route('/fetch', methods=['POST'])
def fetch_data_from_es():
    global es
    global MAX
    schema = Schema(dict(db=And(str, len), fields=And(list, lambda n: len(n) == 2)))
    try:
        params = schema.validate(request.json)
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json', status=400)
    pluck = lambda dict, *args: (dict[arg] for arg in args)
    db, fields = pluck(params, 'db', 'fields')  # destructuring of a map
    res = es.search(index=db, body={
        "query": {
            "bool": {
                "filter": {
                    "bool": {
                        "should": [
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "terms": {
                                                str(fields[0]) + '.keyword': ["1st"]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        },
        "aggs": {
            "c-id": {
                "terms": dict(field=fields[1], size=MAX)
            }
        },
        "from": 0,
        "size": 0
    })
    return Response(json.dumps(res), mimetype='application/json', status=200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
