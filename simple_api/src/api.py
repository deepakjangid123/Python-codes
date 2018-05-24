import json
from flask import Flask
from flask import make_response
from flask import jsonify
from flask import request
from flask import Response
from schema import Schema, And, Use, Optional

app = Flask(__name__)


@app.route('/')  # We can define here method as well, like @app.route('/', methods = ['GET'])
def hello():
    return "Hello Babe!!!"


@app.route('/not-found')
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/request-validate', methods=['POST'])
def req_validator():
    schema = Schema({'title': And(str, len),
                     'age': And(Use(int), lambda n: 18 <= n <= 99),
                     Optional('gender'): And(str, Use(str.lower), lambda s: s in ('squid', 'kid'))})
    try:
        data = schema.validate(request.json)
    except Exception as e:
        return Response(json.dumps({'data': str(e)}), mimetype='application/json', status=400)
    return Response(json.dumps({'data': data}), mimetype='application/json', status=200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
