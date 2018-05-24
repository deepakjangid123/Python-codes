import json
from flask import Flask
from flask import make_response
from flask import jsonify
from flask import request
from flask import abort
from flask import Response
from cerberus import Validator

app = Flask(__name__)


@app.route('/')  # We can define here method as well, like @app.route('/', methods = ['GET'])
def hello():
    return "Hello Babe!!!"


@app.route('/not-found')
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/request-validate', methods=['POST'])
def req_validator():
    print(request.json)
    schema = {'title': {'type': 'integer'}, 'A': {'type': 'list'}}
    v = Validator(schema)
    if not (v.validate(request.json)):
        print(v.errors)
    if not request.json or not 'title' in request.json:
        abort(500)
    task = dict(title=request.json['title'], description=request.json.get('description'), done=False)
    return Response(json.dumps({'task': task}), mimetype='application/json', status=200)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
