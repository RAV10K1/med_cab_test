from flask import Flask, jsonify, request, make_response

API = Flask(__name__)


@API.route('/')
def index():
    return jsonify("Welcome to Medicine Cabinet! Cannabis Strain Recommendation API")


if __name__ == '__main__':
    API.run(debug=True)