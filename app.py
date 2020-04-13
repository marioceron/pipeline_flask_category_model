import json
import logging
import logging.config
import os
import threading

from flask import Flask, jsonify, request, abort

from model.model import CategoryModel

app = Flask(__name__)

logger = logging.getLogger('category')

@app.route('/status', methods=['GET'])
def status():
    logger.info("Request received")
    return jsonify({'status': "OK"})


@app.route('/train', methods=['GET'])
def train():
    try:
        logger.info('train')
        category_model = CategoryModel()
        category_run = category_model.train()
        logger.info("Response sent")
        return jsonify({"train": category_run})
    except Exception as ex:
        logger.error(f"Error: {type(ex)} {ex}")
        abort(500)
        

@app.route('/category', methods=['POST'])
def species():
    try:
        parameters = request.json['text']
        logger.info(parameters)
        category_model = CategoryModel()
        print(parameters)
        category_prediction = category_model.predict(str(parameters))
        logger.info("Response sent")
        return jsonify({"category": category_prediction})
    except Exception as ex:
        logger.error(f"Error: {type(ex)} {ex}")
        abort(500)

        
if __name__ == '__main__':
    app.run()
