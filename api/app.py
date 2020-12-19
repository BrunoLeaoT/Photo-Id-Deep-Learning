from flask import Flask, request, abort, jsonify, send_file, Response
from flask_cors import CORS, cross_origin
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import pandas as pd
import json
import sys
import os
from PIL import Image
from io import StringIO
from prediction import Prediction
application = Flask(__name__)
CORS(application)

predictor = Prediction()
@application.route("/", methods=['GET'])
def init():
    return "Bem vindo a api do Photo-Id Neural Network"

@application.route("/predict-species", methods=['POST'])
def predictSpecies():
    image = request.files.get('image1', '')
    image.save('temporary/image1.jpg', '')
    image = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(150,150)))
    return predictor.makePredictionSpecie(image)

@application.route("/compare-images", methods=['POST'])
def compareImages():
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))

    image2 = request.files.get('image2', '')
    image2.save('temporary/image2.jpg', '')
    image2 = img_to_array(load_img('temporary/image2.jpg', color_mode = "grayscale",target_size=(200,200)))

    res = predictor.compareImages(image1,image2)
    if(res):
        return "Equal"
    return "Not equal" 

@application.route("/identify", methods=['POST'])
def classifyIndividual():
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))
    (value,classe)=  predictor.makePredictionIndividual(image1)
    return jsonify({classe:str(value)})

@application.route("/", methods=['POST'])
def classifyIndividualWithSpecifics():
    return "Bem vindo a api do Photo-Id Neural Network"

if __name__ == '__main__':
    application.run(threaded=True, debug=True)

# Criar uma classe pra rede neural
# criar uma classe para o model
# ja existe a classe para o tratamaneto dos dados
# Utilizar elas para executar a funções da api
# Utilizar um "controller" que usa destas classes para fazer a previsão e a aceitação da resposta do usuário