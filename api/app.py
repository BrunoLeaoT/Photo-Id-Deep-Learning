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
from network_humpback import NetworkHumpback 

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
    try:
        return predictor.makePredictionSpecie(image)
    except Exception as e:
        return Response(status=500, response="Something went wrong")
    

@application.route("/compare-images", methods=['POST'])
def compareImages():
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))

    image2 = request.files.get('image2', '')
    image2.save('temporary/image2.jpg', '')
    image2 = img_to_array(load_img('temporary/image2.jpg', color_mode = "grayscale",target_size=(200,200)))

    try:
        (res,prob) = predictor.compareImages(image1,image2)
        print(prob)
        if(res):
            return Response(status=200,response="Equal")
        return Response(status=200,response="Not equal")
    except Exception as e:
        return Response(status=500, response="Something went wrong") 

@application.route("/identify", methods=['POST'])
def classifyIndividual():
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))
    try:
        (value,classe)=  predictor.makePredictionIndividual(image1)
        return jsonify({classe:str(value)})
    except Exception as e:
        print(e)
        return Response(status=500, response="Something went wrong")

@application.route("/indetify-between-individuals", methods=['POST'])
def classifyIndividualWithSpecifics():
    individuals =  request.form["individuals"].split(",")
    print(individuals)
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))
    
    try:
        (value,classe)=  predictor.makePredictionIndividual(image1,individuals)
        return jsonify({classe:str(value)})
    except Exception as e:
        return Response(status=500)
    
@application.route("/update-base", methods=['POST'])
def update():
    classe =  request.form["classe"]
    image1 = request.files.get('image1', '')
    image1.save('temporary/image1.jpg', '')
    image1 = img_to_array(load_img('temporary/image1.jpg', color_mode = "grayscale",target_size=(200,200)))
    try:
        predictor.update(image1,classe)
        return Response(status=200,response="ok")
    except Exception as e:
        print(e)
        return Response(status=500,response="Something went wrong")


if __name__ == '__main__':
    application.run(threaded=True, debug=True)
