from flask import Flask, render_template, request, redirect, url_for
import os 
import numpy as np
import pandas as pd
import threading
import json
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) # initializing a flask app

# Variable global para almacenar el estado del entrenamiento
is_training_complete = False

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


def train_model():
    """Función que se ejecuta en segundo plano para entrenar el modelo"""
    global is_training_complete
    is_training_complete = False
    os.system("python main.py")
    is_training_complete = True


@app.route('/train',methods=['GET'])  # route to show training progress
def train():
    global is_training_complete
    # Reiniciamos el estado de entrenamiento
    is_training_complete = False
    
    # Iniciamos el entrenamiento en un thread separado
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()
    
    return render_template('train.html')


@app.route('/training-complete',methods=['GET'])  # route to show training completion
def training_complete():
    # Cargar las métricas del modelo desde el archivo JSON
    metrics_path = os.path.join('artifacts', 'model_evaluation', 'metrics.json')
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        # Valores predeterminados en caso de que el archivo no exista
        metrics = {
            "rmse": 0.0,
            "r2": 0.0,
            "mae": 0.0
        }
    
    # Formatear los valores para mostrarlos con 2 decimales
    formatted_metrics = {
        "rmse": round(metrics["rmse"], 2),
        "r2": round(metrics["r2"], 2),
        "mae": round(metrics["mae"], 2)
    }
    
    return render_template('training_complete.html', metrics=formatted_metrics)


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity =float(request.form['volatile_acidity'])
            citric_acid =float(request.form['citric_acid'])
            residual_sugar =float(request.form['residual_sugar'])
            chlorides =float(request.form['chlorides'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
         
            data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":	
	app.run(host="0.0.0.0", port = 8080)