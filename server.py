import subprocess
from flask import json
import io, json

from keras.models import load_model

import mlp
import h5json
import numpy as np
import os
import subprocess
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/train', methods=['GET','POST'])
def train():
    training_data = request.form['training_data']
    epoche = int(request.form['epoche'])
    target_data = request.form['target_data']
    training_data = np.array(eval(training_data), "float32")
    target_data = np.array(eval(target_data), "float32")
    model = mlp.train_model(training_data, target_data, epoche)
    model.save('my_model.h5')
    proc = subprocess.Popen(['python', 'encoder.py', 'my_model.h5'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return proc.communicate()[0]

@app.route('/test', methods=['POST'])
def test():

    model = request.form['model']
    testing_data = request.form['testing_data']
    x = eval(testing_data)
    testing_data = np.array(x, "float32")

    with io.open('model.json', 'w', encoding='utf-8') as f:
         f.write(model)

    if os.path.exists('model.h5'):
        os.remove('model.h5')

    p = subprocess.Popen(['python', 'decoder.py', 'model.json', 'model.h5'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    p_status = p.wait()

    model = load_model('model.h5')
    predictions = model.predict(testing_data).round()

    return str(predictions)


app.run()
