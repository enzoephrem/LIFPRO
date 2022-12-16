from flask import Flask, render_template, request
from flask_session import Session
from tempfile import mkdtemp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import base64
import io
from utils import *
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = True
# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

# Ensure responses aren't cached
@app.after_request
def after_request(response):
	response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0, post-check=0, pre-check=0, "
	response.headers["Expires"] = 0
	response.headers["Pragma"] = "no-cache"
	return response

# Load model from Models

model = tf.keras.models.load_model("models/numeric_prediction.h5")
model.summary()

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/numbers', methods=['GET', 'POST'])
def numbers():

	print("Method: ", request.method)

	if request.method == "POST":

		# Use PIL to process the image
		header_toRemove = "data:image/png;base64,"
		base64_string = (request.form["image"])[len(header_toRemove):]

		# Convert the image from base64 string to a PIL image
		imgdata = base64.b64decode(str(base64_string))

		# Convert it to greyscalled image
		image = Image.open(io.BytesIO(imgdata)).convert("L")

		# Resize it to fit the input of the model
		image = image.resize((28, 28))

		# Get the array form of the image
		imgarr = np.array(image).astype(np.float32)

		# Plot the image
		fig = Figure()
		axis = fig.add_subplot(1, 1, 1)
		axis.imshow(imgarr, cmap= plt.cm.binary)

		output = io.BytesIO()
		FigureCanvas(fig).print_png(output)

		# Place it into a tensor reshaping it and normalizing it
		to_predict = tf.constant(imgarr, shape=(1, 28, 28)) / 255.0

		# Make the prediciton
		prediction = model.predict(to_predict)

		# Print the output 
		pred_label = prediction.argmax()

		print(pred_label)
		
		plot_image = Image.open(output)

		plot_image.save("static/plot.png")

		pred_label = '1'

		return render_template('numbers.html', pred_label = pred_label)

	else:
		return render_template('numbers.html', **locals())


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')
