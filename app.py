from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.save')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        soil_moisture = float(request.form['soil_moisture'])
        ndvi = float(request.form['ndvi'])

        input_data = np.array([[temp, humidity, soil_moisture, ndvi]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        predicted_yield = round(prediction[0][0], 2)

        return render_template('index.html', prediction_text=f"Predicted Crop Yield: {predicted_yield} kg/hectare")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)