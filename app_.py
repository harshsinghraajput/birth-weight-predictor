from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def prediction():
        baby_data = request.get_json()

        baby_df = pd.DataFrame(baby_data)   

        with open('model/model.pkl', 'rb') as obj:
            model = pickle.load(obj)

        prediction = model.predict(baby_df)

        return jsonify({"Prediction": round(float(prediction) , 2)})



if __name__ == '__main__':
    app.run(debug=True)