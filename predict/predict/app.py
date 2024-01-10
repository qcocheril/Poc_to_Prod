from flask import Flask, request, render_template, jsonify
from run import TextPredictionModel

app = Flask(__name__)


@app.route('/')
def home():
    msg = "Hello World"
    return msg


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text_list = []
        text = request.form.get('textbox')
        text_list.append(text)
        print(text)
        model = TextPredictionModel.from_artefacts("../../train/data/artefacts/2024-01-10-18-58-15")

        prediction = model.predict(text_list, 5)

        return jsonify(prediction)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
