from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        form_data = request.form

        data = CustomData(
            gender=form_data.get('gender'),
            race_ethnicity=form_data.get('ethnicity'),
            parental_level_of_education=form_data.get('parental_level_of_education'),
            lunch=form_data.get('lunch'),
            test_preparation_course=form_data.get('test_preparation_course'),
            reading_score=float(form_data.get('reading_score')),
            writing_score=float(form_data.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        result = round(predict_pipeline.predict(pred_df)[0],2)

        return render_template(
            'home.html',
            results=result,
            form_data=form_data
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)