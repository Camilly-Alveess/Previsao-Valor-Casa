from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('models/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('models/LightGBM_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)


@app.route('/')
def home():
    result = None
    return render_template('index.html', result=result)


@app.route('/predict', methods=['POST'])
def predict():
    values = {
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': float(request.form['bathrooms']),
        'sqft_living': int(request.form['sqft_living']),
        'sqft_lot': int(request.form['sqft_lot']),
        'floors': float(request.form['floors']),
        'waterfront': int(request.form['waterfront']),
        'condition': int(request.form['condition']),
        'sqft_basement': int(request.form['sqft_basement']),
        'yr_built': int(request.form['yr_built']),
        'yr_renovated': int(request.form['yr_renovated']),
        'zipcode': int(request.form['zipcode']),
        'lat': float(request.form['lat']),
        'long': float(request.form['long']),
        'sqft_living15': int(request.form['sqft_living15']),
        'sqft_lot15': int(request.form['sqft_lot15'])
    }

    df_to_predict = pd.DataFrame(values, index=[0])

    model_choice = request.form['model']
    if model_choice == 'Random Forest':
        model = random_forest_model
    else:
        model = lgbm_model

    df_records = df_to_predict.to_records(index=False)
    df_records = df_records.tolist()

    result = model.predict(df_records)
    result = "%.2f" % float(result[0])

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
