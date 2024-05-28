from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carregar o modelo
with open('random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Supondo que você tenha outro modelo LGBM salvo de maneira similar
with open('LightGBM_model.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)


@app.route('/')
def home():
    result = None
    return render_template('index.html', result=result)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados do formulário
        values = request.form.getlist('new_house')
        df_to_predict = pd.DataFrame({'num_bed': [values[0]],
                                      'num_bath': [values[1]],
                                      'size_house': [values[2]],
                                      'size_lot': [values[3]],
                                      'num_floors': [values[4]],
                                      'is_waterfront': [values[5]],
                                      'condition': [values[6]],
                                      'size_basement': [values[7]],
                                      'year_built': [values[8]],
                                      'renovation_date': [values[9]],
                                      'zip': [values[10]],
                                      'latitude': [values[11]],
                                      'longitude': [values[12]],
                                      'avg_size_neighbor_houses': [values[13]],
                                      'avg_size_neighbor_lot': [values[14]]
                                      })

      # Selecionar o modelo
        model = request.form['model']
        if model == 'Random Forest':
            model = random_forest_model
        else:
            model = lgbm_model

        # Fazer a predição
        print('\ndf: ', df_to_predict)
        result = model.predict(df_to_predict)
        result = "%.2f" % result

        return render_template('index.html', result=result)

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
