from flask import Flask, render_template, request, jsonify, make_response
from fertilizer_module import fertilizer_bp  # ✅ Import blueprint

import pickle
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
from xhtml2pdf import pisa
from collections import Counter

# === Create the Flask app ===
app = Flask(__name__)
app.register_blueprint(fertilizer_bp)  # ✅ Register blueprint only once

# === Base directory to load models correctly ===
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === Load trained models with full paths ===
models = {
    "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
    "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
    "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
    "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
}

accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        data = [float(request.form[key]) for key in feature_names]
        input_df = pd.DataFrame([data], columns=feature_names)

        predictions = {}
        for name, model in models.items():
            if name in ['Logistic Regression', 'Bayes Net']:
                pred = model.predict(scaler.transform(input_df))[0]
            else:
                pred = model.predict(input_df)[0]
            predictions[name] = pred

        results = [
            {"model": name, "accuracy": accuracies[name], "prediction": predictions[name]}
            for name in models
        ]
        results.sort(key=lambda x: x["accuracy"], reverse=True)

        accuracy_labels = list(accuracies.keys())
        accuracy_values = [round(accuracies[m], 2) for m in accuracy_labels]
        prediction_counts = dict(Counter(predictions.values()))

        return render_template(
            'result.html',
            predictions=predictions,
            results=results,
            inputs=data,
            accuracies=accuracies,
            accuracy_labels=accuracy_labels,
            accuracy_values=accuracy_values,
            prediction_counts=prediction_counts
        )

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        report_data = json.loads(request.form['report_data'])
        report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

        html = render_template("report_template.html", data=report_data)
        pdf = BytesIO()
        pisa_status = pisa.CreatePDF(html, dest=pdf)

        if pisa_status.err:
            return "PDF generation error", 500

        response = make_response(pdf.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
        return response

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify, make_response
# from flask import Flask, render_template, request, jsonify, make_response
# from fertilizer_module import fertilizer_bp  # ✅ Import blueprint at the top

# # === Create the Flask app ===
# app = Flask(__name__)

# # ✅ Register the blueprint AFTER app is created
# app.register_blueprint(fertilizer_bp)

# from flask import Flask, render_template, request, jsonify, make_response
# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa
# from collections import Counter

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [
#             {"model": name, "accuracy": accuracies[name], "prediction": predictions[name]}
#             for name in models
#         ]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         # Prepare chart data
#         accuracy_labels = list(accuracies.keys())
#         accuracy_values = [round(accuracies[m], 2) for m in accuracy_labels]
#         prediction_counts = dict(Counter(predictions.values()))

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies,
#             accuracy_labels=accuracy_labels,
#             accuracy_values=accuracy_values,
#             prediction_counts=prediction_counts
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
#from flask import Flask, render_template, request, jsonify, make_response


# from flask import Flask, render_template, request, jsonify, make_response
# from fertilizer_module import fertilizer_bp  # Import the fertilizer blueprint

# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa

# app = Flask(__name__)
# app.register_blueprint(fertilizer_bp)  # Register the blueprint

# # === Base directories ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# # === Load crop recommendations (JSON) ===
# RECOMMENDATION_FILE = os.path.join(BASE_DIR, "recommendations.json")
# if os.path.exists(RECOMMENDATION_FILE):
#     with open(RECOMMENDATION_FILE) as f:
#         crop_recommendations = json.load(f)
# else:
#     crop_recommendations = {}

# # === Routes ===
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [
#             {
#                 "model": name,
#                 "accuracy": accuracies[name],
#                 "prediction": predictions[name]
#             } for name in models
#         ]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         top_prediction = results[0]['prediction']
#         recommendation = crop_recommendations.get(top_prediction.lower(), None)

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies,
#             recommendation=recommendation,
#             top_prediction=top_prediction
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         prediction = report_data['results'][0]['prediction'].lower()
#         report_data['recommendation'] = crop_recommendations.get(prediction, None)

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify, make_response
# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa

# app = Flask(__name__)

# # === Base directories ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# # === Load fertilizer recommendation data from JSON ===
# FERTILIZER_FILE = os.path.join(BASE_DIR, "fertilizer_recommendations.json")
# with open(FERTILIZER_FILE) as f:
#     fertilizer_recommendations = json.load(f)

# # === Load crop dataset for validation/info ===
# crop_data = pd.read_csv(os.path.join(BASE_DIR, "Crop_recommendation.csv"))

# # === Routes ===
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [
#             {
#                 "model": name,
#                 "accuracy": accuracies[name],
#                 "prediction": predictions[name]
#             } for name in models
#         ]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         top_prediction = results[0]['prediction']

#         # For consistency, you can fetch extra info from crop_data if needed
#         recommendation = {
#             "season": "N/A",
#             "fertilizer": "No specific recommendation found.",
#             "irrigation": "N/A"
#         }

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies,
#             recommendation=recommendation,
#             top_prediction=top_prediction
#         )
#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         prediction = report_data['results'][0]['prediction'].lower()
#         report_data['recommendation'] = {
#             "season": "N/A",
#             "fertilizer": "No specific recommendation found.",
#             "irrigation": "N/A"
#         }

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/fertilizer', methods=['GET', 'POST'])
# def fertilizer():
#     if request.method == 'GET':
#         return render_template('fertilizer_input.html')
#     else:
#         crop = request.form['crop'].lower()
#         N = float(request.form['N'])
#         P = float(request.form['P'])
#         K = float(request.form['K'])

#         recommendation = fertilizer_recommendations.get(crop, {
#             "season": "N/A",
#             "fertilizer": "No specific recommendation found.",
#             "irrigation": "N/A"
#         })

#         return render_template(
#             'fertilizer_result.html',
#             crop=crop.title(),
#             N=N,
#             P=P,
#             K=K,
#             recommendation=recommendation
#         )

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request, jsonify, make_response//
# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# # Load crop recommendations
# RECOMMENDATION_FILE = os.path.join(BASE_DIR, "recommendations.json")
# if os.path.exists(RECOMMENDATION_FILE):
#     with open(RECOMMENDATION_FILE) as f:
#         crop_recommendations = json.load(f)
# else:
#     crop_recommendations = {}

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [
#             {
#                 "model": name,
#                 "accuracy": accuracies[name],
#                 "prediction": predictions[name]
#             } for name in models
#         ]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         top_prediction = results[0]['prediction']
#         recommendation = crop_recommendations.get(top_prediction.lower(), None)

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies,
#             recommendation=recommendation,
#             top_prediction=top_prediction
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         # Add recommendation info if available
#         prediction = report_data['results'][0]['prediction'].lower()
#         report_data['recommendation'] = crop_recommendations.get(prediction, None)

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, jsonify, make_response
# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa
# from collections import Counter

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [
#             {"model": name, "accuracy": accuracies[name], "prediction": predictions[name]}
#             for name in models
#         ]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         # Prepare chart data
#         accuracy_labels = list(accuracies.keys())
#         accuracy_values = [round(accuracies[m], 2) for m in accuracy_labels]
#         prediction_counts = dict(Counter(predictions.values()))

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies,
#             accuracy_labels=accuracy_labels,
#             accuracy_values=accuracy_values,
#             prediction_counts=prediction_counts
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify, make_response
# import pickle
# import numpy as np
# import os
# import json
# import pandas as pd
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#         data = [float(request.form[key]) for key in feature_names]
#         input_df = pd.DataFrame([data], columns=feature_names)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_df))[0]
#             else:
#                 pred = model.predict(input_df)[0]
#             predictions[name] = pred

#         results = [{
#             "model": name,
#             "accuracy": accuracies[name],
#             "prediction": predictions[name]
#         } for name in models]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         return render_template(
#             'result.html',
#             predictions=predictions,
#             results=results,
#             inputs=data,
#             accuracies=accuracies
#         )

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)  #pdf wala

# # """
# app.py
# from flask import Flask, render_template, request, jsonify, make_response
# import pickle
# import numpy as np
# import os
# import json
# from datetime import datetime
# from io import BytesIO
# from xhtml2pdf import pisa

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
#         input_data = np.array(data).reshape(1, -1)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_data))[0]
#             else:
#                 pred = model.predict(input_data)[0]
#             predictions[name] = pred

#         results = [{"model": name, "accuracy": accuracies[name], "prediction": predictions[name]} for name in models]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         return render_template('result.html', predictions=predictions, results=results, inputs=data)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# @app.route('/download-report', methods=['POST'])
# def download_report():
#     try:
#         report_data = json.loads(request.form['report_data'])
#         report_data['generated_on'] = datetime.now().strftime('%d-%m-%Y %I:%M %p')

#         html = render_template("report_template.html", data=report_data)
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, dest=pdf)

#         if pisa_status.err:
#             return "PDF generation error", 500

#         response = make_response(pdf.getvalue())
#         response.headers['Content-Type'] = 'application/pdf'
#         response.headers['Content-Disposition'] = 'attachment; filename=prediction_report.pdf'
#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import os

# app = Flask(__name__)

# # === Base directory to load models correctly ===
# BASE_DIR = os.path.dirname(__file__)
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# # === Load trained models with full paths ===
# models = {
#     "Logistic Regression": pickle.load(open(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"), "rb")),
#     "Random Forest": pickle.load(open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb")),
#     "Gradient Boosting": pickle.load(open(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"), "rb")),
#     "Bayes Net": pickle.load(open(os.path.join(MODEL_DIR, "bayes_net_model.pkl"), "rb"))
# }

# # Load accuracies and scaler
# accuracies = pickle.load(open(os.path.join(MODEL_DIR, "model_accuracies.pkl"), "rb"))
# scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract and preprocess form data
#         data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
#         input_data = np.array(data).reshape(1, -1)

#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_data))[0]
#             else:
#                 pred = model.predict(input_data)[0]
#             predictions[name] = pred

#         results = [{"model": name, "accuracy": accuracies[name], "prediction": predictions[name]} 
#                    for name in models]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         return render_template('result.html', predictions=predictions, results=results)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)//runnable 22-06-2025

# from flask import Flask, render_template, request, jsonify //old new
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load trained models
# models = {
#     "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
#     "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
#     "Gradient Boosting": pickle.load(open("gradient_boosting_model.pkl", "rb")),
#     "Bayes Net": pickle.load(open("bayes_net_model.pkl", "rb"))
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from form
#         data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

#         # Predict using all models
#         predictions = {}
#         for name, model in models.items():
#             predictions[name] = model.predict([data])[0]

#         # Render results for all models
#         return render_template('result.html', predictions=predictions)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# """
# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load trained models and their accuracies
# models = {
#     "Logistic Regression": pickle.load(open("logistic_regression_model.pkl", "rb")),
#     "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
#     "Gradient Boosting": pickle.load(open("gradient_boosting_model.pkl", "rb")),
#     "Bayes Net": pickle.load(open("bayes_net_model.pkl", "rb"))
# }
# accuracies = pickle.load(open("model_accuracies.pkl", "rb"))  # Load model accuracies

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from form
#         data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

#         # Predict using all models
#         predictions = {name: model.predict([data])[0] for name, model in models.items()}

#         # Combine predictions with accuracies and sort by accuracy
#         results = [{"model": name, "accuracy": accuracies[name], "prediction": predictions[name]} 
#                    for name in models]
#         results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

#         # Pass predictions and results to result.html
#         return render_template('result.html', predictions=predictions, results=results)

#     except Exception as e:
#         return jsonify({"error": str(e)})


# if __name__ == '__main__':
#     app.run(debug=True)
#=============================this is new in 2025 old new
# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import os

# app = Flask(__name__)

# # Load trained models
# models = {
#     "Logistic Regression": pickle.load(open("models/logistic_regression_model.pkl", "rb")),
#     "Random Forest": pickle.load(open("models/random_forest_model.pkl", "rb")),
#     "Gradient Boosting": pickle.load(open("models/gradient_boosting_model.pkl", "rb")),
#     "Bayes Net": pickle.load(open("models/bayes_net_model.pkl", "rb"))
# }

# # Load accuracies
# accuracies = pickle.load(open("models/model_accuracies.pkl", "rb"))

# # Load scaler for Logistic Regression and Bayes Net
# scaler = pickle.load(open("models/scaler.pkl", "rb"))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/input')
# def input_page():
#     return render_template('input.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from form and convert to float list
#         data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

#         # Reshape input for model
#         input_data = np.array(data).reshape(1, -1)

#         # Predict using each model
#         predictions = {}
#         for name, model in models.items():
#             if name in ['Logistic Regression', 'Bayes Net']:
#                 pred = model.predict(scaler.transform(input_data))[0]
#             else:
#                 pred = model.predict(input_data)[0]
#             predictions[name] = pred

#         # Combine predictions and accuracies
#         results = [{"model": name, "accuracy": accuracies[name], "prediction": predictions[name]} 
#                    for name in models]
#         results.sort(key=lambda x: x["accuracy"], reverse=True)

#         return render_template('result.html', predictions=predictions, results=results)

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
