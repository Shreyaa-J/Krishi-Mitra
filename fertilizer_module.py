# from flask import Blueprint, render_template, request
# import os
# import json

# fertilizer_bp = Blueprint('fertilizer', __name__)

# # === Load fertilizer recommendation data ===
# BASE_DIR = os.path.dirname(__file__)
# FERTILIZER_FILE = os.path.join(BASE_DIR, "fertilizer_recommendations.json")

# if os.path.exists(FERTILIZER_FILE):
#     with open(FERTILIZER_FILE) as f:
#         fertilizer_recommendations = json.load(f)
# else:
#     fertilizer_recommendations = {}

# @fertilizer_bp.route('/fertilizer', methods=['GET', 'POST'])
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
from flask import Blueprint, render_template, request
import os
import json

fertilizer_bp = Blueprint('fertilizer', __name__)

# === Load fertilizer recommendation data ===
BASE_DIR = os.path.dirname(__file__)
FERTILIZER_FILE = os.path.join(BASE_DIR, "fertilizer_recommendations.json")

# Load the recommendations if file exists
if os.path.exists(FERTILIZER_FILE):
    with open(FERTILIZER_FILE) as f:
        fertilizer_recommendations = json.load(f)
else:
    fertilizer_recommendations = {}

@fertilizer_bp.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'GET':
        return render_template('fertilizer_input.html')
    else:
        try:
            crop = request.form['crop'].lower()
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])

            # Default recommendation if crop not found
            recommendation = fertilizer_recommendations.get(crop, {
                "season": "N/A",
                "fertilizer": "No specific recommendation found.",
                "irrigation": "N/A"
            })

            return render_template(
                'fertilizer_result.html',
                crop=crop.title(),
                N=N,
                P=P,
                K=K,
                recommendation=recommendation
            )
        except Exception as e:
            return f"Error: {e}", 400


