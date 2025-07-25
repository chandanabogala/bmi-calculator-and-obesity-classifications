print("✅ Flask app starting...")  # NEW line

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("bmi_model.pkl")

def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def get_category_label(category_num):
    categories = ["Underweight", "Normal", "Overweight", "Obese"]
    explanation = [
        "BMI less than 18.5",
        "BMI between 18.5 and 24.9",
        "BMI between 25 and 29.9",
        "BMI 30 or higher"
    ]
    return categories[category_num], explanation[category_num]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    bmi = calculate_bmi(height, weight)
    prediction = model.predict([[bmi]])[0]
    category, explanation = get_category_label(prediction)
    return render_template('result.html', bmi=bmi, category=category, explanation=explanation)

if __name__ == '__main__':
    print("✅ Launching Flask server...")
    app.run(debug=True)
