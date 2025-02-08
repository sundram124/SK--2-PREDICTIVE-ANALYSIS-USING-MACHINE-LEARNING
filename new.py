# # from flask import Flask, render_template, request, jsonify
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import RandomForestClassifier
# # import os

# # app = Flask(__name__)

# # # Load the dataset and train the model
# # file_path = "C:/Users/arjun/Desktop/laptop_purchase_data_india.csv"  # Update path if needed
# # if os.path.exists(file_path):
# #     df = pd.read_csv(file_path)
# #     df = df.drop(columns=['Customer_ID'])
# #     df_encoded = pd.get_dummies(df, drop_first=True)
# #     X = df_encoded.drop(columns=['Satisfaction_Rating'])
# #     y = df_encoded['Satisfaction_Rating']
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #     scaler = StandardScaler()
# #     X_train = scaler.fit_transform(X_train)
# #     X_test = scaler.transform(X_test)
# #     model = RandomForestClassifier(n_estimators=100, random_state=42)
# #     model.fit(X_train, y_train)
# # else:
# #     print(f"File not found at {file_path}")

# # # @app.route('/')
# # # def index():
# # #     return render_template('name.html')
# # def hello_world():
# #     return 'Hello, World!'

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.get_json()  # Use get_json() safely
# #         if not data:
# #             return jsonify({'error': 'No input data provided'}), 400

# #         age = data.get('age')
# #         gender = data.get('gender')

# #         if age is None or gender is None:
# #             return jsonify({'error': 'Missing required fields'}), 400

# #         # Encode gender
# #         gender_encoded = 1 if gender.lower() == 'male' else 0

# #         # Standardize input data
# #         input_data = [[age, gender_encoded]]
# #         input_data_scaled = scaler.transform(input_data)

# #         # Predict
# #         prediction = model.predict(input_data_scaled)

# #         return jsonify({'prediction': int(prediction[0])})

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     data = request.json  # Get input data as JSON
# # #     age = data.get('age')
# # #     gender = data.get('gender')
    
# # #     # Feature processing based on the model
# # #     # You will need to encode 'gender' properly using the same method as your model was trained
# # #     gender_encoded = 1 if gender == 'Male' else 0  # Just an example, update as needed
    
# # #     # Assuming the model takes 'age' and 'gender' as inputs
# # #     input_data = [[age, gender_encoded]]  # Example: Add other features as required

# # #     # Standardize input data (assuming your model uses scaling)
# # #     input_data_scaled = scaler.transform(input_data)
    
# # #     # Predict
# # #     prediction = model.predict(input_data_scaled)
    
# # #     # Return prediction
# # #     return jsonify({'prediction': prediction.tolist()})

# # if __name__ == '__main__':
# #     app.run(debug=True)



# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# import os
# import pickle

# app = Flask(__name__)

# # Load the trained model
# with open("your_model.pkl", "rb") as file:
#     model = pickle.load(file)

# # Print the expected number of features
# print(model.n_features_in_)

# # Load and train model
# file_path = "C:/Users/arjun/Desktop/laptop_purchase_data_india.csv"
# if os.path.exists(file_path):
#     df = pd.read_csv(file_path)
#     df = df.drop(columns=['Customer_ID'])
#     df_encoded = pd.get_dummies(df, drop_first=True)
#     X = df_encoded.drop(columns=['Satisfaction_Rating'])
#     y = df_encoded['Satisfaction_Rating']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
# else:
#     print(f"File not found at {file_path}")
#     model = None  # Prevent errors if the model isn't loaded

# @app.route('/')
# def index():
#     return 'Welcome to the Prediction API'

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()  # Retrieve the JSON data from the request
#         if not data:
#             return jsonify({'error': 'No input data provided'}), 400

#         age = data.get('age')
#         gender = data.get('gender')
#         brand = data.get('brand')
#         price = data.get('price')

#         if age is None or gender is None or brand is None or price is None:
#             return jsonify({'error': 'Missing required fields'}), 400

#         # Encode gender (you might need to adapt this depending on your model's expectations)
#         gender_encoded = 1 if gender.lower() == 'male' else 0
#         # Add any additional feature processing here, such as encoding 'brand' or normalizing 'price'

#         input_data = [[age, gender_encoded, price]]  # Example: Extend with more features

#         # Standardize input data (if your model requires it)
#         input_data_scaled = scaler.transform(input_data)

#         # Make prediction using the trained model
#         prediction = model.predict(input_data_scaled)

#         return jsonify({'prediction': int(prediction[0])})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)



import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open("your_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load StandardScaler (if applicable)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Print expected number of features
print(f"Model expects {model.n_features_in_} features.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Encode categorical variables
        gender_mapping = {"Male": 0, "Female": 1}
        brand_mapping = {"Dell": 0, "HP": 1, "Apple": 2, "Lenovo": 3}  # Add more brands as needed

        # Extract features and encode categories
        age = data.get("age", 0)
        gender = gender_mapping.get(data.get("gender", "Male"), 0)  # Default to "Male" (0)
        brand = brand_mapping.get(data.get("brand", "Dell"), 0)  # Default to "Dell" (0)
        price = data.get("price", 0)

        # Combine into input array
        input_data = np.array([age, gender, brand, price]).reshape(1, -1)

        # Standardize input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
