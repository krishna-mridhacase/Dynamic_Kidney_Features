import os
from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
import shap
import lime
from lime import lime_tabular
from sklearn.impute import KNNImputer
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt



app = Flask(__name__)

# Load the model
model = pickle.load(open('models/kidney.pkl', 'rb'))




# Load each model from the 'models/' directory
ada_boost_model = pickle.load(open('models/ada_boost_model.pkl', 'rb'))
cat_boost_model = pickle.load(open('models/cat_boost_model.pkl', 'rb'))
knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))
random_forest_model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
xgb_boost_model = pickle.load(open('models/xgb_boost_model.pkl', 'rb'))
extra_tree_model = pickle.load(open('models/extra_tree_model.pkl', 'rb'))



# Load the training data for SHAP KernelExplainer (example: replace with actual training data)
X_train = pd.read_csv('notebooks/X_train.csv')  # Replace with your actual training data path
X_train = X_train.drop(columns=['Unnamed: 0'])


X_test = pd.read_csv('notebooks/X_test.csv')  # Replace with your actual training data path
X_test = X_test.drop(columns=['Unnamed: 0'])

knn_imputer = KNNImputer(n_neighbors=3)
knn_imputer.fit(X_train)  # Fit the imputer on the training data


# Define important features that are critical for prediction
important_features = ['specific_gravity', 'haemoglobin', 'serum_creatinine', 'albumin', 'packed_cell_volume']

# Initialize SHAP explainers
tree_explainer = shap.TreeExplainer(model)
kernel_explainer = shap.KernelExplainer(model.predict_proba, X_train)

# Feature names (in the same order as the DataFrame columns used for training)
feature_names = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
                 'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume',
                 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia']

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')


class_names=[ 'CKD', 'Healthy']


def predict(values):
    try:
        if len(values) == 24:
            # Convert the input values to a DataFrame with the appropriate feature names
            input_df = pd.DataFrame([values], columns=feature_names)

            pred = model.predict(input_df)[0]  

            shap_values = tree_explainer.shap_values(input_df)
            # print(f"SHAP values: {shap_values}")  # Log the SHAP values for debugging
            return pred, shap_values, input_df
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def warning_to_patient(values):
    try:
        if len(values) == 24:
            # Convert the input values to a DataFrame with the appropriate feature names
            input_df = pd.DataFrame([values], columns=feature_names)

            pred = model.predict(input_df)[0]  
            # Get the probability scores for each class
            predicted_probabilities = model.predict_proba(input_df)

            # Extract the probabilities for the positive class (assuming 1 is the Negative class)
            ckd_class_probability = predicted_probabilities[0][0]  # Probability of class 0 (CKD)
            non_ckd_class_probability = predicted_probabilities[0][1]  # Probability of class 1 (Non-CKD)

            return pred, ckd_class_probability, non_ckd_class_probability
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

# Prepare a tree-like string representation
def display_tree_results(predictions, final_pred):
    tree_structure = """
    Prediction Tree:
    ├── Ada Boost: {}
    ├── Cat Boost: {}
    ├── K-Nearest Neighbors: {}
    ├── Random Forest: {}
    ├── XGBoost: {}
    ├── Extra Trees: {}
    └── Final Prediction (Majority Vote): {}
    """.format(
        'CKD' if predictions['Ada Boost'] == 0 else 'Healthy',
        'CKD' if predictions['Cat Boost'] == 0 else 'Healthy',
        'CKD' if predictions['K-Nearest Neighbors'] == 0 else 'Healthy',
        'CKD' if predictions['Random Forest'] == 0 else 'Healthy',
        'CKD' if predictions['XGBoost'] == 0 else 'Healthy',
        'CKD' if predictions['Extra Trees'] == 0 else 'Healthy',
        'CKD' if final_pred == 0 else 'Healthy'
    )
    return tree_structure




def plot_shap_values(shap_impact):
    # Create a bar plot of the top 5 SHAP values
    features = list(shap_impact.keys())
    values = list(shap_impact.values())
    
    plt.figure(figsize=(8, 4))  # Adjust the figure size to fit the page better
    plt.barh(features, values, color='skyblue')
    plt.xlabel('SHAP Value')
    plt.title('Top 5 Feature SHAP Values')
    plt.gca().invert_yaxis()  # To display the highest value on top
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.savefig('static/shap_plot.png')
    plt.close()



def plot_force_plot(user_input_df):
    try:
        # Use TreeExplainer for consistency
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input_df)

        # Ensure shap_values is properly indexed
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Binary classification case: shap_values is a list of two arrays
            class_index = int(model.predict(user_input_df)[0])  # Get predicted class index
            shap_values_class = shap_values[class_index]  # Use SHAP values for the predicted class
            expected_value = explainer.expected_value[class_index]  # Expected value for the predicted class
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # shap_values is a 3-dimensional array (n_samples, n_features, n_classes)
            class_index = int(model.predict(user_input_df)[0])  # Get predicted class index
            shap_values_class = shap_values[0, :, class_index]  # Use SHAP values for the predicted class
            expected_value = explainer.expected_value[class_index]  # Expected value for the predicted class
        else:
            raise ValueError("Unexpected SHAP values structure.")

        # Generate and display the force plot
        shap.initjs()
        plt.figure(figsize=(10, 5))
        
        # Use matplotlib=True to render the force plot with matplotlib
        shap.force_plot(
            expected_value,  # Expected value for the predicted class
            shap_values_class,  # SHAP values for the predicted class
            user_input_df.iloc[0, :],  # Single sample
            feature_names=feature_names,  # Pass feature names for better visualization
            matplotlib=True,  # Enable matplotlib
            show=False  # Do not display the plot immediately
        )

        force_plot_path = os.path.join('static', 'force_plot.png')
        plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Force plot saved at {force_plot_path}")
    except Exception as e:
        print(f"Error displaying force plot: {e}")



def plot_lime_explanation(input_df, class_names, file_path='static/lime_explanation.html'):
    try:
        print(f"Input DataFrame shape: {input_df.shape}")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            class_names=class_names,
            mode='classification'
        )
        exp = explainer.explain_instance(
            data_row=input_df.iloc[0],
            predict_fn=model.predict_proba
        )
        with open(file_path, 'w', encoding='utf-8') as f:  # Use UTF-8 encoding
            f.write(exp.as_html())
        print(f"Explanation saved as {file_path}")
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")




        

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    high_values = {}
    shap_impact = {}
    pred = None
    user_input_dict = {}
    try:
        if request.method == 'POST':
            # Capture user input as a dictionary
            user_input_dict = request.form.to_dict()


            # Convert the input to the appropriate types (e.g., int, float)
            
            for key, value in user_input_dict.items():
                if value == '':
                    user_input_dict[key] = np.nan  # Handle empty inputs as NaN
                else:
                    try:
                        user_input_dict[key] = int(value)  # Attempt to convert to int
                    except ValueError:
                       user_input_dict[key] = float(value)  # Convert to float if int conversion fails

            # Filter out features that are NaN (or null)
            non_null_features = [key for key, value in user_input_dict.items() if not pd.isnull(value)]


            # Convert the dictionary to a list in the correct order of features
            to_predict_list = [user_input_dict.get(feature, np.nan) for feature in feature_names]

            # Convert to a DataFrame for prediction
            input_df = pd.DataFrame([to_predict_list], columns=feature_names)
        
            print("User Inputs: ", input_df)   
            print("User Input Shape : ", input_df.shape)

            # Impute missing values using the KNN imputer
            imputed_input_df = knn_imputer.transform(input_df)

            print("We have all features: ", imputed_input_df)
            print("All Input Shape : ", imputed_input_df.shape)

            # Make a prediction using the model
            pred, shap_values, input_df = predict(imputed_input_df.flatten())

            print("Prediction: ", pred)
            print("Shap Values: ", shap_values)
            print("All Inputs: ", input_df)
            print("Shap Shape: ", shap_values.shape)



            if shap_values is not None and shap_values.size > 0:
                if pred == 1:
                    # Analyze SHAP values for Class 1
                    shap_impact = {feature_names[i]: shap_values[0][i][1] for i in range(len(feature_names))}
                else:
                    # Analyze SHAP values for Class 0
                    shap_impact = {feature_names[i]: shap_values[0][i][0] for i in range(len(feature_names))}

                # Get top 5 features by absolute SHAP value
                top_features = sorted(shap_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                shap_impact = dict(top_features)
                print("SHAP impact:", shap_impact)

                # Plot the top 5 SHAP values
                plot_shap_values(shap_impact)

                # Plot the force plot
                plot_force_plot(input_df)

                # Plot LIME explanation
                plot_lime_explanation(input_df, class_names)
                print("Lime is working here")

                # Make predictions with each model
                ada_pred = ada_boost_model.predict(input_df)[0]
                cat_pred = cat_boost_model.predict(input_df)[0]
                knn_pred = knn_model.predict(input_df)[0]
                rf_pred = random_forest_model.predict(input_df)[0]
                xgb_pred = xgb_boost_model.predict(input_df)[0]
                extra_pred = extra_tree_model.predict(input_df)[0]

                # Collect predictions
                model_predictions = {
                    'Ada Boost': ada_pred,
                    'Cat Boost': cat_pred,
                    'K-Nearest Neighbors': knn_pred,
                    'Random Forest': rf_pred,
                    'XGBoost': xgb_pred,
                    'Extra Trees': extra_pred
                }

                # Calculate final prediction based on majority vote
                
                ckd_votes = sum(pred == 0 for pred in model_predictions.values())
                healthy_votes = sum(pred == 1 for pred in model_predictions.values())
                final_prediction = 1 if healthy_votes > ckd_votes else 0  # 1 = healthy(non-ckd), 0 = ckd

                # Display the tree-like result
                tree_result = display_tree_results(model_predictions, final_prediction)

                # Working with Warning
                pred, ckd_class_probability, non_ckd_class_probability = warning_to_patient(imputed_input_df.flatten())

                # Convert predicted class back to label (assuming 1 is 'Healthy' and 0 is 'CKD')
                predicted_label = 'Healthy' if pred == 1 else 'CKD'

                # Check if important features are missing in the new patient input
                missing_features = [feature for feature in important_features if feature not in non_null_features]

                # if pred is not None:
                #     # Convert predicted class back to label (assuming 0 is 'ckd' and 1 is 'not ckd')
                #     # Display the result
                #     print(f'The predicted class for the new patient is: {predicted_label}')
                #     print(f'Probability of being Non-CKD (positive class): {positive_class_probability:.2f}')
                #     print(f'Probability of being CKD (negative class): {negative_class_probability:.2f}')

                #     # Check for missing important features and give a cautionary alert
                #     if missing_features:
                #         print("⚠️ Warning: The following important features are missing:")
                #         for feature in missing_features:
                #             print(f"- {feature}")
                #         print("Even though we provide you with prediction results, please note that having these tests is crucial for accurate diagnosis.")

    except Exception as e:
        print(f"Error during prediction: {e}")
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    # Pass the user_input_dict to the template along with other data
    return render_template('predict.html', pred=pred, predicted_label=predicted_label, 
        ckd_class_probability=ckd_class_probability,
        non_ckd_class_probability=non_ckd_class_probability,
        missing_features=missing_features,shap_impact=shap_impact, user_input=user_input_dict,  
        tree_result=tree_result, model_predictions=model_predictions, final_prediction=final_prediction)



if __name__ == '__main__':
    app.run(debug=True)
