import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data  
def load_data(file_path):
    try:
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure dataset is in the same directory as main file.")
        st.stop()

# Preprocess data
def preprocess_data(dataset):
    # Define features and target
    X = dataset[['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'venue', 'bat_team', 'bowl_team']]
    y = dataset['total']
    
    # One-Hot Encoding for categorical features
    categorical_features = ['venue', 'bat_team', 'bowl_team']
    numeric_features = ['overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5']
    
    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.1, random_state=555)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Function to calculate custom accuracy
def custom_accuracy(y_test, y_pred, threshold):
    right = np.sum(np.abs(y_pred - y_test) <= threshold)
    return (right / len(y_pred)) * 100

# Define the Streamlit app
def main():
    st.title("INDIAN PREMIER LEAGUE SCORE PREDICTION")
    st.write("This is a simple web app to predict the final score of an IPL match based on the current match conditions.")
    
    # Load and preprocess data
    dataset = load_data('data/ipl.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(dataset)
    
    # Define and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Button to show the dataset
    
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    custom_acc = custom_accuracy(y_test, y_pred, 20)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Update metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Squared Error', 'Model Accuracy', 'Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Predicted Score'],
        'Value': [f"{r2:.2f}", f"{mse:.2f}", f"{custom_acc:.2f}%", f"{mae:.2f}", f"{rmse:.2f}", "N/A"]
    })
        
    # User inputs
    st.sidebar.header('User Input')
    overs = st.sidebar.number_input('Overs Played', min_value=0, max_value=20, value=10)
    runs = st.sidebar.number_input('Current Runs', min_value=0, max_value=500, value=100)
    wickets = st.sidebar.number_input('Wickets Lost', min_value=0, max_value=10, value=2)
    runs_last_5 = st.sidebar.number_input('Runs in Last 5 Overs', min_value=0, max_value=100, value=50)
    wickets_last_5 = st.sidebar.number_input('Wickets in Last 5 Overs', min_value=0, max_value=10, value=1)
    
    # Input for venue, batting team, and bowling team
    venue = st.sidebar.selectbox('Venue', options=dataset['venue'].unique())
    bat_team = st.sidebar.selectbox('Batting Team', options=dataset['bat_team'].unique())
    bowl_team = st.sidebar.selectbox('Bowling Team', options=dataset['bowl_team'].unique())
    
    # Prepare user input data
    user_input = pd.DataFrame({
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5],
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team]
    })
    
    # Display user input data
    st.write("### User Input")
    st.table(user_input)

    # Transform user input data
    user_input_transformed = preprocessor.transform(user_input)
    
    if st.button('Predict Score'):
        try:
            prediction = model.predict(user_input_transformed)
            # Update and display metrics with predicted score
            metrics_df.at[5, 'Value'] = f"{int(round(prediction[0]))}"
            st.write("### Model Performance Metrics")
            st.table(metrics_df)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    # Show graph button
    if st.button('Show Graph'):
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title('Actual vs. Predicted Scores')
        
        st.pyplot(fig)
    
    # Add footer with stylish underline
    st.markdown(
        """
        <style>
        /* Style the title */
        .stApp .css-1d391kg {
            font-size: 2.5rem;
            color: #14299c;
            text-align: center;
        }
        
        /* Style the sidebar */
        .stSidebar {
            background-color: #f0f0f5;
            padding: 20px;
        }
        
        /* Style the input widgets */
        .stSidebar .stNumberInput, .stSidebar .stSelectbox {
            margin-bottom: 20px;
        }
        
        /* Style the buttons */
        .stButton button {
            background-color: #14299c;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
         .stButton button:hover {
            background-color: #0f1f6b;
        }
        
        /* Style the tables */
        .stTable {
            margin-top: 20px;
        }
        
        /* Style the footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000;
            color: #f1f1f1;
            text-align: center;
            padding: 15px;
            font-size: 16px;
            box-shadow: 0 -1px 5px rgba(0,0,0,0.3);
        }
        .footer a {
            color: #14299c;
            text-decoration: none;
            font-size: 22px;
            font-weight: bold;
            position: relative;
            display: inline-block;
        }
         .footer a::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -5px;
            width: 100%;
            height: 2px;
            background: #14299c;
            transform: scaleX(0);
            transform-origin: bottom right;
            transition: transform 0.3s ease-out;
        }
        .footer a:hover::after {
            transform: scaleX(1);
            transform-origin: bottom left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

if __name__ == '__main__':
    main()
