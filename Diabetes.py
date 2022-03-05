import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# title and subtitle
st.write("""                                                                           
# Diabetes Detection
Detect if someone has diabetes using ML and Python
""")

# Reads the CSV File
df = pd.read_csv('C:/Users/VIKASH K S/PycharmProjects/HackHub_HealthCare/diabetes.csv')

# Creates a subheading
st.subheader('Data Information:')

# Creates a Table
st.dataframe(df)

# Show Statistics on the data at hand
st.write(df.describe())

# Showing data as chart
chart = st.bar_chart(df)

# Split the data into Independent X and Dependent Y variables
X = df.iloc[:, 0:8].values          # getting all elements till Age
Y = df.iloc[:, -1].values           # getting Outcome column alone

# Split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)


    # Store a dictionary into a variable
    user_data = {'preganancies' : pregnancies,
                 'glucose' : glucose,
                 'blood_pressure' : blood_pressure,
                 'skin_thickness' : skin_thickness,
                 'insulin' : insulin,
                 'BMI' : BMI,
                 'DPF' : DPF,
                 'age' : age
                 }

    #Transform the Data into a Dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the user input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

# Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classifications
st.subheader('Classifications:')
st.write(prediction)
if prediction ==1:
    st.write('High Chance of Diabetes')
else:
    st.write('Low Chance of Diabetes')
