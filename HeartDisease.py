import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# title and subtitle
st.write("""                                                                           
# Heart Disease Detection
Detect if someone has Heart Disease using ML and Python
""")

# Reads the CSV File
df = pd.read_csv('C:/Users/VIKASH K S/PycharmProjects/HackHub_HealthCare/heart.csv')

# Creates a subheading
st.subheader('Data Information:')

# Creates a Table
st.dataframe(df)

# Show Statistics on the data at hand
st.write(df.describe())

# Showing data as chart
chart = st.bar_chart(df)

# Split the data into Independent X and Dependent Y variables
X = df.iloc[:, 0:13].values          # getting all elements till Age
Y = df.iloc[:, -1].values           # getting Outcome column alone

# Split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():
    age = st.sidebar.slider('Age', 0, 77, 52)
    sex = st.sidebar.slider('Sex', 0, 1, 1)                                                            # 0-female, 1-male
    cp = st.sidebar.slider('Chest Pain', 0, 3, 0)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 125)
    chol = st.sidebar.slider('Cholestrol', 126, 564, 212)
    fbs = st.sidebar.slider('Fasting Blood Sugar', 0, 1, 0)
    restecg = st.sidebar.slider('Rest ECG', 0, 2, 1)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 168)
    exang = st.sidebar.slider('Exercise Induced Angina', 0, 1, 0)                                       # 0-no, 1-yes
    oldpeak = st.sidebar.slider(' ST depression induced by exercise relative to rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.slider('Slope', 0, 2, 2)
    ca = st.sidebar.slider('Number of major vessels', 0, 4, 2)
    thal = st.sidebar.slider('thal', 0, 3, 3)                                                           # Normal =0, Fixed Defect=1, Reversable Defect=2


    # Store a dictionary into a variable
    user_data = {'age': age,
                 'sex': sex,
                 'cp': cp,
                 'trestbps': trestbps,
                 'chol': chol,
                 'fbs': fbs,
                 'restecg': restecg,
                 'thalach': thalach,
                 'exang': exang,
                 'oldpeak': oldpeak,
                 'slope': slope,
                 'ca': ca,
                 'thal': thal,
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
    st.write('High Chance of Heart Disease')
else:
    st.write('Low Chance of Heart Disease')