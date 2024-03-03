import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# Import Modules for Machine Learning
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import altair as alt
import seaborn as sns
from sklearn.metrics import confusion_matrix

pakistan_soil = pd.read_csv("pakistan_soils.csv")

#poof?

st.sidebar.header('More Information')

st.title("""
 :seedling: Croptimizer

""")

st.write("""
Welcome to Croptimizer!

By: Laira L, Meghan L, Stella H, & Arjun R

We use machine learning to calculate which crops you should grow based on your conditions. We also include a calculator to find 
which nutrient additives you should add to your crops. Growing the most favorable crops to your area as well as applying the optimal nutrient
additives can reduce excess fertilizer and nutrient additives which can lead to higher costs and eutrophication. 

Due to the recent flooding in Pakistan which destroyed a large portion of agricultural land, we have a specific modification
of the tool to make it easier for Pakistani farmers to use.

Use the dropdown boxes below to select the application right for you. 

Use the sidebar if you would like to see the data sources.

""")

if (st.button('Learn more about Croptimizer!')):
    st.write("""
    Many developing countries that rely on farming as a large source of GDP are struggling against climate change and the effects 
    it has on their agriculture. Sustainable agriculture uses practices that help maintain or improve conditions for farming to 
    protect the livelihood of future generations. It helps ensure that countries with large farming operations can maintain or 
    increase their yields and strengthen their economy and people without damage to the environment.
    """)

    st.write("""
    The Croptimizer Web Application can be used to tell farmers what crops are ideal to grow depending on the characteristics and 
    composition of the soil. Additionally to reduce fertilizer runoff to prevent eutrophication, especially in areas prone to flooding, like Pakistan. 
    We created a calculator to determine nutrient amounts when creating a fertilizer solution, which is ideal for hydroponics, 
    but can also be applied to soil.
    """)

    st.image(
        'https://media.npr.org/assets/img/2022/08/29/gettyimages-1242738982_custom-443c2892eb4ca3bb2a68e31aeb7f3f1368d3f5d2.jpg')
    st.write('2022 Flooding in Pakistan (NPR)')

    st.write("""
    Pakistan is known to have some of the highest disaster risk levels in the world. 
    Monsoon season, which spans from mid-June to the end of September, causes prolonged rainfall and heavy precipitation 
    during the summer, often causing extreme flooding. One of the most recent severe floods was in the summer of 2022 when 
    Pakistan declared a state of emergency due to the extreme rainfall that flooded almost one-third of the country. 
    We chose to focus on Pakistan to enable farmers to utilize our website to analyze their soil composition after 
    the flooding and identify crops that are best suited for their current soil as well as individually optimizing 
    their soil composition to meet the ideal nutrient soil requirements.
    """)

# tab1, tab2, tab3 = st.tabs(['Pakistani Farmers', 'Predict your Crop Tool', 'Fertilizer Tool'])

with st.expander("""Are you a farmer in Pakistan?"""):
    st.write("""

    Simply enter your state below and click on the Pakistan Application Button and scroll down to see your results.

    """)

    states_pakistan = pakistan_soil['Sampling Area'].to_numpy()
    state = st.selectbox("Enter your State:",
                         options=states_pakistan)

    if (st.button('See your Results')):

        data_p = pd.read_csv("rf_values.csv")

        st.write("""
                #### 1 means the algorithm predicted the crop would grow

                    """)

        if state == 'Bahawalpur':
            data_s = data_p[['label', 'Can Grow 0']]
            x_label = 'Can Grow 0'
        elif state == 'Charsada':
            data_s = data_p[['label', 'Can Grow 1']]
            x_label = 'Can Grow 1'
        elif state == 'Gujranwala':
            data_s = data_p[['label', 'Can Grow 2']]
            x_label = 'Can Grow 2'
        elif state == 'Khewra':
            data_s = data_p[['label', 'Can Grow 3']]
            x_label = 'Can Grow 3'
        elif state == 'Mian channu':
            data_s = data_p[['label', 'Can Grow 4']]
            x_label = 'Can Grow 4'
        elif state == 'Mianwali':
            data_s = data_p[['label', 'Can Grow 5']]
            x_label = 'Can Grow 5'
        elif state == 'Multan':
            data_s = data_p[['label', 'Can Grow 6']]
            x_label = 'Can Grow 6'
        elif state == 'Abbottabad':
            data_s = data_p[['label', 'Can Grow 7']]
            x_label = 'Can Grow 7'
        elif state == 'Haripur':
            data_s = data_p[['label', 'Can Grow 8']]
            x_label = 'Can Grow 8'
        elif state == 'Layyah':
            data_s = data_p[['label', 'Can Grow 9']]
            x_label = 'Can Grow 9'
        elif state == 'Mansehra':
            data_s = data_p[['label', 'Can Grow 10']]
            x_label = 'Can Grow 10'
        elif state == 'Okara':
            data_s = data_p[['label', 'Can Grow 11']]
            x_label = 'Can Grow 11'
        elif state == 'Sahiwal':
            data_s = data_p[['label', 'Can Grow 12']]
            x_label = 'Can Grow 12'
        else:
            data_s = data_p[['label', 'Can Grow 13']]
            x_label = 'Can Grow 13'

        data_s = data_s.sort_values(by=x_label, ascending=False)

        st.dataframe(data_s, use_container_width=True)

        st.write("""
                    #### Which Crops Can be Grown?

                    The crops with the blue line can be grown.
                    """)

        c = alt.Chart(data_s).mark_bar().encode(
            x=x_label,
            y='label'
        )

        st.altair_chart(c, theme=None, use_container_width=True)

    st.write("""
        ##### Get More Data
        Would you like to try other algorithms besides Random Forest? See the dataset below for original scaled values and 
        use the dropdown box below. """)

    if (st.button('See Original Values')):
        p_data = pd.read_csv("p_data.csv")
        st.dataframe(p_data, use_container_width=True)

# model = st.sidebar.selectbox(
# 'Choose Machine Learning Algorithm:',
# options=np.array(['Logistic Regression', 'Random Forest', 'Decision Tree'])
# )


data = pd.read_csv("Crop_recommendation.csv")

# print(data.head())

labels = data[['label']]

data['Ratio N'] = data['N'] / (data['N'] + data['P'] + data['K'])
data['Ratio P'] = data['P'] / (data['N'] + data['P'] + data['K'])
data['Ratio K'] = data['K'] / (data['N'] + data['P'] + data['K'])

# One hot-encode the label
data_encode = pd.get_dummies(data, columns=['label'])

data_encode_dropped = data_encode.drop(
    ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Ratio N', 'Ratio P', 'Ratio K'], axis=1).head(5)

encoded_labels = data_encode_dropped.columns

data_encode.sample(frac=1)


def running_a_model_for_a_crop(label_crop):
    y = data_encode[label_crop]
    x = data_encode[['temperature', 'humidity', 'ph', 'rainfall', 'Ratio N', 'Ratio P', 'Ratio K']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def machine_learning(x):
    if x == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif x == 'Random Forest':
        model = RandomForestClassifier(random_state=0)
    else:
        model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1)
    return model


def what_can_you_grow(x_data, model_choice):
    values = np.array([])
    scores = np.array([])
    for i in encoded_labels:
        x_train, x_test, y_train, y_test = running_a_model_for_a_crop(i)
        model_choice.fit(x_train, y_train)
        score = model_choice.score(x_test, y_test)
        scores = np.append(scores, score)
        pred = model_choice.predict(x_data)
        values = np.append(values, pred)
    return values, scores


with st.expander("Are you a farmer or gardener who wants a more detailed prediction based on your conditions?"):
    st.write("""
    Use the sliders below to enter information about your area. Note the unit you use to record nitrogen, potassium, and 
    phosphorus content is not important as long as it is consistent among the three. 

    Then, click on the button Predict your Crops in the sidebar and then scroll down.

    """)
    col1, col2 = st.columns(2)

    with col1:
        n = st.slider("Enter the nitrogen content of your soil", 0, 100, 18)
        k = st.slider("Enter the potassium content of your soil", 0, 100, 9)
        p = st.slider("Enter the phosphorus content of your soil", 0, 100, 9)
        ph = st.slider("Enter the pH of your soil", 0, 14, 7)

    with col2:
        rain = st.slider("Enter the average yearly rainfall of your area in mm", 0, 1000, 200)
        temp = st.slider("Enter the average temperature of your area in degrees Celsius", 0, 100, 20)
        humd = st.slider("Enter the average humidity of your area in degrees Celsius", 0, 100, 40)

    st.write("""
    If you would like to choose among Random Forest, Logistic Regression, or Decision Trees as your machine learning algorithm use 
    the select in the sidebar. """)

    model = st.selectbox(
        'Choose Machine Learning Algorithm:',
        options=np.array(['Logistic Regression', 'Random Forest', 'Decision Tree'])
    )

    x_train, x_test, y_train, y_test = running_a_model_for_a_crop('label_coffee')

    # Making the model

    ml_model = machine_learning(model)

    scaler = MinMaxScaler().fit(x_train)

    if (st.button('Predict your Crops')):
        n_ratio = n / (n + p + k)
        p_ratio = p / (n + p + k)
        k_ratio = k / (n + p + k)

        x_data = np.array([[temp, humd, ph, rain, n_ratio, p_ratio, k_ratio]])

        x_data = scaler.transform(x_data)
        x_data.reshape(-1, 1)

        # Which crops can you use?

        values, scores = what_can_you_grow(x_data, ml_model)

        crops_we_can_grow = pd.DataFrame()

        crops_we_can_grow['Crop'] = encoded_labels
        crops_we_can_grow['Can Grow'] = values
        crops_we_can_grow['Accuracy Score'] = scores

        st.write(
            """
            #### Results
            """
        )

        st.write("""
        Below you will find the outcome of the machine learning algorithm. When a crop has a value of 1, then your conditions are suitable for its growth.
        The dataset also contains the accuracy for the respective machine learning algorithm which was used to make the prediction. Because the machine learning 
        algorithms are separate and some crops have similar requirements, multiple crops can be predicted to be ideal for your conditions.

        """)

        st.write("""
        #### 1 means the algorithm predicted the crop would grow

            """)

        crops_we_can_grow = crops_we_can_grow.sort_values(by='Can Grow', ascending=False)

        st.dataframe(crops_we_can_grow, use_container_width=True)

        # st.bar_chart(crops_we_can_grow['Crop'], crops_we_can_grow['Can Grow'])

        data_3 = {"Crop": crops_we_can_grow['Crop'], "Can Grow": crops_we_can_grow['Can Grow']}
        df = pd.DataFrame(data_3)

        st.write("""
            #### Which Crops Can be Grown?

            The Crop with the blue line can be grown.


            """)

        c = alt.Chart(df).mark_bar().encode(
            x='Can Grow',
            y='Crop'
        )

        st.altair_chart(c, theme=None, use_container_width=True)

with st.expander("Nutrient Additive Calculator"):
    st.write("""
    The calculator is ideal for hydroponics (growing plants without soil as a media). 
    However, it can still be applied to traditional agriculture but be mindful of slight inaccuracy. 
    """)
    data_nut = pd.read_csv("soil_hack_data.csv")
    liters = st.number_input("Input the amount of water in liters:", 1)
    crop_nut = st.selectbox("Which crop?", options=np.array(
        ['Cotton', 'Rice', 'Sugarcane', 'Wheat', 'Barley', 'Legumes', 'Sorghum', 'Leafy Vegetables',
         'Tomatoes', 'Cucumbers']))


    def calcSolution(crop, liters):

        row = 0

        if crop == "cotton":
            row = 0
        elif crop == "rice":
            row = 1
        elif crop == "sugarcane":
            row = 2
        elif crop == "wheat":
            row = 3
        elif crop == "barley":
            row = 4
        elif crop == "legumes":
            row = 5
        elif crop == "sorghum":
            row = 6
        elif crop == "leafy vegetables":
            row = 7
        elif crop == "tomatoes":
            row = 8
        elif crop == "cucumbers":
            row = 9
        dfRow = data_nut.iloc[row]
        return "To plant " + crop + " you will need " + str(dfRow.iloc[13] * liters) + " mg of Calcium nitrate, " + str(
            dfRow.iloc[14] * liters) + " mg of Potassium phosphate, " + str(
            dfRow.iloc[15] * liters) + " mg of Potassium nitrate, " + str(
            dfRow.iloc[16] * liters) + " mg of Magnesium sulfate, " + str(
            dfRow.iloc[17] * liters) + " mg of Chelated iron, " + str(dfRow.iloc[18] * liters) + " mg of Boron, " + str(
            dfRow.iloc[19] * liters) + " mg of Copper sulfate, " + str(
            dfRow.iloc[20] * liters) + " mg of Manganese sulfate, " + str(
            dfRow.iloc[21] * liters) + " mg of Zinc sulfate, and " + str(
            dfRow.iloc[22] * liters) + " mg of Sodium molybdate to make a " + str(liters) + " liter solution."


    st.write(calcSolution(crop_nut, liters))

    st.write("""
    ###                  :corn: :sunflower: :apple:
    """)

with st.expander("Do you want to learn more about the machine learning algorithms?"):
    st.write("""This comparison of the machine learning algorithms uses the label rice. 
    So the machine learning predicts whether or not the conditions are right to make rice.
    The real application will apply this to all 22 crops.
    """)

    st.write("""
    The confusion matrices show the true and false positives and negatives for each 
    machine learning model.
    """)

    # Showing the statistics for the different machine learning with rice

    x_train_r, x_test_r, y_train_r, y_test_r = running_a_model_for_a_crop('label_rice')

    logreg = machine_learning('Logistic Regression')
    dtree = machine_learning('Decision Tree')
    randomf = machine_learning('Random Forest')

    st.write("""
    ##### Logistic Regression - Default
    """)

    logreg.fit(x_train_r, y_train_r)

    score_log_reg = logreg.score(x_test_r, y_test_r)

    st.write("""Here is the accuracy of the Logistic Regression out of 1:""")
    st.write(score_log_reg)

    y_pred_lr = logreg.predict(x_test_r)

    cm = confusion_matrix(y_test_r, y_pred_lr)

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots()

    sns.heatmap(cm_matrix, ax=ax, annot=True, fmt='d', cmap='YlGnBu')
    st.write(fig)

    st.write("""
    ##### Decision Tree
    """)

    dtree.fit(x_train_r, y_train_r)

    score_dtree = dtree.score(x_test_r, y_test_r)

    st.write("""Here is the accuracy of the Decision Tree out of 1:""")
    st.write(score_dtree)

    y_pred_dt = dtree.predict(x_test_r)

    cm = confusion_matrix(y_test_r, y_pred_dt)

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots()

    sns.heatmap(cm_matrix, ax=ax, annot=True, fmt='d', cmap='YlGnBu')
    st.write(fig)

    st.write("""
    ##### Random Forest
    """)
    randomf.fit(x_train_r, y_train_r)

    score_randomf = randomf.score(x_test_r, y_test_r)

    st.write("""Here is the accuracy of the Random Forest out of 1:""")
    st.write(score_randomf)

    y_pred_rf = randomf.predict(x_test_r)

    cm = confusion_matrix(y_test_r, y_pred_rf)

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])

    fig, ax = plt.subplots()

    sns.heatmap(cm_matrix, ax=ax, annot=True, fmt='d', cmap='YlGnBu')
    st.write(fig)



if (st.sidebar.button('Data Sources')):
    st.sidebar.write('Here are the sources we used throughout this project:')

    st.sidebar.write("""

  Haris Aziz, Muhammad, et al. “Identification of Soil Type in Pakistan Using Remote Sensing and Machine Learning.” 
  National Library of Medicine, PubMed Central, 3 Oct. 2022, www.ncbi.nlm.nih.gov/pmc/articles/PMC9575843/#ref-27. 

  Hochmuth, George J. “HS787/CV265: Fertilizer Management for Greenhouse Vegetables-Florida Greenhouse Vegetable Production Handbook, 
  Vol 3.” Ask IFAS - Powered by EDIS, University of Florida, 21 Apr. 2022, edis.ifas.ufl.edu/publication/cv265. 

  Ingle, Atharva. “Crop Recommendation Dataset.” Kaggle, 2021, Accessed 2024. 

  Mohiuddin, Muhammad, et al. “Relationship of selected soil properties with the micronutrients in salt-affected soils.” 
  Land, vol. 11, no. 6, 4 June 2022, p. 845, https://doi.org/10.3390/land11060845. 

  “Utah Hydroponic Solutions.” Utah State University, 4 Apr. 2022. 

  “Weatherandclimate.com.” Weatherandclimate.com, weatherandclimate.com/.




  """)

    # fig, ax = plt.subplots()
    # ax.set_title('Crops That Can be Grown')
    # ax.barh(crops_we_can_grow['Crop'], crops_we_can_grow['Can Grow'])
    # ax.set_ylabel('Crop')
    # ax.set_xlabel('Can be Grown? (Yes = 1)')
    # st.pyplot(fig)












