import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)

st.set_page_config(
    page_title="Impact of Dune Plants on Caretta Caretta",
    page_icon="🐢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_raw_data():
    """
    This function returns a pandas DataFrame with the raw data.
    """
    raw_df = pd.read_csv('ReddingRootsCaseStudy22_csv.csv')
    raw_df = raw_df[0:93]
    raw_df = raw_df.drop(columns=["Comments", "Notes", "Unnamed: 42"])
    raw_df = raw_df.drop(columns=["Species", "Key"])
    return raw_df

def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """
    clean_data = pd.read_csv('cleaned_df.csv')
    return clean_data
df_c = get_cleaned_data()
def summary_table(df):
    """
    Return a summary table with the descriptive statistics about the dataframe.
    """
    summary = {
    "Number of Variables": [len(df.columns)],
    "Number of Observations": [df.shape[0]],
    "Missing Cells": [df.isnull().sum().sum()],
    "Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
    "Duplicated Rows": [df.duplicated().sum()],
    "Duplicated Rows (%)": [round(df.duplicated().sum() / df.shape[0] * 100, 2)],
    "Categorical Variables": [len([i for i in df.columns if df[i].dtype==object])],
    "Numerical Variables": [len([i for i in df.columns if df[i].dtype!=object])],
    }
    return pd.DataFrame(summary).T.rename(columns={0: 'Values'})

def grab_col_names(dataframe, cat_th=10, car_th=20):  #kategorik, nümerik değişkenleri ayıklamak için
  ###cat_cols
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                 dataframe[col].dtypes != "O"]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                 dataframe[col].dtypes == "O"]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]
  ###num_cols
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
  num_cols = [col for col in num_cols if col not in num_but_cat]
  print(f"observations: {dataframe.shape[0]}")
  print(f"variables: {dataframe.shape[1]}")
  print(f"cat_cols: {len(cat_cols)}")
  print(f"num_cols: {len(num_cols)}")
  print(f"cat_but_car: {len(cat_but_car)}", f"cat_but_car name: {cat_but_car}")
  print(f"num_but_cat: {len(num_but_cat)}", f"num_but_cat name: {num_but_cat}")
  return cat_cols, num_cols, cat_but_car


condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "Model Prediction", "Model Evaluation")
)

########################## INTRODUCTION ###################################

if condition == 'Introduction':
    col1, col2 = st.columns([2, 8])

    with col1:
        st.image("logo_turtle.png", width=140)
    with col2:
        st.title("Impact of Dune Plants on Loggerhead Sea Turtles")
    st.write("An explanatory website for our Samsung IC capstone project.")
    st.header("Introduction")

    st.markdown("""
        In this project, observing the effects of dune plants on loggerhead sea turtle eggs hatching success is aimed. 
        The nesting success of loggerhead sea turtles (Caretta caretta) is highly dependent on dune plant
        roots, since they lay their eggs near dunes. Dune plant roots can cause harm to the eggs and reduce
        the chances of hatching by breaking them. Therefore, it is crucial to analyze and optimize the effect of
        dune plants upon the hatching of the eggs.
    """)


    image1 = Image.open('seaturtle_img.jpg')
    width, height = image1.size
    image1 = image1.resize((420, 300))
    st.image(image1, caption='Loggerhead Sea Turtle')

    st.markdown("""
        ### About the Dataset
        The dataset is publicly available in the Dryad data repository. It is published in 2024 and there is also
        a reference paper describing the process of data collection and statistical analysis of the data [2]. In
        the process of creating the data, they monitored 93 nests for 6 months in 2022. The data are in tabular
        form and there are 42 features and 93 samples.
        [Link to the Dataset](https://datadryad.org/stash/dataset/doi:10.5061/dryad.zw3r228dk)
        """)
    st.markdown("""
    ### Plant Types in Pie Chart
    The plants classified as "Others" includes plant types such as: palm, christmas cactus, unknown, dune sunflower,
    seaside spurge (sandmat). These are not shown in the pie chart since their percentage is really low.
    """)

    image2 = Image.open('piechart.png')
    width, height = image2.size
    image2 = image2.resize((550, 500))
    st.image(image2, caption='Pie chart to show different plant types in the dataset ')

    st.markdown("""
        ### Sustainable Development Goal (SDG) of the Project
        Since the main objective of the project is loggerhead sea turtles, this project falls within the scope of
        SDG-14 (life under water).
        [For further information about SDG-14](https://sdgs.un.org/goals/goal14)
        """)

    image2 = Image.open('sdg14.png')
    width, height = image2.size
    image2 = image2.resize((200, 200))
    st.image(image2, caption='SDG-14 Logo')

########################## EDA ###################################

elif condition == 'EDA':
    col1, col2 = st.columns([2, 11])

    with col1:
        st.image("logo_turtle.png", width=110)
    with col2:
        st.title("Explanatory Data Analysis (EDA)")

    df_c = get_cleaned_data()
    df_raw = get_raw_data()

    st.header("Dataset Review")
    dataset_choice = st.radio("Choose Dataset Preview", ("Original Dataset", "Cleaned and Processed Dataset"))

    #Kullanıcının seçimine göre dataset'i gösterelim
    if dataset_choice == "Original Dataset":
            if st.button("Head"):
                st.write(df_raw.head())

            if st.button("Tail"):
                st.write(df_raw.tail())

            if st.button("Show All DataFrame"):
                st.dataframe(df_raw)

    elif dataset_choice == "Cleaned and Processed Dataset":
        if st.button("Head"):
            st.write(df_c.head())
        if st.button("Tail"):
            st.write(df_c.tail())
        if st.button("Show All DataFrame"):
            st.dataframe(df_c)

    st.header("Summary of the Dataset Properties")
    if st.button("Summary of original dataset"):
        st.write(summary_table(df_raw))
        cat_cols, num_cols, cat_but_car = grab_col_names(df_raw)
        df_cat_cols = pd.DataFrame({"Categorical Columns": cat_cols})
        df_num_cols = pd.DataFrame({"Numeric Columns": num_cols})
        df_car_cols = pd.DataFrame({"Numeric Columns": cat_but_car})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(df_cat_cols)

        with col2:
            st.write(df_num_cols)

        with col3:
            st.write(df_car_cols)
    if st.button("Summary of cleaned dataset"):
        st.write(summary_table(df_c))
        cat_cols, num_cols, cat_but_car = grab_col_names(df_c)
        df_cat_cols = pd.DataFrame({"Categorical Columns": cat_cols})
        df_num_cols = pd.DataFrame({"Numeric Columns": num_cols})
        df_car_cols = pd.DataFrame({"Numeric Columns": cat_but_car})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(df_cat_cols)

        with col2:
            st.write(df_num_cols)

        with col3:
            st.write(df_car_cols)


    st.header("Visualizing Missing Values")
    import missingno as msno
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.button("Matrix Rep."):
        msno.matrix(df_raw)
        st.pyplot()
    if st.button("Bar Plot Rep."):
        msno.bar(df_raw)
        st.pyplot()

    st.markdown("""...
        """)

    st.header("Visualizing Outliers")

########### outlierları görme: ##############
    cat_cols, num_cols, cat_but_car = grab_col_names(df_raw)
    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
      quartile1 = dataframe[col_name].quantile(q1)
      quartile3 = dataframe[col_name].quantile(q3)
      IQR_range = quartile3 - quartile1
      up_lim = quartile3 + 1.5 * IQR_range
      low_lim = quartile1 - 1.5 * IQR_range
      return low_lim, up_lim

    for col in num_cols:
      low, up = outlier_thresholds(df_raw, col)

      def grab_outliers(dataframe, col_name, index=False):
        low, up = outlier_thresholds(dataframe, col_name)
        count = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)][col_name].shape[0]
        print(col_name.upper(), f" has {count} outliers.")
        if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] != 0:
          if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head(5))
          else:
            print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
        if index:
          outlier_index = (dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]).index
          return outlier_index

    for col in num_cols:
      grab_outliers(df_raw, col, index=True)


    def check_outlier(dataframe, col_name):
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
          return True
        else:
          return False
    for col in num_cols:
      print(col, check_outlier(df_raw, col))

    ################ outlierları baskılama (threshold ile): ################
    def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.90):
      low, up = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.90)
      dataframe.loc[(dataframe[variable] < low), variable] = low
      dataframe.loc[(dataframe[variable] > up), variable] = up

    #outlier boxplot göstereceğimiz bir dropdown elementi:
    selected_column = st.selectbox('Please select a numerical column', num_cols)
    outliers = grab_outliers(df_raw, selected_column)
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_raw[selected_column])
    plt.title(f'Box Plot: {selected_column}', fontsize=15)
    plt.xlabel(selected_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Correlation Analysis")

    if st.checkbox("Correlation Matrix"):

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_raw[num_cols].corr(), annot=True, cmap='coolwarm',
                    cbar_kws={'label': 'Korelasyon'})
        plt.title('Korelasyon Matrisi (Heatmap)', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot()

########################## MODEL PREDICTION ###################################
elif condition == 'Model Prediction':
    col1, col2 = st.columns([2, 12])
    with col1:
        st.image("logo_turtle.png", width=100)
    with col2:
        st.title("Model Prediction")

    st.markdown("""
            This section is made to help scientists who work in related projects. """)

    def load_model(modelfile):
        loaded_model = pickle.load(open(modelfile, 'rb'))
        return loaded_model

    model_option = st.selectbox(
        "Select Machine Learning Model:",
        ("Random Forest", "Decision Tree", "Linear Regression")
    )

    html_temp = """
            <div>
            <h1 style="color:BLACK;text-align:left;"> Enter New Data and Predict</h1>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)
    with st.expander(" ℹ️ Information About the Features", expanded=False):
     st.write("""
            <ol>
            <li>ZoneID - Zone of nest locations</li>
            <li>Lat - Latitude of nest</li>
            <li>Long - Longitude of nest</li>
            <li>VegPresence - Presence/absence (1/0) of vegetation around nest</li>
            <li>VegType - Species of vegetation around nest</li>
            <li>RootPresence - Presence/absence (1/0) of roots around nest</li>
            <li>PlantRoot - Species of plant roots belonged to</li>
            <li>DistBarrier - Distance of nest to the barrier (m)</li>
            <li>DistHighWater - Distance of nest to the high water mark (m)</li>
            <li>TotalDist - Total width of beach (m)</li>
            <li>LocOnBeach - Location of nest on the beach</li>
            <li>Division - Which division nest was located on beach; beach was divided into thirds</li>
            <li>SurfaceDepth - Depth from surface to first egg (cm)</li>
            <li>BottomDepth - Depth from surface to bottom of nest chamber (cm)</li>
            <li>InternalDepth - Internal nest chamber depth (cm)</li>
            <li>CavityWidth - Width of the nest cavity, from wall to wall (cm)</li>
            <li>Hatched - Number of eggs hatched</li>
            <li>Unhatched - Number of eggs unhatched</li>
            <li>Developed_UH - Number of unhatched eggs with developed hatchling</li>
            <li>LivePip - Number of live pipped hatchlings</li>
            <li>DeadPip - Number of dead pipped hatchlings</li>
            <li>Yolkless - Number of yolkless eggs</li>
            <li>EncasedTotal - Number of total root-encased eggs</li>
            <li>DevEnc_UH - Number of root-encased unhatched eggs with developed hatchling</li>
            <li>H_Encased - Number of root-encased hatched eggs</li>
            <li>UH_Encased - Number of root-encased unhatched eggs</li>
            <li>InvadedTotal - Number of total root-invaded eggs</li>
            <li>H_Invaded - Number of root-invaded hatched eggs</li>
            <li>UH_Invaded - Number of root-invaded unhatched eggs</li>
            <li>Live - Number of live hatchlings</li>
            <li>Dead - Number of dead hatchlings</li>
            <li>Depredated - Depredation of nest (yes/no; 1/0)</li>
            <li>RootDamageProp - Proportion of root damaged eggs</li>
            <li>HS - Hatch success</li>
            <li>ES - Emergence success</li>
            <li>TotalEggs - Total eggs within the nest</li>
            </ol>
            """, unsafe_allow_html=True)

    st.subheader("Please enter information about the data 🐢🌿")

    col1, col2 = st.columns([2, 2])
    with col1:
        zone = st.number_input("ZoneID", 1, 10, value=5)
        lat = st.number_input("Lat", 1.0, 50.0, value=27.14)
        long = st.number_input("Long", -100.0, 0.0, value=-82.48)
        vegpresence = st.number_input("VegPresence ", 0, 1, value=1)
        vegtype_options = ['-railroad vine', '-sea oats', 'no', '-sea purslane', "-sea grapes"]
        vegtype = st.selectbox("VegType", options=vegtype_options)
        rootpres = st.number_input("RootPresence", 0, 1, value=1)
        roottype_options = ['Railroad Vine', 'Sea Oats', 'no']
        roottype = st.selectbox("PlantRoot", options=roottype_options)
        distbarr = st.number_input("DistBarrier", -20.0, 20.0, value=1.52)
        disthighw = st.number_input("DistHighWater", 0.0, 1000.0, value=14.05)
        totaldist = st.number_input("TotalDist", 0.0, 50.0, value=15.54)
        LocOnBeach = st.number_input("LocOnBeach ", 0.0, 5.0, value=1.02)
        division_options = ['M', 'U', 'L']
        Division = st.selectbox("Division", options=division_options)
        SurfaceDepth = st.number_input("SurfaceDepth ", 0.0, 1000.0, value=20.0)
        BottomDepth = st.number_input("BottomDepth", 0.0, 1000.0, value=33.0)
        InternalDepth = st.number_input("InternalDepth ", 0.0, 1000.0, value=20.0)
        CavityWidth = st.number_input("CavityWidth ", 0.0, 1000.0, value=23.0)
        Hatched = st.number_input("Hatched ", 0.0, 1000.0, value=45.0)
        Unhatched = st.number_input("Unhatched", 0.0, 1000.0, value=6.0)

    with col2:
        Developed_UH = st.number_input("Developed_UH ", 0.0, 1000.0, value=16.0)
        LivePip = st.number_input("LivePip", 0.0, 1000.0, value=1.0)
        DeadPip = st.number_input("DeadPip ", 0.0, 1000.0, value=5.0)
        Yolkless = st.number_input("Yolkless  ", 0.0, 1000.0, value=0.0)
        EncasedTotal = st.number_input("EncasedTotal  ", 0.0, 1000.0, value=0.0)
        DevEnc_UH = st.number_input("DevEnc_UH", 0.0, 1000.0, value=0.0)
        H_Encased = st.number_input("H_Encased  ", 0.0, 1000.0, value=0.0)
        UH_Encased = st.number_input("UH_Encased ", 0.0, 1000.0, value=3.0)
        InvadedTotal = st.number_input("InvadedTotal  ", 0.0, 1000.0, value=0.0)
        H_Invaded = st.number_input("H_Invaded", 0.0, 1000.0, value=0.0)
        UH_Invaded = st.number_input("UH_Invaded", 0.0, 1000.0, value=0.0)
        Live = st.number_input("Live", 0.0, 1000.0, value=3.0)
        Dead = st.number_input("Dead", 0.0, 1000.0, value=2.0)
        Depredated = st.number_input("Depredated  ", 0.0, 1000.0, value=0.0)
        RootDamageProp = st.number_input("RootDamageProp", 0.0, 1000.0, value=0.0125)
        ES = st.number_input("ES ", 0.0, 1000.0, value=95.0)
        TotalEggs = st.number_input("TotalEggs", 0.0, 1000.0, value=65.0)
        MultiVegNum = 0.0
        feature_list = [zone, lat, long, vegpresence, vegtype, rootpres, roottype, distbarr, disthighw,
                            totaldist, LocOnBeach, Division, SurfaceDepth, BottomDepth, InternalDepth,
                            CavityWidth, Hatched, Unhatched, Developed_UH, LivePip, DeadPip, Yolkless,
                            EncasedTotal, DevEnc_UH, H_Encased, UH_Encased, InvadedTotal, H_Invaded, UH_Invaded,
                            Live, Dead, Depredated, RootDamageProp, ES, TotalEggs, MultiVegNum
                            ]
        cat_cols, num_cols, cat_but_car = grab_col_names(df_c)

        from sklearn.preprocessing import RobustScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer

        columns = ['ZoneID', 'Lat', 'Long', 'VegPresence', 'VegType', 'RootPresence', 'PlantRoot', 'DistBarrier', 'DistHighWater', 'TotalDist',
                   'LocOnBeach', 'Divisions', 'SurfaceDepth', 'BottomDepth', 'InternalDepth', 'CavityWidth', 'Hatched', 'Unhatched', 'Developed_UH',
                   'LivePip', 'DeadPip', 'Yolkless', 'EncasedTotal', 'DevEnc_UH',
                   'H_Encased', 'UH_Encased', 'InvadedTotal', 'H_Invaded', 'UH_Invaded', 'Live', 'Dead', 'Depredated', 'RootDamageProp'
                   , 'ES', 'TotalEggs', 'MultiVegNum']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), num_cols),
                ('cat', OneHotEncoder(), cat_cols)
            ])

        def one_hot_encoder(dataframe, categoric_cols, drop_first=False):
            dataframe = pd.get_dummies(dataframe, columns=categoric_cols, drop_first=drop_first, dtype=int)
            return dataframe

        expected_names = ['VegPresence', 'RootPresence', 'DistBarrier', 'DistHighWater', 'TotalDist', 'LocOnBeach', 'SurfaceDepth', 'BottomDepth', 'InternalDepth', 'CavityWidth', 'Hatched', 'Unhatched', 'Developed_UH', 'DeadPip', 'EncasedTotal', 'H_Encased', 'UH_Encased', 'Live', 'RootDamageProp', 'ES', 'TotalEggs', 'VegType_-christmas cactus', 'VegType_-crested saltbrush', 'VegType_-dune sunflower', 'VegType_-palm', 'VegType_-railroad vine', 'VegType_-salt grass', 'VegType_-sea grapes', 'VegType_-sea oats',
       'VegType_-sea purslane', 'VegType_-seaside spurge (sandmat)', 'VegType_-unknown', 'VegType_no', 'PlantRoot_Sea Oats', 'PlantRoot_no', 'Divisions_M', 'Divisions_U', 'ZoneID_4.0', 'ZoneID_5.0', 'ZoneID_6.0', 'ZoneID_7.0', 'ZoneID_8.0', 'ZoneID_9.0', 'Lat_27.13', 'Lat_27.14', 'Lat_27.15', 'Lat_27.16', 'Long_-82.49', 'Long_-82.48', 'Long_-82.47', 'LivePip_1.0', 'LivePip_2.0', 'LivePip_3.0', 'LivePip_7.0', 'Yolkless_1.0', 'Yolkless_2.0', 'DevEnc_UH_0.1956521739130435', 'DevEnc_UH_1.0',
       'DevEnc_UH_3.0', 'DevEnc_UH_6.0', 'InvadedTotal_1.0', 'InvadedTotal_2.0', 'InvadedTotal_3.0', 'InvadedTotal_5.0', 'InvadedTotal_6.0', 'InvadedTotal_10.0', 'InvadedTotal_19.0', 'H_Invaded_5.0', 'H_Invaded_6.0', 'H_Invaded_12.0', 'UH_Invaded_0.30434782608695654', 'UH_Invaded_1.0', 'UH_Invaded_2.0', 'UH_Invaded_3.0', 'UH_Invaded_7.0', 'UH_Invaded_10.0', 'Dead_1.0', 'Dead_2.0', 'Dead_3.0', 'Dead_4.0', 'Dead_5.0', 'Dead_7.0', 'MultiVegNum_1', 'MultiVegNum_2', 'MultiVegNum_3',
       'MultiVegNum_4']

        df_c = get_cleaned_data()
        cat_cols, num_cols, cat_but_car = grab_col_names(df_c)

        if st.button('Predict'):
            new_data = pd.DataFrame([feature_list], columns=columns)
            df_c = df_c.drop("HS",axis=1)
            df_c = pd.concat([df_c, new_data], ignore_index=True)
            df_c = one_hot_encoder(df_c, cat_cols, drop_first=True)
            scaler = RobustScaler()
            num_cols.remove('HS')
            df_c[num_cols] = scaler.fit_transform(df_c[num_cols])
            last_row = df_c.iloc[-1:]
            last_row = last_row.rename(columns=dict(zip(last_row.columns, expected_names)))
            if model_option == "Linear Regression":
                model = load_model("IC_lin_reg_model.pkl")
                prediction = model.predict(last_row)
                st.write('## Result')
                st.success(f"Hatching Success Prediction: {prediction.item()}")
            elif model_option == "Decision Tree":
                model = load_model("IC_cart_model.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item()}")
            elif model_option == "Random Forest":
                model = load_model("IC_rf_model.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item()}")

    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

########################## MODEL EVALUATION ###################################
elif condition == 'Model Evaluation':
    col1, col2 = st.columns([2, 12])
    with col1:
        st.image("logo_turtle.png", width=100)
    with col2:
        st.title("Model Evaluation")
    st.markdown("""In this section, performance results of different models will be shown and visualized.""")

    st.markdown("""
            ### Performance Metrics""")
    col1, col2, col3 = st.columns([3,3, 3])
    with col1:
        st.markdown("""Root Mean Squared Error (RMSE)""")
        st.image("rmse.jpg", width=220)

    with col2:
        st.markdown("""Mean Squared Error (MSE)""")
        st.image("MSE.png", width=220)
    with col3:
        st.markdown("""R2 Score""")
        st.image("R2formula.png", width=220)

    model_option = st.selectbox(
        "Select machine learning model to see its results:",
        ("Random Forest", "Decision Tree", "Linear Regression", "SVR", "ElasticNet", "XGBRegressor")
    )

    if model_option == "Random Forest":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""Train Results:""")
            rf_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': [0.01077, 0.1037, 0.9945]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)
        with col2:
            st.markdown("""Test Results:""")
            rf_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': [0.0276, 0.1663, 0.9837]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)
        with col3:
            st.markdown("""5 fold CV + HP Tuning Results:""")
            rf_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': [0.37484, 0.0560, 0.9581]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)

    if model_option == "Decision Tree":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""Train Results:""")
            dt_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': ["2.045e-35", "4.522e-18", 1.0]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)
        with col2:
            st.markdown("""Test Results:""")
            dt_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': [0.0084, 0.0924, 0.9949]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)
        with col3:
            st.markdown("""5 fold CV + HP Tuning Results:""")
            dt_res = {
                'Metric': ["MSE", "RMSE", "R2"],
                'Result': [0.35915, 0.15984, 0.87666]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)

    if model_option == "Linear Regression": #MSE VE RMSE DEĞERLERİNİ HENÜZ YOK.
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("""Train Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0, 0, 0.999]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col2:
                st.markdown("""Test Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0, 0, 1.0]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col3:
                st.markdown("""5 fold CV + HP Tuning Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0.3867, 0.48432, 0.7738]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)

    if model_option == "SVR":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("""Train Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0.5799, 0.7615, 0.7069]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col2:
                st.markdown("""Test Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0.5454, 0.7385, 0.6802]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col3:
                st.markdown("""5 fold CV + HP Tuning Results:""")
                dt_res = {
                    'Metric': ["MSE", "RMSE", "R2"],
                    'Result': [0.1108, 0.0152, 0.9913]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)



    # Örnek veriler oluşturalım
    model_names = ['RANDOM FOREST', 'DECISION TREE', 'LINEAR REGRESSION', 'SVR']
    mse_scores = [0.05607,0.15984, 0.38674, 0.01528]
    rmse_scores = [0.37484, 0.35915, 0.48432, 0.11087]
    r2_scores = [0.95814, 0.87666, 0.77388, 0.99136]

    # DataFrame oluşturalım
    df_scores = pd.DataFrame({
        'Model': model_names,
        'MSE': mse_scores,
        'RMSE': rmse_scores,
        'R2': r2_scores
    })

    df_scores.set_index('Model', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    model_indices = range(len(df_scores.index))
    metrics = ['MSE', 'RMSE', 'R2']
    colors = [(102/255, 204/255, 102/255), (0/255, 102/255, 204/255), (255/255, 128/255, 192/255)]

    for i, metric in enumerate(metrics):
        bars = ax.bar([p + i * bar_width for p in model_indices],
               df_scores[metric],
               width=bar_width,
               color=colors[i],
               label=metric)

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performances')
    ax.set_xticks([p + 1.5 * bar_width for p in model_indices])
    ax.set_xticklabels(df_scores.index)
    ax.legend()

    st.pyplot(fig)



