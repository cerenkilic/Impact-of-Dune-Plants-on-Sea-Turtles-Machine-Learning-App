import streamlit as st
import  base64
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)

st.set_page_config(
    page_title="Impact of Dune Plants on Caretta Caretta",
    page_icon="🐢️",
    initial_sidebar_state="expanded",
)

sidebar_bg_image = "side-bar.png"


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        border-radius: 15px;
        overflow: visible;

    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_png_as_page_bg(sidebar_bg_image)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Introduction", "Plants", "EDA", "Model Prediction", "Model Evaluation", "Our Team"],
        default_index=0
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

    summary = {
    "Number of Variables": [len(df.columns)],
    "Number of Observations": [df.shape[0]],
    "Missing Cells": [df.isnull().sum().sum()],
    #"Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
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

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(dataframe[col_name].describe(quantiles).T)
    with col2:
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            dataframe[col_name].hist(bins=20, color="#4b7369")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(col_name, fontsize=14)
            st.pyplot(fig)

########################## INTRODUCTION ###################################

#Slide show fotoğrafları:
slide_images = [
    "slideshow_images/photo1.jpg",
    "slideshow_images/photo2.jpg",
    "slideshow_images/photo3.jpg",
    "slideshow_images/photo4.jpg",
    "slideshow_images/photo5.jpg",
    "slideshow_images/photo6.jpg",
    "slideshow_images/photo7.jpg",
    "slideshow_images/photo8.jpg",
    "slideshow_images/photo9.jpg",
    "slideshow_images/photo10.jpg",
    "slideshow_images/photo11.jpg",
    "slideshow_images/photo12.jpg",
    "slideshow_images/photo13.jpg",
    "slideshow_images/photo14.jpg",
]

refresh_rate = 3
# Introduction kısmı
if selected == 'Introduction':
    count = st_autorefresh(interval=refresh_rate * 1000, key="slideshow")
    col1, col2 = st.columns([2, 8])
    with col1:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=165)
    with col2:
        st.title("Impact of Dune Plants on Loggerhead Sea Turtles")
    st.write("An explanatory website for our Samsung IC capstone project.")
    st.header("Introduction")

    #SLAYT
    slide_index = count % len(slide_images)
    slide_image = Image.open(slide_images[slide_index])
    slide_image = slide_image.resize((400, 250))
    st.image(slide_image, use_column_width=True)
    st.markdown("""(The images above are generated by using Canva AI.)""")
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("invadedeggs.jpg", width=200, caption='Eggs ruined by roots [1]')
    with col2:
        st.image("ruinedeggs_hatchling.jpg", width=403, caption='Invaded eggs by roots and trapped loggerhead hatchling [1]')

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

    image3 = Image.open('sdg14.png')
    width, height = image3.size
    image3 = image3.resize((200, 200))
    st.image(image3, caption='SDG-14 Logo')

    st.markdown("""
        ### Article About the Dataset
        To learn more about the dataset you can access the dataset article (referenced as [1]) : [Link to the Article](https://onlinelibrary.wiley.com/doi/full/10.1002/ece3.11207)
        """)

######################### PLANTS ################################

elif selected == 'Plants':
    col1, col2 = st.columns([2, 7])

    with col1:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=185)
    with col2:
        # Başlık ve Açıklama
        st.markdown("<h1 style='margin-top: 40px;'>Dune Plants</h1>", unsafe_allow_html=True)

    plants_info = {
        "Beach Naupaka": {
            "images": ["plant_images/beach-naupka.jpg", "plant_images/beach-naupaka-2.jpg",
                       "plant_images/beach-naupaka-3.jpg"],
            "description": "Beach naupaka is a shrub found in coastal areas. It has white to pale yellow flowers and is known for its salt tolerance. It helps stabilize sand dunes, which is crucial for the nesting success of Caretta carettas."
        },
        "Christmas Cactus": {
            "images": ["plant_images/christmas-cactus.jpg","plant_images/christmas-cactus-2.jpg","plant_images/christmas-cactus-3.png"],
            "description": "Christmas cactus is a popular houseplant known for its beautiful flowers that bloom around Christmas time. It can grow in coastal areas and contributes to dune stabilization, indirectly supporting the nesting habitats of Caretta carettas."
        },
        "Crested Saltbush": {
            "images": ["plant_images/crested-saltbush.jpg","plant_images/crested-saltbush-2.jpg","plant_images/crested-saltbush-3.jpg"],
            "description": "Crested saltbush is a perennial shrub that grows in saline soils. It helps stabilize coastal soils, reducing erosion and providing a safer nesting ground for Caretta carettas."
        },
        "Dune Sunflower": {
            "images": ["plant_images/dune-sunflower.jpg","plant_images/dune-sunflower-2.jpg","plant_images/dune-flowers-3.jpg"],
            "description": "Dune sunflower is a flowering plant that grows in sandy soils along the coast. Its root systems help to stabilize dunes, which is essential for the nesting success of Caretta carettas."
        },
        "Palm": {
            "images": ["plant_images/palmtree.jpg","plant_images/palm-2.jpg","plant_images/palm-3.jpg"],
            "description": "Palms are a diverse group of plants that are commonly found in tropical and subtropical regions. Certain palm species can help stabilize coastal dunes, providing a supportive environment for Caretta caretta nesting."
        },
        "Railroad Vine": {
            "images": ["plant_images/railroad-vine.jpeg","plant_images/railroad-vine-2.jpg","plant_images/railroad-vine-3.jpeg"],
            "description": "Railroad vine, also known as beach morning glory, is a fast-growing vine that helps stabilize sand dunes. Its extensive root system prevents erosion, creating a more stable nesting area for Caretta carettas."
        },
        "Salt Grass": {
            "images": ["plant_images/salt-grass.jpg","plant_images/salt-grass-2.jpg","plant_images/salt-grass-3.jpeg"],
            "description": "Salt grass is a halophytic grass species that grows in saline environments. It plays a crucial role in coastal ecosystems by stabilizing soil and reducing erosion, thereby supporting the nesting success of Caretta carettas."
        },
        "Sea Grapes": {
            "images": ["plant_images/sea-grapes.jpeg","plant_images/sea-grapes-2.jpeg","plant_images/sea-grapes-3.jpeg"],
            "description": "Sea grapes are coastal plants that grow in sandy soils. They help prevent beach erosion and provide shade and protection for Caretta caretta nests."
        },
        "Sea Oats": {
            "images": ["plant_images/sea-oats.jpg","plant_images/sea-oats-2.jpeg","plant_images/sea-oats-3.jpg"],
            "description": "Sea oats are a grass species that grow on sand dunes. Their root systems help to stabilize the dunes, which is vital for providing a safe nesting habitat for Caretta carettas."
        },
        "Sea Purslane": {
            "images": ["plant_images/sea-purslane.jpg","plant_images/sea-purslane-2.jpg","plant_images/sea-purslane-3.jpg"],
            "description": "Sea purslane is a succulent plant found in coastal areas. It helps to stabilize sand and prevent erosion, providing a more secure environment for Caretta caretta nesting."
        },
        "Seaside Sandmat": {
            "images": ["plant_images/seaside-sandmat.jpg","plant_images/seaside-sandmat-2.jpg","plant_images/seaside-sandmat-3.jpg"],
            "description": "Seaside sandmat is a groundcover plant that grows in coastal regions. It helps stabilize sandy soils, reducing erosion and supporting the nesting habitats of Caretta carettas."
        }
    }

    def resize_image(image_path, output_size=(300, 200)):
        with Image.open(image_path) as img:
            resized_img = img.resize(output_size)
            return resized_img


    for plant_name, plant_data in plants_info.items():
        st.write(f"### {plant_name}")
        cols = st.columns(3)
        images = [resize_image(image) for image in plant_data["images"]]
        for col, image in zip(cols, images):
            with col:
                st.image(image)
        st.write(plant_data["description"])

    ########################## EDA ###################################

elif selected == 'EDA':
 
    col1, col2 = st.columns([1, 4])
     
    with col1:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=165)
        
    with col2:
         
        st.markdown("<h1 style='margin-top: 40px;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

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
        df_car_cols = pd.DataFrame({"Cardinal Columns": cat_but_car})
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
        df_car_cols = pd.DataFrame({"Cardinal Columns": cat_but_car})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(df_cat_cols)

        with col2:
            st.write(df_num_cols)

        with col3:
            st.write(df_car_cols)

    st.header("Scatter Plot")
    cat_cols, num_cols, cat_but_car = grab_col_names(df_c)
    x_axis = st.selectbox('Please select X axis', num_cols)
    y_axis = st.selectbox('Please select Y axis', num_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_c[x_axis], y=df_c[y_axis], ax=ax, color="#4b7369")
    plt.title(f'Scatter Plot: {x_axis} vs. {y_axis}', fontsize=15)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Summary of Numerical Features")
    selected_col = st.selectbox('Select a column', num_cols)
    num_summary(df_raw, selected_col, plot=True)


    st.header("Histogram of Categorical Features")
    selected_column = st.selectbox('Please select a categorical column', cat_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df_c[selected_column], ax=ax)
    plt.title(f'Histogram: {selected_column}', fontsize=15)
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Visualizing Missing Values")
    import missingno as msno
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.button("Matrix Rep."):
        msno.matrix(df_raw)
        st.pyplot()
    if st.button("Bar Plot Rep."):
        msno.bar(df_raw)
        st.pyplot()

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
    sns.boxplot(x=df_raw[selected_column], color="#4b7369")
    plt.title(f'Box Plot: {selected_column}', fontsize=15)
    plt.xlabel(selected_column, fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    st.header("Target Summary with Numeric Columns")

    col1, col2 = st.columns(2)

    selected_column = col1.selectbox('Please select a numerical column', num_cols, key='num_col_select')
    target_value = col2.text_input("Target", value="HS", disabled=True)

    def target_summary_with_num(dataframe, target, numerical_col):
        temp_df = dataframe.groupby(target).agg({numerical_col: "mean"})
        print(temp_df)
        fig, ax = plt.subplots(figsize=(15, 7))
        temp_df.plot(kind="bar", y=numerical_col, color="#4b7369", ax=ax)
        st.pyplot(fig)  #


    target_summary_with_num(df_raw, "HS", selected_column)

    st.header("Target Summary with VegTypes")


    def target_summary_with_vegtypes(dataframe, target):
        summary_df = dataframe.groupby('VegType').agg(
            mean_target=(target, 'mean'),
            count_target=(target, 'count')
        ).sort_values(by='mean_target')

        fig, ax = plt.subplots(figsize=(15, 7))
        summary_df['mean_target'].plot(kind='barh', color='#4b7369', ax=ax)

        for index, value in enumerate(summary_df['mean_target']):
            count = summary_df['count_target'][index]
            ax.text(value, index, f' {count}', va='center', ha='left', color='black', fontsize=10)

        plt.xlabel('Hatching Success (HS)')
        plt.ylabel('Vegetation Type (VegType)')
        plt.title('Hatching Success (HS) by Vegetation Type')
        plt.tight_layout()
        st.pyplot(fig)



    if st.checkbox("Show"):
            target_summary_with_vegtypes(df_c, "HS")


    st.header("Correlation Analysis")

    if st.checkbox("Correlation Matrix"):
        st.image("corr_map.png")
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(df_raw[num_cols].corr(), annot=True, cmap='coolwarm',
        #            cbar_kws={'label': 'Korelasyon'})
        # plt.title('Korelasyon Matrisi (Heatmap)', fontsize=15)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # st.pyplot()

########################## MODEL PREDICTION ###################################
elif selected == 'Model Prediction':
    col1, col2 = st.columns([2, 7])
    with col1:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=185)
    with col2:
        
     st.markdown("<h1 style='margin-top: 40px;'>Model Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
            This section is made to help scientists who work in related projects. """)

    def load_model(modelfile):
        loaded_model = pickle.load(open(modelfile, 'rb'))
        return loaded_model


    model_option = st.selectbox(
        "Select Machine Learning Model:",
        ("Random Forest", "Decision Tree", "ElasticNet", "SVR", "XGBoost", "Linear Regression")
    )
    html_temp = """
                <div>
                <h1 style="color:BLACK;text-align:left;"> Predictive Analytics for Scientists</h1>
                </div>
                """
    st.subheader("Please enter information about the data 🐢🌿")

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
        nestId = "99 1010 NEW ONE"
        zone = st.number_input("ZoneID", 1, 10, value=5)
        lat = st.number_input("Lat", 1.0, 50.0, value=27.14)
        long = st.number_input("Long", -100.0, 0.0, value=-82.48)
        vegpresence = st.number_input("VegPresence ", 0, 1, value=1)
        vegtype_options = ['-railroad vine', '-sea oats', 'no', '-sea purslane', "-sea grapes",
                           "-beach naupaka","-christmas cactus","-crested saltbush","-dune sunflower",
                           "-palm","-salt grass","-seaside spurge(sandmat)"]
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

        feature_list = [id, zone, lat, long, vegpresence, vegtype, rootpres, roottype, distbarr, disthighw,
                        totaldist, LocOnBeach, Division, SurfaceDepth, BottomDepth, InternalDepth,
                        CavityWidth, Hatched, Unhatched, Developed_UH, LivePip, DeadPip, Yolkless,
                        EncasedTotal, DevEnc_UH, H_Encased, UH_Encased, InvadedTotal, H_Invaded, UH_Invaded,
                        Live, Dead, Depredated, RootDamageProp, ES, TotalEggs
                        ]

        def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
            quartile1 = dataframe[col_name].quantile(q1)
            quartile3 = dataframe[col_name].quantile(q3)
            IQR_range = quartile3 - quartile1
            up_lim = quartile3 + 1.5 * IQR_range
            low_lim = quartile1 - 1.5 * IQR_range
            return low_lim, up_lim

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

        def check_outlier(dataframe, col_name):
            low, up = outlier_thresholds(dataframe, col_name)
            if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
                return True
            else:
                return False


        def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.90):
            low, up = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.90)
            dataframe.loc[(dataframe[variable] < low), variable] = low
            dataframe.loc[(dataframe[variable] > up), variable] = up
        def check_outlier(dataframe, col_name):
            low, up = outlier_thresholds(dataframe, col_name)
            if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
                return True
            else:
                return False

        def one_hot_encoder1(dataframe,
                             categorical_cols):  # bu yeni datayla farklılık olmasın diye sonra da işimize yarayacak.
            # https://drlee.io/surviving-the-one-hot-encoding-pitfall-in-data-science-62d8254cf3f6

            ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
            encoded_cols = ohe.fit_transform(dataframe[categorical_cols])
            encoded_df = pd.DataFrame(encoded_cols, index=dataframe.index)

            if hasattr(ohe, 'get_feature_names_out'):
                encoded_df.columns = ohe.get_feature_names_out(categorical_cols)
            else:
                encoded_df.columns = ohe.get_feature_names(categorical_cols)
            encoded_columns = encoded_df.columns
            dataframe = pd.concat([dataframe.drop(columns=categorical_cols), encoded_df], axis=1)
            return dataframe, encoded_columns


        def carettas_data_prep(df):
            # df = df.drop(columns=["Comments", "Notes", "Unnamed: 42", "Species", "Key", "ExDate"])
            # df = df[0:93]
            cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
            cat_cols.append("VegType")
            # missing value handling:
            for col in num_cols:
                if df[col].isnull().any():
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
            df["Divisions"].fillna(df["Divisions"].mode()[0], inplace=True)
            df["VegType"].fillna("no", inplace=True)
            df["PlantRoot"].fillna("no", inplace=True)
            # outlier handling
            for col in num_cols:
                # print(col, check_outlier(df, col))
                if check_outlier(df, col):  # eğer true ise
                    replace_with_thresholds(df, col)
            # VegType'taki çoklu bitki çeşitlerini ayrı samplelar haline getirme:
            df_aa = df.copy()

            def split_vegtype(df_aa):
                rows = []
                for _, row in df.iterrows():
                    veg_types = str(row['VegType']).split('\n') if pd.notnull(row['VegType']) else []
                    for veg in veg_types:
                        new_row = row.copy()
                        new_row['VegType'] = veg.strip()
                        rows.append(new_row)
                return pd.DataFrame(rows)

            new_df = split_vegtype(df_aa)
            multi_entries_mask = df['VegType'].str.contains('\n', na=False)
            cleaned_df = df[~multi_entries_mask]
            df = pd.concat([cleaned_df, new_df], ignore_index=True)  # merged_df --> df
            # VegType yazım hatalarını düzeltme:
            corrections = {
                " -railorad vine": "-railroad vine",
                "-railorad vine": "-railroad vine",
                " -railroad vine": "-railroad vine",
                " -sea oats on dune": "-sea oats",
                "-sea grape": "-sea grapes",
                "-beach purslane": "-sea purslane",
                "-salt grass or torpedo grass (seashore dropseed)": "-salt grass",
                "-salt grass/torpedo grass/seashore dropseed": "-salt grass",
                "-sea oats on dune": "-sea oats",
                "-crested saltbush": "-crested saltbrush",
                " -sea oats": "-sea oats"
            }
            df["VegType"] = df["VegType"].replace(corrections)
            # duplicate data handling:
            df = df.drop_duplicates(keep='last')  # biri hariç diğer tekrarlayanları sildim.
            df.reset_index(drop=True, inplace=True)
            # MultiVegNum feature oluşturma:
            group_sizes = df.groupby('NestID').size()
            df.loc[:, 'MultiVegNum'] = df['NestID'].map(group_sizes - 1)
            # vegpresence'ı 0 olup vegtype girili olanlar varsa droplamak için:
            faulty_veg = df[(df['VegPresence'] == 0) & (df['VegType'] != 'no')]
            df = df.drop(faulty_veg.index)
            df.reset_index(drop=True, inplace=True)
            # encoding
            cat_cols, num_cols, cat_but_car = grab_col_names(df)
            df = df.drop(columns=["NestID"])  # kardinal ama önceden işimize yaradığından şimdi dropladım.
            df, colnames_1h = one_hot_encoder1(df, cat_cols)
            # normalization of numeric cols
            # X_scaled = RobustScaler().fit_transform(df[num_cols])
            # df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
            y = df["HS"]
            X = df.drop(["HS"], axis=1)
            num_cols = [col for col in num_cols if col != 'HS']
            X_scaled = RobustScaler().fit_transform(X[num_cols])
            X[num_cols] = pd.DataFrame(X_scaled, columns=X[num_cols].columns)
            return X, y, df

        cat_cols, num_cols, cat_but_car = grab_col_names(df_c)

        columns = ["NestID", 'ZoneID', 'Lat', 'Long', 'VegPresence', 'VegType', 'RootPresence', 'PlantRoot',
                   'DistBarrier',
                   'DistHighWater', 'TotalDist', 'LocOnBeach', 'Divisions', 'SurfaceDepth', 'BottomDepth',
                   'InternalDepth', 'CavityWidth', 'Hatched', 'Unhatched', 'Developed_UH', 'LivePip', 'DeadPip',
                   'Yolkless', 'EncasedTotal', 'DevEnc_UH', 'H_Encased', 'UH_Encased', 'InvadedTotal', 'H_Invaded',
                   'UH_Invaded', 'Live', 'Dead', 'Depredated', 'RootDamageProp', 'ES', 'TotalEggs']

        expected_org_names = ['DistBarrier', 'DistHighWater', 'TotalDist', 'LocOnBeach',
                              'SurfaceDepth', 'BottomDepth', 'InternalDepth', 'CavityWidth',
                              'Hatched', 'Unhatched', 'Developed_UH', 'DeadPip', 'EncasedTotal',
                              'H_Encased', 'UH_Encased', 'Live', 'RootDamageProp', 'ES', 'TotalEggs',
                              'VegType_-christmas cactus', 'VegType_-crested saltbrush',
                              'VegType_-dune sunflower', 'VegType_-palm', 'VegType_-railroad vine',
                              'VegType_-salt grass', 'VegType_-sea grapes', 'VegType_-sea oats',
                              'VegType_-sea purslane', 'VegType_-seaside spurge (sandmat)',
                              'VegType_-unknown', 'VegType_no', 'PlantRoot_Sea Oats', 'PlantRoot_no',
                              'Divisions_M', 'Divisions_U', 'ZoneID_4.0', 'ZoneID_5.0', 'ZoneID_6.0',
                              'ZoneID_7.0', 'ZoneID_8.0', 'ZoneID_9.0', 'Lat_27.13', 'Lat_27.14',
                              'Lat_27.15', 'Lat_27.16', 'Long_-82.49', 'Long_-82.48', 'Long_-82.47',
                              'VegPresence_1.0', 'RootPresence_1.0', 'LivePip_1.0', 'LivePip_2.0',
                              'LivePip_3.0', 'LivePip_7.0', 'Yolkless_1.0', 'Yolkless_2.0',
                              'DevEnc_UH_1.0', 'DevEnc_UH_3.0', 'DevEnc_UH_6.0', 'DevEnc_UH_nan',
                              'InvadedTotal_1.0', 'InvadedTotal_2.0', 'InvadedTotal_3.0',
                              'InvadedTotal_5.0', 'InvadedTotal_6.0', 'InvadedTotal_10.0',
                              'InvadedTotal_19.0', 'H_Invaded_5.0', 'H_Invaded_6.0', 'H_Invaded_12.0',
                              'UH_Invaded_1.0', 'UH_Invaded_2.0', 'UH_Invaded_3.0', 'UH_Invaded_7.0',
                              'UH_Invaded_10.0', 'UH_Invaded_nan', 'Dead_1.0', 'Dead_2.0', 'Dead_3.0',
                              'Dead_4.0', 'Dead_5.0', 'Dead_7.0', 'Depredated_7.0', 'Depredated_13.0',
                              'Depredated_26.0', 'Depredated_31.0', 'Depredated_36.0',
                              'Depredated_38.0', 'Depredated_82.49999999999989', 'MultiVegNum_1',
                              'MultiVegNum_2', 'MultiVegNum_3', 'MultiVegNum_4']

        cat_cols, num_cols, cat_but_car = grab_col_names(df_c)
        new_data = pd.DataFrame([feature_list], columns=columns)
        df = pd.read_csv("ReddingRootsCaseStudy22_csv.csv")
        df = df.drop(columns=["Comments", "Notes", "Unnamed: 42", "Species", "Key", "ExDate"])
        df = df[0:93]
        df = pd.concat([df, new_data], ignore_index=True)
        X, y, df = carettas_data_prep(df)
        df_lr = df.drop("HS", axis=1)
        last_row = X.iloc[-1:]
        df_reset = last_row.reset_index(drop=True)
        new_col_names = last_row.columns
        last_row = last_row.reindex(columns=expected_org_names, fill_value=0)

        if st.button('Predict'):
            if model_option == "Linear Regression":
                model = load_model("model_files/LR_model_050724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            elif model_option == "Decision Tree":
                model = load_model("model_files/nnCART_model_080724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            elif model_option == "Random Forest":
                model = load_model("model_files/nnRF_model_080724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            elif model_option == "ElasticNet":
                model = load_model("model_files/nnENet_model_080724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            elif model_option == "SVR":
                model = load_model("model_files/nnSVR_model_080724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            elif model_option == "XGBoost":
                model = load_model("model_files/nnXGBRegressor_model_080724.pkl")
                prediction = model.predict(last_row)
                st.write('## Result:')
                st.success(f"Hatching Success Prediction: {prediction.item():.2f}%")
            progress = int(prediction)
            st.progress(progress)

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
elif selected == 'Model Evaluation':
    col1, col2 = st.columns([2, 7])
    with col1:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=185)
    with col2:
         
        st.markdown("<h1 style='margin-top: 40px;'>Model Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("""In this section, performance results of different models will be shown and visualized.""")

    st.markdown("""
            ### Performance Metrics""")
    col1, col2, col3 = st.columns([3,3, 3])
    with col1:
        st.markdown("""Mean Absolute Error (MAE)""")
        st.image("MAE.png", width=220)

    with col2:
        st.markdown("""Mean Squared Error (MSE)""")
        st.image("MSE.png", width=220)
    with col3:
        st.markdown("""R2 Score""")
        st.image("r2.png", width=220)

    model_option = st.selectbox(
        "Select machine learning model to see its results:",
        ("Random Forest", "Decision Tree", "Linear Regression", "SVR", "ElasticNet", "XGBRegressor")
    )

    if model_option == "Random Forest":
        st.markdown("""### Random Forest Model Results:""")
        st.markdown("""(5 fold cross validation is applied.)""")
        st.markdown("""RFE: Recursive Feature Elimination""")

        col1, col2, col3 = st.columns([1, 1,1])
        with col1:
            st.markdown("""Before HP Tuning:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [23.3109, 3.1591, 0.9573]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)
        with col2:
            st.markdown("""After HP Tuning:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [20.9452, 3.0095, 0.95737]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)

        with col3:
            st.markdown("""RFE Results:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [19.6486, 2.9414, 0.9583]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)

        st.image("rfe_images/RF_rfecv_visualization.png",width=700)

    if model_option == "Decision Tree":
        st.markdown("""### Decision Tree Model Results:""")
        st.markdown("""(5 fold cross validation is applied to obtain all the results.)""")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""Before HP Tuning:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [39.7303, 4.4694, 0.8887]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)
        with col2:
            st.markdown("""After HP Tuning:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [28.5947, 3.6541, 0.9311]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)

        with col3:
            st.markdown("""RFE Results:""")
            rf_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [28.8450, 3.6859, 0.9309]
            }
            rfres1 = pd.DataFrame(rf_res)
            st.write(rfres1)

        st.image("rfe_images/CART_rfecv_visualization.png", width=700)

    if model_option == "Linear Regression":
            st.markdown("""### Linear Regression Model Results:""")
            st.markdown("""(5 fold cross validation is applied to obtain all the results.)""")
            st.markdown("""RFE: Recursive Feature Elimination""")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("""Before HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [952.1115, 9.2696, -0.9017]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col2:
                st.markdown("""After HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [952.1115, 9.2696, 0.9017]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col3:
                st.markdown("""RFE Results:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [206.8268, 1.8211, 0.5705]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)

            st.image("rfe_images/LR_rfecv_visualization.png", width=700)

    if model_option == "SVR":
        st.markdown("""### Support Vector Machine (Regressor) Model Results:""")
        st.markdown("""(5 fold cross validation is applied to obtain all the results.)""")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""Before HP Tuning:""")
            dt_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [732.7053, 16.6085, -0.0998]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)
        with col2:
            st.markdown("""After HP Tuning:""")
            dt_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [3.1643, 1.1800, 0.9937]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)
        with col3:
            st.markdown("""RFE Results:""")
            dt_res = {
                'Metric': ["MSE", "MAE", "R2"],
                'Result': [2.0679, 0.7905, 0.9961]
            }
            dtres1 = pd.DataFrame(dt_res)
            st.write(dtres1)

    if model_option == "ElasticNet":
            st.markdown("""### ElasticNet  Model Results:""")
            st.markdown("""(5 fold cross validation is applied to obtain all the results.)""")
            st.markdown("""RFE: Recursive Feature Elimination""")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("""Before HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [105.0971, 6.3360, 0.8554]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col2:
                st.markdown("""After HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [1.3037, 0.7081, 0.9974]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col3:
                st.markdown("""RFE Results:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [1.0494, 0.619, 0.9978]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)

            st.image("rfe_images/ENet_rfecv_visualization.png", width=700)

    if model_option == "XGBRegressor":
            st.markdown("""### XGBoost (Regressor)  Model Results:""")
            st.markdown("""(5 fold cross validation is applied to obtain all the results.)""")
            st.markdown("""RFE: Recursive Feature Elimination""")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("""Before HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [59.9960, 4.0882, 0.8926]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col2:
                st.markdown("""After HP Tuning:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [31.9786, 3.4557, 0.9557]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)
            with col3:
                st.markdown("""RFE Results:""")
                dt_res = {
                    'Metric': ["MSE", "MAE", "R2"],
                    'Result': [50.3124, 4.0194, 0.9194]
                }
                dtres1 = pd.DataFrame(dt_res)
                st.write(dtres1)

            st.image("rfe_images/XGBRegressor_rfecv_visualization.png", width=700)

    #model performances grafiği:
    model_names = ['RANDOM FOREST', 'DECISION TREE','SVR',"ENET","XGBOOST"]
    mse_scores = [20.945, 28.594, 3.1643, 1.049, 31.978]
    mae_scores = [3.009, 3.654, 1.18, 0.619, 3.455]
    r2_scores = [0.9573, 0.9311, 0.993, 0.9978, 0.9557]

    df_scores = pd.DataFrame({
        'Model': model_names,
        'MSE': mse_scores,
        'MAE': mae_scores,
        'R2': r2_scores
    })

    df_scores.set_index('Model', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    model_indices = range(len(df_scores.index))
    metrics = ['MSE', 'MAE', 'R2']
    colors = ['#f5a454', '#94bea3', '#bce6ec']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_scores.index, df_scores['MSE'], color=colors[0])
    ax.set_xlabel('Models')
    ax.set_ylabel('MSE Scores')
    ax.set_title('Model MSE Performances')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_scores.index, df_scores['MAE'], color=colors[1])
    ax.set_xlabel('Models')
    ax.set_ylabel('MAE Scores')
    ax.set_title('Model MAE Performances')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df_scores.index, df_scores['R2'], color=colors[2])
    ax.set_xlabel('Models')
    ax.set_ylabel('R2 Scores')
    ax.set_title('Model R2 Performances')
    st.pyplot(fig)

##########################OUR TEAM ###################################
elif selected == 'Our Team':
    count = st_autorefresh(interval=refresh_rate * 1000, key="our-team-slideshow")
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        """,
        unsafe_allow_html=True
    )
    image_col, title_col = st.columns([1, 3])

    with image_col:
        st.image("CarettaCarettaTurtle-Photoroom.png", width=185)

    with title_col:
        st.markdown("<h1 style='text-align: left; margin-top: 50px;'>Our Team</h1>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("our_team/mihriban-ozdemir.jpeg", width=200)
        st.markdown(
            "<b>Mihriban Özdemir</b>&nbsp;&nbsp;<a href='www.linkedin.com/in/mihribanozdemir' class='fa fa-linkedin'></a>&nbsp;<a href='https://github.com/mihribanozdemir' class='fa fa-github'></a>",
            unsafe_allow_html=True)

    with col2:
        st.image("our_team/ceren-kilic.jpeg", width=200)
        st.markdown(
            "<b>Ceren Kılıç</b>&nbsp;&nbsp;<a href='https://www.linkedin.com/in/cernkilic/' class='fa fa-linkedin'></a>&nbsp;<a href='https://github.com/cerenkilic' class='fa fa-github'></a>",
            unsafe_allow_html=True)

    with col3:
        st.image("our_team/turkan-risvan.jpeg", width=200)
        st.markdown(
            "<b>Türkan Rişvan</b>&nbsp;&nbsp;<a href='https://www.linkedin.com/in/t%C3%BCrkan-ri%C5%9Fvan/' class='fa fa-linkedin'></a>&nbsp;<a href='https://github.com/turkan-risvan' class='fa fa-github'></a>",
            unsafe_allow_html=True)

    our_team_slide_images = [
        "our-team-slideshow/1.jpg",
        "our-team-slideshow/2.jpg",
        "our-team-slideshow/3.jpg",
        "our-team-slideshow/4.jpg",
    ]

    slide_index = count % len(our_team_slide_images)
    slide_image = Image.open(our_team_slide_images[slide_index])
    slide_image = slide_image.resize((700, 400))
    st.image(slide_image, use_column_width=True)



