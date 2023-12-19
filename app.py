#importing all the important libraries
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Opsi Menu",["Information Page","Dataset","Visualization","Diabetes Predict",])
 
if rad=="Information Page":
    st.title("Diabetes Home Page")
    st.write("Muhammad Rafi & Nyeto Leo Tsuli")
    st.image("debet.jpg")
    st.write("Diabetes merupakan kondisi medis yang terjadi ketika kadar gula darah seseorang menjadi terlalu tinggi. Hal ini dapat terjadi karena tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif. Insulin adalah hormon yang diproduksi oleh pankreas yang membantu mengatur kadar gula darah. Diabetes terbagi menjadi dua jenis utama: tipe 1 dan tipe 2.")    
    st.write("- **Tipe 1**: Biasanya muncul pada masa anak-anak atau remaja di mana tubuh gagal memproduksi insulin. Penderita tipe 1 diabetes membutuhkan insulin sintetis secara teratur untuk mengatur kadar gula darah mereka.")   
    st.write("- **Tipe 2**: Lebih umum terjadi dan berkaitan dengan gaya hidup serta faktor genetik. Pada tipe 2, tubuh tidak menggunakan insulin dengan efektif, atau tidak memproduksi cukup insulin untuk mempertahankan kadar gula darah yang sehat.")
    st.write("Di Indonesia, diabetes telah menjadi masalah kesehatan masyarakat yang signifikan. Menurut data dari Kementerian Kesehatan Indonesia, prevalensi diabetes di Indonesia telah meningkat secara signifikan dalam beberapa tahun terakhir. Faktor-faktor seperti pola makan yang tidak sehat, kurangnya aktivitas fisik, dan peningkatan obesitas telah berkontribusi pada peningkatan kejadian diabetes. Organisasi Kesehatan Dunia (WHO) juga mencatat bahwa pencegahan dan pengelolaan diabetes adalah salah satu tantangan kesehatan utama di Indonesia, dan edukasi serta perubahan gaya hidup menjadi penting dalam mengendalikan laju penyebarannya.")
    st.image("gejala diabetes.jpg")
    st.title("Gejala dan Fakta Terkait Diabetes")

    st.write(
        """
        Diabetes mellitus dapat memiliki sejumlah gejala yang umumnya menjadi ciri-ciri penyakit ini. Beberapa gejala yang sering terkait dengan diabetes antara lain:

        ### Gejala Umum Diabetes:
        1. **Peningkatan rasa haus dan sering buang air kecil**: Penderita diabetes sering merasa haus berlebihan dan mengalami peningkatan frekuensi buang air kecil, terutama di malam hari.
        2. **Penurunan berat badan yang tidak disengaja**: Meskipun sering makan, penderita diabetes dapat mengalami penurunan berat badan yang tidak wajar.
        3. **Rasa lapar yang berlebihan**: Terutama setelah makan, penderita diabetes masih merasa lapar.
        4. **Kelelahan dan lemah**: Penderita diabetes sering merasa lelah dan lemah, bahkan setelah beristirahat yang cukup.
        5. **Kulit kering dan gatal**: Kulit yang kering dan gatal dapat menjadi gejala diabetes.
        6. **Luka sulit sembuh**: Luka atau luka kecil yang sulit sembuh atau menyembuh dengan lambat dapat menjadi tanda diabetes.
        7. **Penglihatan kabur**: Perubahan tiba-tiba dalam penglihatan atau penglihatan yang kabur dapat terkait dengan diabetes.

        ### Hubungan dengan BMI (Body Mass Index):
        BMI adalah ukuran standar yang digunakan untuk mengevaluasi berat badan seseorang berdasarkan tinggi badan. Orang dengan indeks massa tubuh (BMI) yang tinggi cenderung memiliki risiko lebih tinggi untuk mengembangkan diabetes tipe 2. Kaitannya dengan diabetes sering kali terjadi pada orang yang mengalami obesitas atau kelebihan berat badan.

        ### Fakta Terkait BMI dan Diabetes:
        1. **Obesitas Meningkatkan Risiko Diabetes**: Orang dengan BMI tinggi atau obesitas memiliki risiko lebih tinggi terkena diabetes tipe 2.
        2. **Pola Makan Sehat dan Aktivitas Fisik Penting**: Menjaga berat badan sehat dengan mengadopsi pola makan yang sehat dan berolahraga secara teratur dapat membantu mengurangi risiko terkena diabetes.

        Penting untuk diingat bahwa gejala diabetes dapat bervariasi dan tidak semua orang dengan diabetes akan mengalami gejala yang sama. Jika Anda atau seseorang yang Anda kenal mengalami gejala yang dicurigai terkait diabetes, penting untuk berkonsultasi dengan profesional medis untuk diagnosis dan perawatan yang tepat.
        """
    )

#Dataset
if rad=="Dataset":
    st.title("Data Universal yang di Test Diabetes")
    st.subheader("Total 767 Kasus yang teridentifikasi")
    df = pd.read_csv("diabetes_finaledit.csv")
    st.write(df)
    st.write(df.describe())
    glucose_zero = round((len(df[df['Glucose']==0]) / len(df)) * 100, 2)
    bloodpressure_zero = round((len(df[df['BloodPressure']==0]) / len(df)) * 100, 2)
    skinthickness_zero = round((len(df[df['SkinThickness']==0]) / len(df)) * 100, 2)
    insulin_zero = round((len(df[df['Insulin']==0]) / len(df)) * 100, 2)
    bmi_zero = round((len(df[df['BMI']==0]) / len(df)) * 100, 2)
    st.write(f'{glucose_zero}% of the data in the dataset has 0 in Glucose feature')
    st.write(f'{bloodpressure_zero}% of the data in the dataset has 0 in BloodPressure feature')
    st.write(f'{skinthickness_zero}% of the data in the dataset has 0 in SkinThickness feature')
    st.write(f'{insulin_zero}% of the data in the dataset has 0 in Insulin feature')
    st.write(f'{bmi_zero}% of the data in the dataset has 0 in BMI feature')

#Visualization
if rad=="Visualization":
    df = pd.read_csv("diabetes_finaledit.csv")
    st.title("Visualization")
    selected_option = st.radio("Choose an option:", ['Histogram', 'CDF Plot', 'Heatmap'])
    
    if selected_option == 'Histogram':
        # it is physiologically impossible the anthropometry and physiological measurements to be `0`
        # replaced the value zero in these features with Nan value
        df_copy = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

        # insert the unaffected column to the dataframe `df_copy`
        df_copy['Pregnancies'] = df['Pregnancies']
        df_copy['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction']
        df_copy['Age'] = df['Age']
        df_copy['Outcome'] = df['Outcome']
        # use KNNImputer to impute the Nan value
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=2)
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df_copy.columns)
        # distribution of label `Outcome`
        # convert the values in label `Outcome`
        # 1 for `Yes` and 0 for `No` --- diabetes
        imputed_df['Outcome'] = ['Yes' if i == 1 else 'No' for i in imputed_df['Outcome']]
        # visualization for the distribution
        figure, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]})
        sns.countplot(data=imputed_df, x='Outcome', ax=axes[0])
        axes[0].set_title('Distribution of Diabetes Outcome')
        axes[0].grid(alpha=0.4)
        axes[1].pie(imputed_df['Outcome'].value_counts().sort_values(), labels=imputed_df['Outcome'].unique(),
                    autopct='%1.1f%%', explode=[0.1, 0])
        axes[1].set_title('Proportion of Outcome')
        plt.tight_layout()
        st.pyplot(figure)

        st.title("Histogram")
        st.subheader("Pengaruh Dari Glukosa")
        st.write(pd.DataFrame(imputed_df['Glucose'].describe()).transpose())
        # Visualization: Distribution of feature 'Glucose'
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
        sns.histplot(data=imputed_df, x='Glucose', ax=axes[0], color='darkblue')
        axes[0].set_title('Histogram: Distribution of Glucose Level')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Glucose', ax=axes[1], color='darkblue')
        axes[1].set_title('Boxplot: Range of Glucose Level')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        #Tekanan Darah
        st.subheader("Pengaruh Dari Tekanan Darah")
        st.write(pd.DataFrame(imputed_df['BloodPressure'].describe()).transpose())
        sns.histplot(data=imputed_df, x='BloodPressure', ax=axes[0], color='darkblue')
        axes[0].set_title('Histogram: Distribution of Blood Pressure')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='BloodPressure', ax=axes[1], color='darkblue')
        axes[1].set_title('Boxplot: Range of Blood Pressure')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)
        # potential outliers in feature `BloodPressure`
        # calculation of interquantile range, IQR
        Q3 = imputed_df['BloodPressure'].quantile(0.75)
        Q1 = imputed_df['BloodPressure'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper and lower bound for the `BloodPressure`
        upper_bound = Q3 + (IQR*1.5)
        lower_bound = Q1 - (IQR*1.5)

        # subset the data where the `Blood Pressure` of the instances/rows are potentially an outliers
        glucose_outliers = imputed_df[(imputed_df['BloodPressure']>upper_bound) | (imputed_df['BloodPressure']<lower_bound)]

        # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(glucose_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(glucose_outliers)} data yang mewakili sekitar {outliers_percent}% dari outlier data tekanan darah.')

        #Ketebalan Kulit
        st.subheader("Pengaruh Dari Ketebalan Kulit")
        st.write(pd.DataFrame(imputed_df['SkinThickness'].describe()).transpose())

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
        sns.histplot(data=imputed_df, x='SkinThickness', color='darkblue', ax=axes[0])
        axes[0].set_title('Histogram: Distribution of Skin Thickness')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='SkinThickness', color='darkblue', ax=axes[1])
        axes[1].set_title('Boxplot: Range of Skin Thickness')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        # potential outliers in feature `SkinThickness`
        # find the interquantile range, IQR
        Q3 = imputed_df['SkinThickness'].quantile(0.75)
        Q1 = imputed_df['SkinThickness'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper bound and lower bound
        upper_bound = Q3 + (1.5*IQR)
        lower_bound = Q1 - (1.5*IQR)

        # subset the data where the `SkinThickness` of the instances/rows that are potentially an outliers
        skinthickness_outliers = imputed_df[(imputed_df['SkinThickness']>upper_bound) | (imputed_df['SkinThickness']<lower_bound)]


        # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(skinthickness_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(skinthickness_outliers)} data yang mewakili {outliers_percent}% Dari outlier data Ketebalan Kulit.')

        #Insulin
        st.subheader("Pengaruh Dari Insulin ")
        st.write(pd.DataFrame(imputed_df['Insulin'].describe()).transpose())

        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
        sns.histplot(data=imputed_df, x='Insulin', color='darkblue', ax=axes[0])
        axes[0].set_title('Histogram: Distribution of Insulin')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Insulin', color='darkblue', ax=axes[1])
        axes[1].set_title('Boxplot: Range of Insulin')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        # find the interquantile range, IQR
        Q3 = imputed_df['Insulin'].quantile(0.75)
        Q1 = imputed_df['Insulin'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper bound and lower bound
        upper_bound = Q3 + (1.5*IQR)
        lower_bound = Q1 - (1.5*IQR)

        # subset the data where the `Insulin` of the instances/rows that are potentially an outliers
        insulin_outliers = imputed_df[(imputed_df['Insulin']>upper_bound) | (imputed_df['Insulin']<lower_bound)]

    # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(insulin_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(insulin_outliers)} dari {outliers_percent}% data dari outlier Insulin.')

        #BMI
        st.subheader("Pengaruh Dari BMI (Indeks Massa Tubuh)")
        # distribution of feature `BMI`
        # summary stats of `BMI`
        st.write(pd.DataFrame(imputed_df['BMI'].describe()).transpose())
        # visualization - distribution of `BMI`
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
        sns.histplot(data=imputed_df, x='BMI', color='darkblue', ax=axes[0])
        axes[0].set_title('Histogram: Distribution of BMI')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='BMI', color='darkblue', ax=axes[1])
        axes[1].set_title('Boxplot: Range of BMI')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        Q3 = imputed_df['BMI'].quantile(0.75)
        Q1 = imputed_df['BMI'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper bound and lower bound
        upper_bound = Q3 + (1.5*IQR)
        lower_bound = Q1 - (1.5*IQR)

        # subset the data where the `BMI` of the instances/rows that are potentially an outliers
        bmi_outliers = imputed_df[(imputed_df['BMI']>upper_bound) | (imputed_df['BMI']<lower_bound)]
        # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(bmi_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(bmi_outliers)} dari {outliers_percent}% data outliers pada BMI.')

        #Silsilah Keluarga (Pedigree)
        st.subheader("Pengaruh Dari Pedigree (Keturunan Diabetes)")
        # distribution of feature DiabetesPedigreeFunction`
        # summary stats of `DiabetesPedigreeFunction`
        st.write(pd.DataFrame(imputed_df['DiabetesPedigreeFunction'].describe()).transpose())

        # visualization - distribution of `DiabetesPedigreeFunction`
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
        sns.histplot(data=imputed_df, x='DiabetesPedigreeFunction', color='darkblue', ax=axes[0])
        axes[0].set_title('Histogram: Distribution of Diabetes Pedigree Function')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='DiabetesPedigreeFunction', color='darkblue', ax=axes[1])
        axes[1].set_title('Boxplot: Range of Diabetes Pedigree Function')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        # potential outliers in feature `DiabetesPedigreeFunction`
        # find the interquantile range, IQR
        Q3 = imputed_df['DiabetesPedigreeFunction'].quantile(0.75)
        Q1 = imputed_df['DiabetesPedigreeFunction'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper bound and lower bound
        upper_bound = Q3 + (1.5*IQR)
        lower_bound = Q1 - (1.5*IQR)

        # subset the data where the `DiabetesPedigreeFunction` of the instances/rows that are potentially an outliers
        dpf_outliers = imputed_df[(imputed_df['DiabetesPedigreeFunction']>upper_bound) | (imputed_df['DiabetesPedigreeFunction']<lower_bound)]

        # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(dpf_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(dpf_outliers)} data dari {outliers_percent}% outlier pada Pedigree / Silsilah Keluarga.')

        #Berdasarkan Umur
        st.subheader("Pengaruh Dari Usia")
        # distribution of feature `Age`
        # summary stats for `Age`
        st.write(pd.DataFrame(imputed_df['Age'].describe()).transpose())

        # visualization: distribution of `Age`
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[2,1]})
        sns.histplot(data=imputed_df, x='Age', ax=axes[0], color='darkblue')
        axes[0].set_title('Histogram: Distribution of Age')
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Age', ax=axes[1], color='darkblue')
        axes[1].set_title('Boxplot: Range of Age')
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        # potential outliers in feature `Age`
        # calculation of interquantile range, IQR
        Q3 = imputed_df['Age'].quantile(0.75)
        Q1 = imputed_df['Age'].quantile(0.25)
        IQR = Q3 - Q1

        # define the upper and lower bound for the `Age`
        upper_bound = Q3 + IQR*1.5
        lower_bound = Q1 - IQR*1.5

        # subset the data where the `Age` of the instances/rows are potentially an outliers
        age_outliers = imputed_df[(imputed_df['Age']>upper_bound) | (imputed_df['Age']<lower_bound)]

        # percentage of outliers for feature `BloodPressure`
        outliers_percent = round((len(age_outliers) / len(imputed_df)) * 100, 2)
        st.write(f'Terdapat {len(age_outliers)} dari {outliers_percent}% data outlier pada Usia.')

        #Kehamilan
        st.subheader("Pengaruh Dari Kehamilan")
    # distribution of feature `Pregnancies`
        # convert the data type of `Pregnancies` to integer
        imputed_df['Pregnancies'] = imputed_df['Pregnancies'].astype(int)

        # visualization: disrtibution of `Pregnancies`
        plt.figure(figsize=(8,5))
        sns.countplot(data=imputed_df, x='Pregnancies')
        plt.title('Distribution of Pregnancies')
        plt.grid(alpha=0.4)

        pregnancies_count = imputed_df['Pregnancies'].value_counts()
        for index, count in enumerate(pregnancies_count):
            plt.text(index, count, str(count), ha='center', va='bottom')

        st.pyplot(figure)

    if selected_option == 'CDF Plot':
        df_copy = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

        # insert the unaffected column to the dataframe `df_copy`
        df_copy['Pregnancies'] = df['Pregnancies']
        df_copy['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction']
        df_copy['Age'] = df['Age']
        df_copy['Outcome'] = df['Outcome']
        # use KNNImputer to impute the Nan value
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=2)
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df_copy.columns)
        imputed_df['Outcome'] = ['Yes' if i == 1 else 'No' for i in imputed_df['Outcome']]
        #CDF PLOT
        st.title("Cumulative Distribution Function ")
        st.subheader("Statistik Diabetes yang Terkena dan Tidak")
        # summary stats for both groups
        st.write(imputed_df.groupby('Outcome').Glucose.describe())
        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='Glucose', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Glucose for Diabetic & Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Glucose', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Glucose Level in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')
        groups.get_group('Yes')

        res = mannwhitneyu(groups.get_group('Yes').Glucose, groups.get_group('No').Glucose)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean glucose level between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean glucose level between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean blood pressure for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan Tekanan Darah")
        st.write(imputed_df.groupby('Outcome').BloodPressure.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='BloodPressure', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Blood Pressure for Diabetic & Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='BloodPressure', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Blood Pressure in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').BloodPressure, groups.get_group('No').BloodPressure)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean blood pressure between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean blood pressure between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean skin thickness for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan Ketebalan Kulit")
        st.write(imputed_df.groupby('Outcome').SkinThickness.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='SkinThickness', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Skin Thickness for Diabetic & Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='SkinThickness', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Skin Thickness in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').SkinThickness, groups.get_group('No').SkinThickness)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean skin thickness between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean skin thickness between diabetic and non-diabetic groups. ')

        st.subheader("Berdasarkan Insulin")
        # Question: Is there a statistical significant difference of mean insulin level for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.write(imputed_df.groupby('Outcome').Insulin.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='Insulin', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Insulin Level for Diabetic & Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Insulin', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Insulin in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').Insulin, groups.get_group('No').Insulin)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean insulin level between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean insulin level between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean BMI for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan BMI (Indeks Massa Badan)")
        st.write(imputed_df.groupby('Outcome').BMI.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='BMI', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of BMI for Diabetic & Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='BMI', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of BMI in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').BMI, groups.get_group('No').BMI)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean BMI between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean BMI between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean Pregnancies for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan Kehamilan")
        st.write(imputed_df.groupby('Outcome').Pregnancies.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='Pregnancies', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Pregnancies for Diabetic and Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Pregnancies', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Pregnancies in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').Pregnancies, groups.get_group('No').Pregnancies)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean Pregnancies between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean Pregnancies between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean Diabetes Pedigree Function for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan Pedigree/Keturunan Diabetes")
        st.write(imputed_df.groupby('Outcome').DiabetesPedigreeFunction.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='DiabetesPedigreeFunction', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Diabetes Pedigree Function for Diabetic and Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='DiabetesPedigreeFunction', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Diabetes Pedigree Function in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').DiabetesPedigreeFunction, groups.get_group('No').DiabetesPedigreeFunction)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
            print('Reject null hypothesis. There is a statistical significant difference of mean Diabetes Pedigree Function between diabetic and non-diabetic groups.')
        else:
            print('Insufficient evidence to conclude a statistical significant difference of mean Diabetes Pedigree Function between diabetic and non-diabetic groups. ')

        # Question: Is there a statistical significant difference of mean age for diabetic and non-diabetic groups?
        # summary stats for both groups
        st.subheader("Berdasarkan Umur")
        st.write(imputed_df.groupby('Outcome').Age.describe())

        # visualization - Empirical CDF plot & boxplot
        figure, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.5,1]})
        sns.ecdfplot(data=imputed_df, x='Age', hue='Outcome', ax=axes[0])
        axes[0].set_title('Empirical CDF of Age for Diabetic and Non-diabetic', fontsize=10)
        axes[0].grid(alpha=0.4)
        sns.boxplot(data=imputed_df, y='Age', x='Outcome', ax=axes[1])
        axes[1].set_title('Distribution of Age in Diabetic & Non-diabetic Groups', fontsize=10)
        axes[1].grid(alpha=0.4)
        plt.tight_layout()
        st.pyplot(figure)

    if selected_option == 'Heatmap':
        st.title("Korelasi Antar Numerik")
        df_copy = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

         # insert the unaffected column to the dataframe `df_copy`
        df_copy['Pregnancies'] = df['Pregnancies']
        df_copy['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction']
        df_copy['Age'] = df['Age']
        df_copy['Outcome'] = df['Outcome']
        # use KNNImputer to impute the Nan value
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=2)
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df_copy.columns)
        imputed_df['Outcome'] = ['Yes' if i == 1 else 'No' for i in imputed_df['Outcome']]
        

        from scipy.stats import mannwhitneyu

        groups = imputed_df.groupby('Outcome')

        res = mannwhitneyu(groups.get_group('Yes').Age, groups.get_group('No').Age)
        print('Statistics:', res.statistic)
        print('p value:', res.pvalue)

        if res.pvalue < 0.05:
                print('Reject null hypothesis. There is a statistical significant difference of mean age between diabetic and non-diabetic groups.')
        else:
                print('Insufficient evidence to conclude a statistical significant difference of mean age between diabetic and non-diabetic groups. ')

        # Calculate correlation matrix
        correlation = imputed_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pregnancies', 'DiabetesPedigreeFunction', 'Age']].corr()
        # Create heatmap using Seaborn
        heatmap = sns.heatmap(correlation, vmin=-1, vmax=1, cmap='plasma', annot=True)
        # Display the heatmap using st.pyplot()
        st.pyplot(heatmap.figure)
        #Diabetes Prediction

#loading the Diabetes dataset
df2=pd.read_csv("Diabetes Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#Diabetes Page

#heading over to the Heart Disease section
if rad=="Diabetes Predict":
    st.header("Pengecekkan Diabetes Menggunakan Predict")
    st.write("Nilai jawaban harus sesuai.")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Masukkan Kadar Glukosa Tubuh (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Masukkan Kadar Level Insulin Dalam Tubuh (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Masukkan BMI (Indeks Massa Badan) (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Masukkan Umur (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are st.writeed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("Memungkinkan Kamu terkena diabetes")
        elif prediction2==0:
            st.success("Kamu Aman dari diabetes")