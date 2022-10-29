import streamlit as st
import scipy.stats as stats
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.metrics import roc_curve,auc

st.title("Adaptive Machine Learning")

from PIL import Image 

st.subheader('Automated Statistical Tests & Feature Selection')


col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')

with col2:
    st.image("th__30_-removebg-preview.png")


with col3:
    st.write(' ')

import pandas as pd
import numpy as np 

upload_file=st.file_uploader("Upload a csv File",type='csv')
if upload_file is not None:
	df=pd.read_csv(upload_file)
	st.write("Check Your DataSet Below\n")
	st.write(df)
	st.write("Shape of Your DataSet",df.shape)
else: 
	st.error("Please Upload a CSV File")


target=st.selectbox('Select Target Variable',df.columns)
st.write("Your Target Variable is",target)
X=df.drop(target,axis=1)
Y=df[target]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,
                        strategy ='median')
imputer = imputer.fit(df[['MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']])
df_new=pd.DataFrame(imputer.transform(df[['MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]))

df_new.columns=['MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']

df['REASON']=df['REASON'].fillna(0)
df['JOB']=df['JOB'].fillna(0)

df_new['REASON']=['DebtCon' if row==0 else row for row in df['REASON']]
df_new['JOB']=['Other' if row==0 else row for row in df['JOB']]

df_new[['BAD','LOAN']]=df[['BAD','LOAN']]

df_dummy=pd.get_dummies(df_new)

col1, col2, col3= st.columns(3)
with col1:
    st.write(' ')

with col2:
    st.write("Select Any Operations You want to Perform from SideBar\n")

with col3:
    st.write(' ')



Steps=st.sidebar.selectbox("Which Option you want to Choose?",('Data Cleaning','Statistical Tests','Feature Selection','Model Building'))

if Steps=='Data Cleaning':
	option = st.selectbox("Select the Step you want to Perform",('Check Null Values','Remove Null Values','Replace Null Values','Convert Dummies'))

	if option=='Check Null Values':
		st.write('Check Null Values',df.isnull().sum())


if Steps=='Statistical Tests':
	option = st.selectbox("Which Test you want to Perform?",('Hypothesis Test','ChiSquare Test','Correlation'))


	if option=='Hypothesis Test':

		p_value=st.slider('Select your P Value',0.0,0.05,0.01)

		variable=st.selectbox('Select the Continuous Variable you want to analyze',X.columns)
		x = np.array(df_new[df_new[target] == 1][variable])  
		y = np.array(df_new[df_new[target] == 0][variable]) 
		t, pValue  = stats.ttest_ind(x,y, axis = 0)
		if pValue < p_value:
			st.write("Alternate Hypothesis is True-There is Statistical Difference",pValue)  
		else:
			st.write("Null Hypothesis is True- There is no Statistical Difference",pValue)


	if option=='ChiSquare Test':

		p_value=st.slider('Select your P Value',0.0,0.05,0.01)
		variable1=st.selectbox('Select the Categorical Variable you want to analyze',X.columns)
		crosstab = pd.crosstab(df_new[variable1],df_new[target]) 
		chi, pValue, dof, expected =  stats.chi2_contingency(crosstab)
		if pValue < p_value:
			st.write("Alternate Hypothesis is True - Categorical variable has some effect",pValue)
		else:
			st.write("Null Hypothesis is True - Categorical variable has no effect",pValue)

	if option=='Correlation':
		selected_columns=st.multiselect('Select your variables for Correlation',df_new.columns)
		df1=df_new[selected_columns]
		st.dataframe(df1.corr())
		st.write("View HeatMap Below")
		fig=plt.figure()
		sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis')
		st.pyplot()

from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import matplotlib.pyplot as plt

if Steps=='Feature Selection':
	option = st.selectbox("Which Technique you want to Use?",('ChiSquare Feature Importance','Information Gain - Decision Tree','RFECV - Random Forest Feature Importance'))

	if option=='ChiSquare Feature Importance':
		x=df_new.drop(['BAD','REASON','JOB'],axis=1)
		y=df_new[target]
		feature_selector = SelectKBest(chi2, k = "all")
		fit = feature_selector.fit(x,y)
		pValues = pd.DataFrame(fit.pvalues_)
		scores = pd.DataFrame(fit.scores_)
		input_variable_names = pd.DataFrame(x.columns)
		summary_stats = pd.concat([input_variable_names, pValues, scores], axis = 1)
		summary_stats.columns = ["input_variable", "p_value", "chi2_score"]
		summary_stats.sort_values(by = "p_value", inplace = True)
		fig=plt.figure()
		sns.barplot(x='input_variable',y='chi2_score',data=summary_stats)
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()
		st.dataframe(summary_stats)

	if option=='RFECV - Random Forest Feature Importance':
		df_Dummies=pd.get_dummies(df_new)
		rfc = RandomForestClassifier()
		X=df_Dummies.drop('BAD',axis=1)
		Y=df_Dummies['BAD']
		select = RFECV(estimator=rfc, cv=10)
		select = select.fit(X,Y)
		df_feature=pd.DataFrame(select.ranking_,columns=['Rankings'])
		df_feature['Features']=X.columns
		df_feature=df_feature.sort_values(by='Rankings').reset_index(drop=True)
		st.dataframe(df_feature)

	if option=='Information Gain - Decision Tree':
		df_Dummies=pd.get_dummies(df_new)
		X=df_Dummies.drop('BAD',axis=1)
		Y=df_Dummies['BAD']
		model=DecisionTreeClassifier(max_depth=3,random_state=12345)
		model.fit(X,Y)
		fn=X.columns
		cn=['0','1']
		fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
		tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True)
		fig.savefig('DecisionTree.png')
		st.image("DecisionTree.png")

from sklearn.preprocessing import StandardScaler

if Steps=='Model Building':
	scalar=StandardScaler()
	X=df_dummy.drop(['BAD'],axis=1)
	X_std=scalar.fit_transform(X)
	Y=df_dummy['BAD']
	Test_Size=st.slider('Select The Size for your Test Data',0.0,1.0,0.1)
	x_train,x_test,y_train,y_test=train_test_split(X_std,Y,test_size=Test_Size,random_state=12345)

	option = st.selectbox("Which Model you want to Build?",('Logistic Regression','Random Forest','Decision Tree'))

	if option=='Logistic Regression':
		C=[0.01,0.1,1,10,100]
		c=st.selectbox('Choose Your C value',C)

		model=LogisticRegression(C=c)
		model.fit(x_train,y_train)

		pred=model.predict(x_test)
		accuracy=accuracy_score(y_test,pred)
		st.write('The Accuracy of Logistic Regression is', accuracy)

		fpr,tpr,threshold=roc_curve(y_test,model.predict_proba(x_test)[:,1])
		roc_auc=auc(fpr,tpr)
		st.write('Area under the ROC Curve', roc_auc)


	if option=='Decision Tree':
		depth=st.slider('Select The Max Depth',2,10,1)

		model=DecisionTreeClassifier(max_depth=depth)
		model.fit(x_train,y_train)

		pred=model.predict(x_test)
		accuracy=accuracy_score(y_test,pred)
		st.write('The Accuracy of Decision Tree is', accuracy)










		