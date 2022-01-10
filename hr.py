import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from PIL import Image



data=pd.read_csv('HR-Employee-Attrition.csv')

#1



side=st.sidebar.selectbox('Go to:',('What is HR Analytics?','Exploratory Data Analysis','Machine Learning Model'))

if side=='What is HR Analytics?':
	st.title('HR Analytics Prediction: Why do People Resign?')
	image = Image.open('hr_analytics.png')
	st.image(image, width=700)
	st.markdown("""
	 HR analytics is the process of collecting and analyzing Human Resource (HR) data in order to improve an organization’s workforce performance. The process can also be referred to as talent analytics, people analytics, or even workforce analytics. This method of data analysis takes data that is routinely collected by HR and correlates it to HR and organizational objectives. Doing so provides measured evidence of how HR initiatives are contributing to the organization’s goals and strategies.

	Need of HR Analytics Most organizations already have data that is routinely collected, so why the need for a specialized form of analytics? Can HR not simply look at the data they already have? Unfortunately, raw data on its own cannot actually provide any useful insight. It would be like looking at a large spreadsheet full of numbers and words. Once organized, compared and analyzed, this raw data provides useful insight.They can help answer questions like:

	What patterns can be revealed in employee turnover?
	How long does it take to hire employees?
	What amount of investment is needed to get employees up to a fully productive speed?
	Which of our employees are most likely to leave within the year?
	Are learning and development initiatives having an impact on employee performance?

	The process of HR Analytics:

	HR Analytics is made up of several components that feed into each other.

	To gain the problem-solving insights that HR Analytics promises, data must first be collected.
	The data then needs to be monitored and measured against other data, such as historical information, norms or averages.
	This helps identify trends or patterns. It is at this point that the results can be analyzed at the analytical stage.
	The final step is to apply insight to organizational decisions.
	So lets start with our analysis!!!
	""")

#2
if side=='Exploratory Data Analysis':
	eda_side=st.sidebar.radio('Select:',('Check the data','Analysis','Inference'))

	#eda
	if eda_side=='Check the data':
		st.title('Taking a glance at the data')
		st.image(Image.open('checkdata.jfif'),width=150)
		data_eda_side=st.sidebar.radio('Check the data:',('Data','Columns & Datatypes','Null Values'))
		if data_eda_side=='Data':
			st.subheader('The dataset-->')
			st.dataframe(data)
		if data_eda_side=='Columns & Datatypes':
			st.subheader('The columns and their datatypes-->')
			st.dataframe(data.dtypes)
		if data_eda_side=='Null Values':
			st.subheader('The Null Values-->')
			st.dataframe(data.isnull().sum())

	#eda
	if eda_side=='Analysis':
		plot_side=st.selectbox('Factors affecting Attrition:',('Analysis','A. How is attrition dependent on Age?','B. Is income the main factor towards employee attrition?','C. Does the Department of work impact attrition?','D. How does the environment satisfaction impact attrition?','E. Does company stocks for employees impact attrition?','F. How does Work Life Balance impact the overall attrition rates?','G. How does work experience affect attrition?','H. How does Work duration in current role impact Attrition?','I. Does Hike percentage impact Attrition?','J. Are managers a reason of people resigning?','K. How does self Job Satisfaction impact the Attrition?'))
		if plot_side=='Analysis':
			st.title('Analysis')
			st.image(Image.open('Analysis.png'),width=500)

		if plot_side=='A. How is attrition dependent on Age?':
			st.write('')
			age_att=data.groupby(['Age','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig = plt.figure(figsize=(10, 4))
			plt.title('Agewise Counts of People in an Organization')
			sns.lineplot(x = "Age", y = "Counts", data = age_att,hue='Attrition')
			st.pyplot(fig)
			st.subheader('Observation')
			st.write('As seen in the chart above, the attrition is maximum between the age groups 28-32. The attrition rate keeps on falling with increasing age, as people look after stability in their jobs at these point of times. Also at a very younger age, i.e. from 18-20, the chances of an employee leaving the organization is far more- since they are exploring at that point of time. It reaches a break even point at the age of 21')

		if plot_side=='B. Is income the main factor towards employee attrition?':	
			
			st.write('')
			rate_att=data.groupby(['MonthlyIncome','Attrition']).apply(lambda x:x['MonthlyIncome'].count()).reset_index(name='Counts')
			rate_att['MonthlyIncome']=round(rate_att['MonthlyIncome'],-3)
			rate_att=rate_att.groupby(['MonthlyIncome','Attrition']).apply(lambda x:x['MonthlyIncome'].count()).reset_index(name='Counts')
			fig1 = plt.figure(figsize=(10, 4))
			plt.title('Monthly Income basis counts of People in an Organization')
			sns.lineplot(x = 'MonthlyIncome', y = "Counts", data = rate_att,hue='Attrition')
			st.pyplot(fig1)
			st.subheader('Observation')
			st.write('As seen in the above chart, the attrition rate is evidently high at very low income levels- less than 5k monthly. This decreases further- but a minor spike is noticed aorund 10k- indicating the middle class liveliood. They tend to shift towards a better standard of living, and hence move to a different job. When the monthly income is pretty decent, the chances of an employee leaving the organization is low- as seen by the flat line')

		if plot_side=='C. Does the Department of work impact attrition?':
			
			st.write('')
			dept_att=data.groupby(['Department','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig2 = plt.figure(figsize=(10, 4))
			plt.title('Department wise Counts of People in an Organization')
			sns.barplot(x = 'Department', y = "Counts", data = dept_att,hue='Attrition')
			st.pyplot(fig2)
			st.subheader('Observation')
			st.write('This data comprises of only 3 major departments- among which Sales department has the highest attrition rates, followed by the Human Resource Department. Research and Development has the least attrition rates, that suggests the stability and content of the department as can be seen from the chart above.')

		if plot_side=='D. How does the environment satisfaction impact attrition?':
			
			st.write('')
			sats_att=data.groupby(['EnvironmentSatisfaction','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Environment Satisfaction level Counts of People in an Organization')
			sns.lineplot(x = 'EnvironmentSatisfaction',y='Counts', data=sats_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('In the satisfaction Level 1-2, the chances of peope leaving the organization slightly decreases. This is indicative of the better hopes with which people stay in an organization. However, as we move from 2-3, people tend to move on to get better opportunities and experiences. The attrition rate is almost stagnant for the higher satisfaction levels.')
		
		if plot_side=='E. Does company stocks for employees impact attrition?':
			
			st.write('')
			stock_att=data.groupby(['StockOptionLevel','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Stock facilities level wise People in an Organization')
			sns.barplot(x = 'StockOptionLevel',y='Counts', data=stock_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('The tendency of employees to leave the organization is much more when the stock availing options are limited. Since the stocks constitute to a huge amount of money while staying for a few years, people do not want to lose that opportunity. People with very limited/no stcok options have a freedom to leave the organization at will.')

		if plot_side=='F. How does Work Life Balance impact the overall attrition rates?':
			
			st.write('')
			wlb_att=data.groupby(['WorkLifeBalance','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Work Life Balance level Counts of People in an Organization')
			sns.barplot(x = 'WorkLifeBalance',y='Counts', data=wlb_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write(' People with poor levels of Work life balance have adjusted themselves to their jobs, but as seen for the above parameters with a better work life score, people are more accustomed to the better life and want to go for an attrition more. But this trend perishes when the work life balance is really good, and people are satisfied with the work they are doing.')

		if plot_side=='G. How does work experience affect attrition?':
			
			st.write('')
			we_att=data.groupby(['NumCompaniesWorked','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Work Experience level Counts of People in an Organization')
			sns.lineplot(x = 'NumCompaniesWorked',y='Counts', data=we_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('As seen from the chart above, clearly, employees who started their career with the company- or have switched to the company in the initial years of their career, have a higher chances of leaving the organization to a different company. People who have gained much experience- working in multiple companies tend to stay in the company they join.')

		if plot_side=='H. How does Work duration in current role impact Attrition?':
			
			st.write('')
			yrscr_att=data.groupby(['YearsInCurrentRole','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Counts of People working for years in an Organization')
			sns.lineplot(x = 'YearsInCurrentRole',y='Counts', data=yrscr_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('We have seen people are more prone to leave the organization in the starting years on their role. When people are in the same role for a long period of time, they tend to stay longer for moving in an upward role.')

		if plot_side=='I. Does Hike percentage impact Attrition?':
			
			st.write('')
			hike_att=data.groupby(['PercentSalaryHike','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Count of Hike Percentages people receive in an Organization')
			sns.lineplot(x = 'PercentSalaryHike',y='Counts', data=hike_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('Higher hikes motivate people to work better, and stay in the organization. Hence we see the chances of an employee leaving the organization where the hike is lower, is much more than a company that gives a good hike.')

		if plot_side=='J. Are managers a reason of people resigning?':
			
			st.write('')
			man_att=data.groupby(['YearsWithCurrManager','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Count of people spending years with a Manager in an Organization')
			sns.lineplot(x = 'YearsWithCurrManager',y='Counts', data=man_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('We notice 3 major spikes in the attrition rate, when we are analyzing the relationship of an employee with their manager. At the very start, where the time spent with the manager is relatively less- people tend to leave their jobs- considering their relationship with their previous managers. At an average span of 2 years, when employees feel they need an improvement, they also tend to go for a change. When the time spent with the manager is slightly higher (about 7 years)- people tend to find their career progression stagnant, and tend to go for a change. But when the relative time spend with a manager is very high- people are satisfied with their work. Hence the chances of an employee resigning then is significantly low.')

		if plot_side=='K. How does self Job Satisfaction impact the Attrition?':
			
			st.write('')
			jsats_att=data.groupby(['JobSatisfaction','Attrition']).apply(lambda x:x['DailyRate'].count()).reset_index(name='Counts')
			fig3 = plt.figure(figsize=(10, 4))
			plt.title('Job Satisfaction level Counts of People in an Organization')
			sns.lineplot(x = 'JobSatisfaction',y='Counts', data=jsats_att,hue='Attrition')
			st.pyplot(fig3)
			st.subheader('Observation')
			st.write('With an increasing job satisfaction, the attrition rates decrease as can be seen in the chart above. Also from range 1-2 range we can infer (as seen above in Environment Satisfaction), the attrition level falls, but raises from 2-3, where the people tend to coose better opportunities.')

	if eda_side=='Inference':
		st.subheader('We have checked the data, and have come upon to infer the following observations:')
		st.write('1. People are tending to switch to a different jobs at the start of their careers, or at the earlier parts of it. Once they have settled with a family or have found stability in their jobs, they tend to stay long in the same organization- only going for vertical movements in the same organization.')
		st.write('2. Salary and stock ptions have a great motivation on the employees and people tend to leave the organization much lesser. Higher pay and more stock options have seen more employees remain loyal to their company.')
		st.write('3. Work life balance is a great motivation factor for the employees. However, people with a good work-life balance, tend to switch in search of better opportunities and a better standard of living.')
		st.write('4. Departments where target meeting performance is very much crucial (for e.g. Sales) tend to have a greater chances of leaving the organization as compared to departments with more administration perspective (For e.g. Human Resources)')
		st.write('5. People with a good Job Satisfaction and Environment satisfaction are loyal to the organization- and this speaks loud for any Organization. However, people who are not much satisfied with their current project- tend to leave the organization far more.')


if side=='Machine Learning Model':


	X=data[['Age', 'DailyRate',
       'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction','StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]
	y=data[['Attrition']].values.ravel()
	X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
	

	def user_report():
		age=st.slider("Age:",0,90,25,1)
		DailyRate=st.slider("Daily Rate",0,1500,100,100)
		distance=st.slider("Distance From Home",0,60,20,1)
		education=st.slider("Education Level",1,5,3,1)
		environment=st.slider("Environment Satisfaction Level",1,5,3,1)
		hourlyrate=st.slider("Hourly Rate:",1,100,50,1)
		JobInvolvement=st.slider("Job Involvement Level:",1,5,3,1)
		JobLevel=st.slider("Job Level:",1,5,3,1)
		JobSatisfaction=st.slider("JobSatisfaction",1,5,3,1)
		MonthlyIncome=st.slider("Monthly Income",0,50000,1000,1000)
		MonthlyRate=st.slider('Monthly Rate',0,50000,1000,1000)
		NumCompaniesWorked=st.slider("NumCompaniesWorked",0,10,3,1)
		PercentSalaryHike=st.slider("PercentSalaryHike",0,35,10,1)
		PerformanceRating=st.slider("PerformanceRating",1,5,3,1)
		RelationshipSatisfaction=st.slider("RelationshipSatisfaction",1,5,3,1)
		StandardHours=st.selectbox('StandardHours',"80")
		StockOptionLevel=st.slider("StockOptionLevel",1,5,3,1)
		TotalWorkingYears=st.slider("TotalWorkingYears",0,50,3,1)
		TrainingTimesLastYear=st.slider("TrainingTimesLastYear",0,10,3,1)
		WorkLifeBalance=st.slider("WorkLifeBalance",1,5,3,1)
		YearsAtCompany=st.slider("YearsAtCompany",0,50,3,1)
		YearsInCurrentRole=st.slider("YearsInCurrentRole",0,20,3,1)
		YearsSinceLastPromotion=st.slider("YearsSinceLastPromotion",0,20,3,1)
		YearsWithCurrManager=st.slider("YearsWithCurrManager",0,20,3,1)

		user_report_data = {
	        'Age':age,'DistanceFromHome':distance,'DailyRate':DailyRate,
	        'Education':education, 'Environment':environment, 'HourlyRate':hourlyrate, 'JobInvolvement':JobInvolvement,
	        'JobLevel':JobLevel, 'JobSatisfaction':JobSatisfaction, 'MonthlyIncome':MonthlyIncome,'MonthlyRate':MonthlyRate,
	        'NumCompaniesWorked':NumCompaniesWorked, 'PercentSalaryHike':PercentSalaryHike, 'PerformanceRating':PerformanceRating,
	        'RelationshipSatisfaction':RelationshipSatisfaction,'StandardHours':StandardHours,
	        'StockOptionLevel':StockOptionLevel,'TotalWorkingYears':TotalWorkingYears,
	        'TrainingTimesLastYear':TrainingTimesLastYear,'WorkLifeBalance':WorkLifeBalance,
	        'YearsAtCompany':YearsAtCompany, 'YearsInCurrentRole':YearsInCurrentRole,
	        'YearsSinceLastPromotion':YearsSinceLastPromotion,'YearsWithCurrManager':YearsWithCurrManager
	    }
		report_data = pd.DataFrame(user_report_data, index=[0])
		return report_data

	user_data = user_report()
	log_reg=LogisticRegression(C=1000,max_iter=10000)
	log_reg.fit(X_train,y_train)

	submit = st.button('Predict')
	if submit:
		user_result = log_reg.predict(user_data)
		if user_result=='No':
			st.title('No, the candidate will not leave the company.')
		else:
			st.title('Yes, the candidate will leave the company.')

