##Import necessary libraries and modules


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px
from dash import dcc
import dash_core_components as dc    
import plotly.graph_objects as go# Create a Dash app
import plotly.express as px


############# Import required Dash components and libraries
############# Load the necessary data for analysis
employee_df = pd.read_excel('Employee.xlsm', engine='openpyxl')
performanceReview_df = pd.read_excel('Performancereview.xlsm', engine='openpyxl')
Survey_df = pd.read_excel('Survey.xlsm', engine='openpyxl')

############# Handle missing values
employee_df['DegreeField'].fillna('Unknown', inplace=True)
employee_df['MaritalStatus'].fillna('Unknown', inplace=True)
employee_df['NumPreviousCompanies'].fillna(0, inplace=True)
employee_df['EmploymentEndReason'].fillna('Unknown', inplace=True)

############# Convert date columns to datetime
employee_df['EmploymentEndDate'] = pd.to_datetime(employee_df['EmploymentEndDate'])
employee_df['EmploymentStartDate'] = pd.to_datetime(employee_df['EmploymentStartDate'])


############# Define the attrition variable
def determine_attrition(end_date, reason):
    if pd.isna(end_date):
        return 'No Attrition'
    elif reason in ['Went to another company', 'Retired', 'Fired']:
        return 'Yes'
    else:
        return 'No Attrition'
    

employee_df['Attrition'] = employee_df.apply(lambda row: determine_attrition(row['EmploymentEndDate'], row['EmploymentEndReason']), axis=1)

# Merge the dataframes on 'EmployeeId'
merged_df = pd.merge(employee_df, performanceReview_df, on='EmployeeId', how='inner')
merged_df = pd.merge(merged_df, Survey_df, on='EmployeeId', how='inner')

#feature Engineering
merged_df['Age'] = merged_df['ReviewDate'].dt.year - merged_df['YearOfBirth']
merged_df['Tenure'] = merged_df['NumYearsWorked'] + (merged_df['ReviewDate'] - merged_df['EmploymentStartDate']).dt.days / 365
merged_df['EmploymentDuration'] = (merged_df['ReviewDate'] - merged_df['EmploymentStartDate']).dt.days
merged_df['PerformanceRatingDiff'] = merged_df.groupby('EmployeeId')['PerformanceRating'].diff()
merged_df['ReviewYear'] = merged_df['ReviewDate'].dt.year
merged_df['ReviewMonth'] = merged_df['ReviewDate'].dt.month
# Define a threshold for being overworked
overtime_threshold = 40 
merged_df['Overworked'] = (merged_df['OvertimeHours'] > overtime_threshold) | (merged_df['OvertimeDays'] > 0)

############# Derive the employee tenure at company
merged_df_dummy = merged_df
current_date = pd.to_datetime('2020-1-1')
merged_df_dummy['YearsAtCompany'] = (current_date - merged_df_dummy['EmploymentStartDate']).dt.days / 365
merged_df_dummy['YearsAtCompany'] =round(merged_df_dummy['YearsAtCompany'])


############# Drop Dates and unique values, required information is already captured from these columns

merged_df_dummy = merged_df_dummy.drop(['EmployeeId', 'EmploymentEndDate', 'EmploymentStartDate', 'ReviewDate','Attrition'], axis=1)
# Convert categorical columns to dummies
categorical_columns = ['DegreeCompleted', 'DegreeField', 'Department', 'Gender', 'MaritalStatus', 'TravelFrequency', 'QuestionNum', 'Response', 'Overworked','EmploymentEndReason','QuestionText']
merged_df_dummy = pd.get_dummies(merged_df_dummy, columns=categorical_columns)
merged_df_dummy['Attrition']=merged_df['Attrition']
merged_df_dummy['Attrition']=merged_df_dummy['Attrition'].astype(str)

correlation_matrix = merged_df_dummy.corr()

# Find columns with high correlation
threshold = 0.7 
collinear_columns = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            collinear_columns.add(colname)

# Remove collinear columns

attrition_yes_count = len(employee_df[employee_df['Attrition'] == 'Yes'])
attrition_no_count = len(employee_df[employee_df['Attrition'] == 'No Attrition'])
total_employees = len(employee_df)
attrition_rate = (attrition_yes_count / total_employees) * 100

object_columns = merged_df.select_dtypes(include=['object']).columns

# Convert object columns to dummies

# Drop the date columns and 'EmployeeId' column
merged_df_dummy_collinear = merged_df_dummy.drop(columns=collinear_columns)
merged_df_dummy_collinear['YearsAtCompany']=merged_df_dummy['YearsAtCompany']
merged_df_dummy_collinear=merged_df_dummy_collinear.drop('PerformanceRatingDiff',axis=1)
merged_df_dummy_collinear=merged_df_dummy_collinear.drop('WeeklyHoursBudgeted',axis=1)


## Removing post Attrition variables 
merged_df_dummy_input=merged_df_dummy_collinear.drop(['QuestionNum_Q1', 'QuestionNum_Q2',
       'QuestionNum_Q3', 'QuestionNum_Q4', 'Response_Excellent',
       'Response_Fair', 'Response_Good',
       'Response_Neither Satisfied nor Unsatisfied', 'Response_Poor',
       'Response_Somewhat Satisfied', 'Response_Somewhat Unsatisfied',
       'Response_Very Poor', 'Response_Very Satisfied',
       'Response_Very Unsatisfied','EmploymentEndReason_Fired',
       'EmploymentEndReason_Went to another company'],axis=1)


#### Modelling Logistic Regression as base model for benchmarking results
subset_df = merged_df_dummy_input
#subset_df = subset_df.drop(['EmploymentStartDate', 'EmploymentEndDate', 'EmployeeId','YearOfBirth','ReviewDate','PerformanceRatingDiff'], axis=1)

subset_df1 = pd.get_dummies(subset_df.drop('Attrition', axis=1), drop_first=True)

subset_df1 = pd.concat([subset_df['Attrition'], subset_df1], axis=1)

# Define the target variable
X = subset_df1.drop('Attrition', axis=1)
y = subset_df1['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and fit the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Fetch feature importance from the Logistic Regression model
logistic_feature_importance = logistic_model.coef_[0]

# Build and fit the Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)

# Fetch feature importance from the Random Forest model
rf_feature_importance = random_forest_model.feature_importances_

# Create dataframes to visualize the feature importances
logistic_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Logistic Importance': logistic_feature_importance})
rf_feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Random Forest Importance': rf_feature_importance})

# Merge the dataframes to compare the feature importances
feature_importance_comparison = pd.merge(logistic_feature_importance_df, rf_feature_importance_df, on='Feature')

# Sort by importance values to see the most important features
feature_importance_comparison = feature_importance_comparison.sort_values(by=['Logistic Importance', 'Random Forest Importance'], ascending=False)

# Print the common important columns
common_important_columns = feature_importance_comparison[feature_importance_comparison['Logistic Importance'] > 0]['Feature']

print(common_important_columns)


# Testing with the initial Logistic Regression model
initial_logistic_predictions = logistic_model.predict(X_test)
initial_logistic_precision = precision_score(y_test, initial_logistic_predictions, pos_label='Yes')
initial_logistic_recall = recall_score(y_test, initial_logistic_predictions, pos_label='Yes')
initial_logistic_f1 = f1_score(y_test, initial_logistic_predictions, pos_label='Yes')
initial_logistic_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1], average='macro')

# Testing with the initial Random Forest model
initial_rf_predictions = random_forest_model.predict(X_test)
initial_rf_precision = precision_score(y_test, initial_rf_predictions, pos_label='Yes')
initial_rf_recall = recall_score(y_test, initial_rf_predictions, pos_label='Yes')
initial_rf_f1 = f1_score(y_test, initial_rf_predictions, pos_label='Yes')
initial_rf_auc = roc_auc_score(y_test, random_forest_model.predict_proba(X_test)[:, 1], average='macro')

print("Metrics for Initial Logistic Regression Model:")
print("Precision: {:.2f}".format(initial_logistic_precision))
print("Recall: {:.2f}".format(initial_logistic_recall))
print("F1 Score: {:.2f}".format(initial_logistic_f1))
print("AUC: {:.2f}".format(initial_logistic_auc))
print("\nMetrics for Initial Random Forest Model:")
print("Precision: {:.2f}".format(initial_rf_precision))
print("Recall: {:.2f}".format(initial_rf_recall))
print("F1 Score: {:.2f}".format(initial_rf_f1))
print("AUC: {:.2f}".format(initial_rf_auc))



def plot_feature_importance(feature_importance_df):
    # Sort the feature importance values in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Random Forest Importance', ascending=False)

    # Create a bar plot for feature importance
    feature_importance_plot = px.bar(
        feature_importance_df,
        x='Feature',
        y='Random Forest Importance',
        title='Feature Importance (Random Forest)',
    )

    feature_importance_plot.update_layout(xaxis_title='Feature', yaxis_title='Importance')
    
    return feature_importance_plot



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

plot_border_style = {
    'border': '2px solid #808080',  #  can adjust the border width and color
    'border-radius': '5px',  # Rounded corners
    'padding': '10px',  # Padding inside the border
}




# Define the CSS class for the plot borders
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Plots', children=[
            html.Div([
                html.H3("Attrition Statistics"),
                html.Div(f"Attrition Rate: {attrition_rate:.2f}%", style={'fontSize': 20}),
                html.Div(f"Yes: {attrition_yes_count}", style={'fontSize': 20}),
                html.Div(f"No: {attrition_no_count}", style={'fontSize': 20}),
                ]),
            dbc.Row([
                dbc.Col(dcc.Graph(
                    id='attrition_trend_plot',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ), width=6),
                dbc.Col(dcc.Graph(
                    id='overtime_analysis_plot',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ), width=6),
                dbc.Col(dcc.Markdown('''
                Between the years 2005 and 2019, there is a noticeable declining trend in 
                attrition percentage. It starts at a high of 90% in 2005 and gradually 
                decreases to 40.75% in 2019. The attrition percentage experienced 
                fluctuations but generally remained at lower levels from 2010 to 2019. 
                There are minor fluctuations, but the overall trend is towards reduced attrition during this period.
                ''')),

            ]),

            dbc.Row([  # Create a new row for the "Survey Response" plot
                dbc.Col(dcc.Graph(
                    id='survey_responses_plot',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ), width=6),
                dbc.Col(dcc.Graph(
                    id='performance-rating-plot',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ), width=6)
            ]),
            dbc.Alert(
                "These statistics provide insights into the attrition percentages associated with different survey response categories during high attrition years. It's clear that the 'Very Satisfied' category had the highest attrition percentage at 30.0%, while other categories had lower attrition percentages, ranging from 10.0% to 20.0%. Understanding these percentages can help in making informed decisions related to employee satisfaction and retention strategies.",
                color="info",
            ),
            dbc.Row([ dbc.Col(dcc.Graph(
                id='department_attrition_plot',
                config={'displayModeBar': True},
                className='plot-border'  # Apply the CSS class to the plot
                ),width=6),
                dbc.Col(dcc.Graph(
                    id='correl_plot',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ),width=6)
            ]),
            dbc.Alert( "Attrition statistics by department:"
                    "Accounting: 32.66% attrition, 67.34% no attrition."
                    "Human Resources: 30.21% attrition, 69.79% no attrition."
                    "IT: 32.99% attrition, 67.01% no attrition."
                    "Marketing: 35.75% attrition, 64.25% no attrition."
                    "Other: 37.09% attrition, 62.91% no attrition."
                    "Sales: 43.89% attrition, 56.11% no attrition."
                   ,
                color="info",
                ),
            dbc.Row([ dbc.Col(dcc.Graph(
                id='performance_fig',
                config={'displayModeBar': True},
                className='plot-border'  # Apply the CSS class to the plot
                ),width=6),
                dbc.Col(dcc.Graph(
                    id='attrition_percentage_by_survey_questionfig',
                    config={'displayModeBar': True},
                    className='plot-border'  # Apply the CSS class to the plot
                ),width=6)
            ]),
                dbc.Alert( "**Question Q1:**"
                      "Employees who responded as Somewhat Unsatisfied had the highest attrition percentage at 56.44%. This suggests that employees who are somewhat unsatisfied with their situation are more likely to leave the company."
                      "Very Satisfied employees had a relatively low attrition percentage of 40.53%, indicating that high job satisfaction is associated with lower attrition."

                        "**Question Q2:**"

                    "Employees who are Neither Satisfied nor Unsatisfied showed the highest attrition percentage at 51.93%, which is quite high. This suggests that employees who have mixed feelings or are indifferent are more likely to leave."
                    "In contrast, employees who are Somewhat Unsatisfied also have a high attrition percentage of 27.67%, indicating that dissatisfaction plays a significant role in attrition."

                    "**Question Q3:**"

                    "Employees who responded Very Unsatisfied had the highest attrition percentage at 18.52%, which is lower compared to other responses. However, it's still a significant indicator of potential attrition."
                    "Very Satisfied employees have an attrition percentage of 45.62%, which is relatively high, showing that high satisfaction doesn't guarantee retention."

                    "**Question Q4:**"

                    "Employees who rated their experience as Fair had the highest attrition percentage at 51.81%, suggesting that when employees perceive their situation as average, they are more likely to leave."
                    "Excellen employees had the lowest attrition percentage at 38.87%, indicating that providing an excellent experience can help retain employees."
                   ,
                color="info",
                ),
        ]),

        dcc.Tab(label='Predictions', children=[

            html.H2("Feature Importances"),
            dcc.Graph(
                id='feature-importance-plot',
                config={'displayModeBar': True}
                ),
            html.H1("Model Metrics"),
            html.H2("Metrics for Initial Logistic Regression Model:"),
            html.P("Precision: {:.2f}".format(initial_logistic_precision)),
            html.P("Recall: {:.2f}".format(initial_logistic_recall)),
            html.P("F1 Score: {:.2f}".format(initial_logistic_f1)),
            html.P("AUC: {:.2f}".format(initial_logistic_auc)),
            html.H2("Metrics for Initial Random Forest Model:"),
            html.P("Precision: {:.2f}".format(initial_rf_precision)),
            html.P("Recall: {:.2f}".format(initial_rf_recall)),
            html.P("F1 Score: {:.2f}".format(initial_rf_f1)),
            html.P("AUC: {:.2f}".format(initial_rf_auc)),
        ]),
    ])
])



# Define callback to update plots
@app.callback(
    Output('attrition_trend_plot', 'figure'),
    Output('overtime_analysis_plot', 'figure'),
    Output('survey_responses_plot', 'figure'),
    Output('performance-rating-plot', 'figure'),
    Output('department_attrition_plot', 'figure'),
    Output('correl_plot', 'figure'),
    Output('performance_fig', 'figure'),
    Output('attrition_percentage_by_survey_questionfig', 'figure'),
    Output('feature-importance-plot', 'figure'),
    Input('attrition_trend_plot', 'id'),
    Input('overtime_analysis_plot', 'id'),
    Input('survey_responses_plot', 'id'),
    Input('performance-rating-plot', 'id'),
    Input('department_attrition_plot', 'id'),
    Input('correl_plot', 'id'),
    Input('performance_fig', 'id'),
    Input('attrition_percentage_by_survey_questionfig', 'id'),
    Input('feature-importance-plot', 'id')
)

def update_plots(*args):
    total_employees = len(merged_df)
    attrition_yes_count = len(merged_df[merged_df['Attrition'] == 'Yes'])
    average_attrition_rate = (attrition_yes_count / total_employees) * 100
    attrition_counts = merged_df['Attrition'].value_counts()
    attrition_yes_count = attrition_counts['Yes']
    attrition_no_count = attrition_counts['No Attrition']
    attr_percentage = employee_df.copy()
    attr_percentage['EmploymentEndMonth'] = (attr_percentage['EmploymentStartDate']).dt.year
    attr_percentage = attr_percentage.groupby('EmploymentEndMonth')['Attrition'].value_counts(normalize=True).unstack(fill_value=0)['Yes'] * 100
    attr_percentage = pd.Series(attr_percentage).reset_index()
    attr_percentage.columns = ['Year', 'Attrition Percentage']
    attr_percentage = attr_percentage[attr_percentage['Year'] >= 2005]
    attrition_trend_plot = px.line(attr_percentage, x='Year', y='Attrition Percentage', title='Attrition Trend by Year')
    attrition_trend_plot.update_traces(mode='markers+lines', marker=dict(size=8))
    attrition_trend_plot.update_layout(
        xaxis_title='Year',
        yaxis_title='Attrition Percentage (%)',
        legend_title='Attrition'
    )
    
    high_attrition_years = [2006, 2007, 2011,2019] 
    high_attrition_years = [2006, 2007, 2011, 2019]
    survey_year = merged_df.copy()
    survey_year['EmploymentEndMonth'] = survey_year['EmploymentStartDate'].dt.year

    filtered_df = survey_year[survey_year['EmploymentEndMonth'].isin(high_attrition_years)]

    survey_responses = filtered_df.groupby(['EmploymentEndMonth', 'QuestionText', 'Response'])['EmployeeId'].count().reset_index(name='Frequency')

# Create subplots for each year
    survey_responses_plot = px.bar(survey_responses, x='Response', y='Frequency', facet_col='EmploymentEndMonth',
             labels={'Frequency': 'Frequency'},
             title=f'Attrition Percentages by Survey Response',
             width=800, height=500)
    survey_responses_plot.update_traces(texttemplate='%{text:.2f}%',  textposition='outside')



    # Code for "Overtime Analysis during High Attrition Years"
    high_attrition_years = [2006, 2007, 2011, 2019]
    attr_year = employee_df.copy()
    attr_year['EmploymentEndMonth'] = (attr_year['EmploymentStartDate']).dt.year
    filtered_df = attr_year[attr_year['EmploymentEndMonth'].isin(high_attrition_years)]
    average_overtime_days = filtered_df.groupby('EmploymentEndMonth')['OvertimeDays'].mean()
    average_overtime_hours = filtered_df.groupby('EmploymentEndMonth')['OvertimeHours'].mean()
    average_data = pd.DataFrame({
        'EmploymentEndMonth': average_overtime_days.index,
        'Average Overtime Days': average_overtime_days.values,
        'Average Overtime Hours': average_overtime_hours.values
    })
    overtime_analysis_plot = px.line(average_data, x='EmploymentEndMonth', y=['Average Overtime Days', 'Average Overtime Hours'],
                                      labels={'value': 'Average Overtime', 'EmploymentEndMonth': 'Highest Attrition Years'},
                                      title='Overtime Analysis during High Attrition Years')

    # Code for "Performance Ratings Percentage in High Attrition Years"
    performance_year = merged_df.copy()
    performance_year['EmploymentEndMonth'] = performance_year['EmploymentStartDate'].dt.year
    filtered_df = performance_year[performance_year['EmploymentEndMonth'].isin(high_attrition_years)]
    performance_percentages = filtered_df.groupby(['EmploymentEndMonth'])['PerformanceRating'].value_counts(normalize=True).reset_index(name='Percentage')
    colors = px.colors.qualitative.Set1  # You can use a different color scale

    performance_rating_plot = px.bar(performance_percentages, x='PerformanceRating', y='Percentage',
                                     title='Performance Ratings Percentage in High Attrition Years',
                                     labels={'PerformanceRating': 'Performance Rating', 'Percentage': 'Percentage'},
                                     text='Percentage',
                                     facet_col='EmploymentEndMonth',color_discrete_sequence=colors)


# Set colors for the 'PerformanceRating' values
    color_mapping = {rating: color for rating, color in zip(performance_percentages['PerformanceRating'].unique(), colors)}
    performance_rating_plot.update_traces(marker_color=[color_mapping[rating] for rating in performance_percentages['PerformanceRating']])

    performance_rating_plot.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    #performance_rating_plot.update_traces(marker=dict(line=dict(width=2, color='Black')))


    #Code for "Department-wise Attrition Percentage"
    attrition_by_department = employee_df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack(fill_value=0).reset_index()
    attrition_by_department.reset_index(inplace=True)
    attrition_by_department.columns = ['Attrition', 'Department', 'No Attrition', 'Yes']
    attrition_by_department['Attrition Percentage'] = attrition_by_department['Yes'] * 100
    department_attrition_plot = px.bar(attrition_by_department, x='Department', y='Attrition Percentage', color='Attrition',
                                        title='Department-wise Attrition Percentage')
# Show the plot

    correlation_matrix = merged_df.corr()

# Create an interactive heatmap using Plotly Express
    correl_plot = px.imshow(correlation_matrix,
                labels=dict(color='Correlation'),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='Viridis')

# Update the layout for a better display
    correl_plot.update_layout(
    title="Correlation Plot",
    xaxis_title="Features X",
    yaxis_title="Features Y"
    )
    
    attrition_percentage_by_survey_question_df = (
    merged_df.groupby(['QuestionNum', 'Response'])['Attrition'].value_counts(normalize=True).unstack(fill_value=0)['Yes'] * 100
    )
    attrition_percentage_by_survey_question_df = pd.Series(attrition_percentage_by_survey_question_df).reset_index()
    attrition_percentage_by_survey_question_df.columns = ['QuestionNum', 'Response', 'Attrition Percentage']


    attrition_percentage_by_survey_questionfig = px.bar(attrition_percentage_by_survey_question_df, x='QuestionNum', y='Attrition Percentage', color='Response',
             title='Attrition Percentage by Survey Question and Response',
             labels={'Attrition Percentage': 'Attrition Percentage (%)'},
             width=800, height=500,
             hover_data={'Attrition Percentage': ':.2f%'},  # Format as percentage with two decimal places
             text='Attrition Percentage',  # Display the percentage value on the bars
             category_orders={"QuestionNum": ["Q1", "Q2", "Q3", "Q4"]},  # Specify the order of categories
             )

# Customize the hover information and text format
    attrition_percentage_by_survey_questionfig.update_traces(texttemplate='%{text}', textposition='outside')
    attrition_percentage_by_performance = (
    merged_df.groupby('PerformanceRating')['Attrition'].value_counts(normalize=True).unstack(fill_value=0)['Yes'] * 100
    )
    data_df = pd.DataFrame({'PerformanceRating': [1, 2, 3, 4, 5], 'AttritionPercentage': attrition_percentage_by_performance})
    performance_fig = px.pie(data_df, names='PerformanceRating', values='AttritionPercentage', title='Attrition Percentage by Performance Rating', hole=0.4)

    performance_fig.update_traces(textinfo='none')
    feature_importance_comparison = pd.merge(logistic_feature_importance_df, rf_feature_importance_df, on='Feature')

    # Sort by importance values to see the most important features
    feature_importance_comparison = feature_importance_comparison.sort_values(by=['Logistic Importance', 'Random Forest Importance'], ascending=False)


# Create a bar plot for feature importances
    feature_importance_comparison = feature_importance_comparison.sort_values(by='Logistic Importance', ascending=False)

# Create a grouped bar plot for feature importances
    feature_importance_plot = px.bar(
    feature_importance_comparison,
    x='Feature',
    y=['Logistic Importance', 'Random Forest Importance'],
    title='Feature Importance Comparison',
    barmode='group',  # Grouped bar plot
    )

# Customize the appearance of the plot
    feature_importance_plot.update_layout(xaxis_title='Feature', yaxis_title='Importance')
    feature_importance_plot.update_traces(marker_color=['blue', 'green'])  # Change the bar colors

# Show the plot
    feature_importance_plot.show()   
    
    return (attrition_trend_plot, overtime_analysis_plot, survey_responses_plot,performance_rating_plot, department_attrition_plot,correl_plot,attrition_percentage_by_survey_questionfig,performance_fig,feature_importance_plot)

if __name__ == '__main__':
    app.run_server(debug=True)
