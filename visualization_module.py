import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#---------------------------------------------------------

def create_sub_dataframes(df):
    """
    Create and return sub dataframes based on specific conditions related to mental health, LGBT status, disabilities, gender, ethnicity, religion, and caring responsibilities.
    """
   

    ### Mental Health Sub DataFrames
    df_mental_health = pd.DataFrame()
    mental_health_columns_1 = ['How_often_feeling_worried_nervous_anxious', 'How_often_feeling_depressed']
    mental_health_columns_2 = ['Level_of_last_worrying_anxiety_nervousness', 'Level_of_last_depression']
    mental_health_levels_1 = ['Weekly', 'Monthly', 'Daily']
    mental_health_levels_2 = ['A lot', 'Somewhere in between a little and a lot']

    if set(mental_health_columns_1 + mental_health_columns_2).issubset(df.columns):
        
        for column in mental_health_columns_1:
            df_mental_health = pd.concat([df_mental_health, df[df[column].isin(mental_health_levels_1)]])
        for column in mental_health_columns_2:
            df_mental_health = pd.concat([df_mental_health, df[df[column].isin(mental_health_levels_2)]])
        df_mental_health = df_mental_health.drop_duplicates()
        # Create 'has_mental_health' column in the main DataFrame
        df['has_mental_health'] = 'No'
        if not df_mental_health.empty:
            df.loc[df_mental_health.index, 'has_mental_health'] = 'Yes'
        
    ### LGBT Sub DataFrame
    df_LGBT = pd.DataFrame()
    if 'Sexual_Orientation' in df.columns:
        df_LGBT = df[~df['Sexual_Orientation'].isin(['Heterosexual/straight'])]
        # Create 'LGBT' column in the main DataFrame
        df['LGBT'] = 'No'
        if not df_LGBT.empty:
            df.loc[df_LGBT.index, 'LGBT'] = 'Yes'

    ### Disabilities Sub DataFrame
    disability_columns = ['Seeing_Dificulty', 'Hearing_Dificulty', 'Walking_Dificulty', 'Remembering_Dificulty', 'SelfCare_Dificulty', 'Communicating_Dificulty', 'Raising_Water/Soda_Bottle_Dificulty', 'Picking_Up_Small_Objects_Dificulty']
    difficulty_levels = ['Yes, some difficulty', 'Yes, a lot of difficulty', 'Cannot do it at all']

    df_disabilities = pd.DataFrame()
    if set(disability_columns).issubset(df.columns) or 'Disability_or_Long_Term_Health_Condition' in df.columns:
        if set(disability_columns).issubset(df.columns):
            
            for column in disability_columns:
                df_disabilities = pd.concat([df_disabilities, df[df[column].isin(difficulty_levels)]])
            df_disabilities = df_disabilities.drop_duplicates()
            # Create 'Has_Disability' column in the main DataFrame
            df['Has_Disability'] = 'No'
            if not df_disabilities.empty:
                df.loc[df_disabilities.index, 'Has_Disability'] = 'Yes'

        elif 'Disability_or_Long_Term_Health_Condition' in df.columns:
            df_disabilities = df[df['Disability_or_Long_Term_Health_Condition'] == 'Yes']
            # Create 'Has_Disability' column in the main DataFrame
            df['Has_Disability'] = df['Disability_or_Long_Term_Health_Condition']

    ### Women Sub DataFrame
    df_women = pd.DataFrame()
    if 'Gender' in df.columns:
        df_women = df[df['Gender'].isin(['Woman', 'Female'])]

    ### Minority Ethnicity Sub DataFrame
    df_minority_ethnicity = pd.DataFrame()
    if 'Ethnicity' in df.columns:
        df_minority_ethnicity = df[~df['Ethnicity'].isin(['English, Welsh, Scottish, Northern Irish or British', 'Any other white background'])]

    ### Religious Beliefs Sub DataFrame
    df_religious_beliefs = pd.DataFrame()
    if 'Religion' in df.columns:
        df_religious_beliefs = df[~df['Religion'].isin(['No religion', 'Prefer not to say'])]

    ### Caring Responsibilities Sub DataFrame
    df_caring_responsibilities = pd.DataFrame()
    if 'Has_Caring_Responsibility' in df.columns:
        df_caring_responsibilities = df[df['Has_Caring_Responsibility'] == 'Yes']

    ### Update session state
    st.session_state['df_mental_health'] = df_mental_health
    st.session_state['df_LGBT'] = df_LGBT
    st.session_state['df_disabilities'] = df_disabilities
    st.session_state['df_women'] = df_women
    st.session_state['df_minority_ethnicity'] = df_minority_ethnicity
    st.session_state['df_religious_beliefs'] = df_religious_beliefs
    st.session_state['df_caring_responsibilities'] = df_caring_responsibilities

    return df_mental_health, df_LGBT, df_disabilities, df_women, df_minority_ethnicity, df_religious_beliefs, df_caring_responsibilities



# Define a function to return a dictionary of groups and their corresponding dataframes
def update_group_dfs(df, groups_info):
    """
    Updates the group_dfs dictionary with dataframes from session state based on the groups_info mapping.
    
    Parameters:
    - df: The main dataframe to check for column presence.
    - groups_info: A dictionary mapping group names to session state keys and column names.
    
    Returns:
    - A dictionary of groups and their corresponding dataframes.
    """
    group_dfs = {'All Employees': df}
    
    for group_name, (session_key, column_name) in groups_info.items():
        if column_name in df.columns:
            if session_key in st.session_state:
                group_dfs[group_name] = st.session_state[session_key]
            else:
                st.error(f"Data for {group_name.lower()} not found. Please run the Data Preprocessing step first.")
                
    return group_dfs

#################################################################################
# Demographic
#################################################################################
def plot_pie_chart(df, column_name, chart_title):
    """
    Generates and displays a pie chart for the specified column in the DataFrame.
    
    Parameters:
    - df: The DataFrame containing the data.
    - column_name: The name of the column to visualize.
    - chart_title: The title of the chart.
    """
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Exclude null values before calculating value counts and percentages
        filtered_df = df.dropna(subset=[column_name])
        # Calculate value counts and percentages
        counts = filtered_df[column_name].value_counts(normalize=True) * 100
        # Generate and display the pie chart
        fig = px.pie(values=counts.values, names=counts.index, title=chart_title)
        st.plotly_chart(fig)
    else:
        st.error(f"Column '{column_name}' not found in the DataFrame. Please check the column name and try again.")


def plot_horizontal_bar_chart(df, column_name, chart_title, xaxis_title, yaxis_title):
    """
    Generates and displays a horizontal bar chart for the specified column in the DataFrame.
    
    Parameters:
    - df: The DataFrame containing the data.
    - column_name: The name of the column to visualize.
    - chart_title: The title of the chart.
    - xaxis_title: The title of the x-axis.
    - yaxis_title: The title of the y-axis.
    """
    # Exclude null values before calculating value counts and percentages
    filtered_df = df.dropna(subset=[column_name])
    # Count the occurrences and calculate percentages
    counts = filtered_df[column_name].value_counts()
    total_responses = len(filtered_df[column_name])
    percentages = (counts / total_responses) * 100

    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        yaxis_title: counts.index,
        'Percentage': percentages
    })

    # Create the horizontal bar chart using Plotly
    fig = px.bar(data, x='Percentage', y=yaxis_title, text='Percentage', orientation='h')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        title_text=chart_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )

    # Show the figure
    st.plotly_chart(fig)

#################################################################################
# Inclusion
#################################################################################
def plot_group_responses(selected_question, response_categories, group_dfs, colors):
    """
    Plot responses for a selected question across different employee groups.
    
    Parameters:
    - selected_question (str): The question to visualize responses for.
    - response_categories (list): A list of possible response categories for the question.
    - group_dfs (dict): A dictionary of group names and their corresponding dataframes.
    - colors (dict): A dictionary mapping response categories to colors.
    """
    data = []
    for response in response_categories:
        group_values = []
        for group_name, group_df in group_dfs.items():
            # Exclude null values before calculating value counts and percentages
            group_filtered_df = group_df.dropna(subset=[selected_question])
            group_counts = group_filtered_df[selected_question].value_counts(normalize=True) * 100
            group_values.append(group_counts.get(response, 0))

        data.append(go.Bar(
            name=response,
            x=list(group_dfs.keys()),
            y=group_values,
            marker_color=colors[response]
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        title=f'{selected_question}',
        xaxis_title='Employee Group',
        yaxis_title='Percentage',
        legend_title='Response Category'
    )

    st.plotly_chart(fig)





def show_question_mapping_interface(actual_questions, standard_questions):
    for question in actual_questions:
        # Use session_state to store the mapping for each question
        st.session_state['mappings'][question] = st.selectbox(
            question,
            options=[""] + standard_questions,  # Add blank option
            index=0,  # Default to blank option
            format_func=lambda x: x if x else "Select...",
            key=question  # Use question as the unique key
        )


def average_salary(salary_range):
    if 'prefer not to say' in salary_range.lower():
        return 0
    if 'or more' in salary_range.lower():
        # If the range says 'or more', use the single value as the average
        min_salary = salary_range.lower().replace('£', '').replace(' or more', '').replace(',', '')
        return int(min_salary)
    try:
        min_salary, max_salary = salary_range.replace('£', '').replace(',', '').split('to')
        return round((int(min_salary) + int(max_salary)) / 2)
    except ValueError:
        # Log the error value and return None or some default
        st.write(f"Cannot convert {salary_range}")
        return None
  
