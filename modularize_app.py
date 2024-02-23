import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#-------------------------------------
import numpy as np
import re

import torch
import torchvision
from torch import cuda

from huggingface_hub import login
from torch import bfloat16

import transformers
from sentence_transformers import SentenceTransformer
from umap import UMAP
# from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic import BERTopic

# Text summarization
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain import PromptTemplate,  LLMChain
#----------------------------------------------
import base64
#----------------------------------------------
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

import spacy 
#---------------------------------------------------------
from text_analysis_module import process_text_data, perform_topic_modeling, summarize_text, display_summary, convert_df_to_csv_download_link, display_topics_and_docs, get_key_for_value, replace_named_entities

from visualization_module import create_sub_dataframes, update_group_dfs, plot_pie_chart, plot_horizontal_bar_chart, plot_group_responses, show_question_mapping_interface, average_salary



# Initialize session state for the DataFrame
if 'df' not in st.session_state:
    st.session_state['df'] = None
# # Initialize session state for the DataFrame
# if 'df' not in st.session_state or st.session_state['df'] is None:
#     st.session_state['df'] = pd.DataFrame()

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Questions", "Demographic Analysis", "Social Mobility Analysis", "Inclusion Analysis", "Text Analysis"])


# # Initialize the Presidio Analyzer and Anonymizer engines
# analyzer = AnalyzerEngine()
# anonymizer = AnonymizerEngine()

# def anonymize_text(text):
#     if isinstance(text, str):  # Check if text is a string
#         # Use the analyzer to find personal information in the text
#         analyzer_results = analyzer.analyze(text=text, language='en')
        
#         # Use the anonymizer to anonymize the text based on the analyzer results
#         anonymized_result = anonymizer.anonymize(
#             text=text,
#             analyzer_results=analyzer_results,
#             operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"})}
#         )
        
#         return anonymized_result.text
#     else:
#         return text


# # Load a pre-trained spaCy model
# nlp = spacy.load("en_core_web_lg")

# def replace_named_entities(text, replace_with="[REPLACED]"):
#     if isinstance(text, str):  # Check if text is a string
#         # Process the text with the spaCy model
#         doc = nlp(text)
        
#         # Replace named entities with the specified placeholder
#         replaced_text = []
#         for token in doc:
#             if token.ent_type_ != "":
#                 replaced_text.append(replace_with)
#             else:
#                 replaced_text.append(token.text)
        
#         # Join the tokens back into a string
#         replaced_text = " ".join(replaced_text)
        
#         return replaced_text
#     else:
#         return text


# Define a mapping of group names to their corresponding session state keys and column names
groups_info = {
    'Caring Responsibilities': ('df_caring_responsibilities', 'Has_Caring_Responsibility'),
    'Religious Beliefs': ('df_religious_beliefs', 'Religion'),
    'Minority Ethnicity': ('df_minority_ethnicity', 'Ethnicity'),
    'Women': ('df_women', 'Gender'),
    'Disabilities': ('df_disabilities', 'Has_Disability'),
    'LGBT': ('df_LGBT', 'LGBT'),
    'Mental Health': ('df_mental_health', 'has_mental_health')
}
#################################################################################
# Demographic
#################################################################################




#################################################################################
# Inclusion
#################################################################################

    ################################################################################

rename_columns = {'How long have you worked at the organization?': 'Service_Length',
                'What is your annual salary':'Salary',
                'Which part of the business do you work in?': 'Organization_Part',
                'Which function do you work in?': 'Organization_Function',
                'Which grade do you work at?': 'Grade',
                'Which brand do you work in?': 'Brand',
                'What is your role in the organisation?':'Organisational_Role',
                'Which department do you work in?': 'Department',
                'Job Share': 'Job_Share',
                'Flexibility with start and finish times': 'Flexibility_with_start_and_finish_times',
                'Working from home': 'Working_from_home',
                'Flexible Hours Based on Output': 'Flexible_Hours_Based_on_Output',
                'Remote Working': 'Remote_Working',
                'Condensed Hours': 'Condensed_Hours',
                'School Hours': 'School_Hours',
                'Term Time': 'Term_Time',
                'None of the above': 'None_of_the_above',##
                'Prefer not to say': 'PNTS',###
                'Flexible Working Comments': 'Flexible_Working_Comments',
                'What is your age?': 'Age',
                'What generation were you born into?': 'Generation',
                'What best describes your gender?': 'Gender',
                'Prefer to self describe your gender': 'Self_Describe_Gender',  
                'Is your gender identity the same as the sex you were assigned at birth?': 'Gender_Identification_Same_as_Birth',
                'Do you have a disability or long term health condition?': 'Disability_or_Long_Term_Health_Condition?',
                'Do you have difficulty seeing, even if wearing glasses?': 'Seeing_Dificulty',
                'Do you have difficulty hearing, even if using a hearing aid?': 'Hearing_Dificulty',
                'Do you have difficulty walking or climbing steps?': 'Walking_Dificulty',
                'Do you have difficulty remembering or concentrating?': 'Remembering_Dificulty',
                'Do you have difficulty (with self-care such as) washing all over or dressing?': 'SelfCare_Dificulty',
                'Using your usual language, do you have difficulty communicating?': 'Communicating_Dificulty',
                'Do you have difficulty raising a 2 litre bottle of water or soda from waist to eye level?': 'Raising_Water/Soda_Bottle_Dificulty',
                'Do you have difficulty using your hands and fingers?': 'Picking_Up_Small_Objects_Dificulty',
                'Elaborate on how your difficulty above affect you at work': 'Difficulty_Comment',
                'Do you consider yourself to be neuro-divergent':'Do you consider yourself to be neuro-divergent?',
                'Is your work schedule or work tasks arranged to account for difficulties you have in doing certain activities?': 'Work_Adaptation_for_Difficulties',
                'Has your workplace been modified to account for difficulties you have in doing certain activities?': 'Workplace_Modification_for_Difficulties',
                'Would you describe your national identity as English?': 'English',
                'Would you describe your national identity as Welsh?': 'Welsh',
                'Would you describe your national identity as Scottish?': 'Scottish',
                'Would you describe your national identity as Northern Irish?': 'Northern_Irish',
                'Would you describe your national identity as British?': 'British',
                'Would you prefer not to say your national identity?': 'Prefer_Not_To_Say',
                'would you describe your national identity yourself?': 'National_Identity__Not_in_List',
                'What is your ethnicity?': 'Ethnicity',
                'Please tell us your ethnicity below if you do not personally identify with any of the options above.': 'Ethnicity_Not_in_List',
                'What is your first language?':'Language',
                'What is your sexual orientation?': 'Sexual_Orientation',
                'Prefer to self describe your sexual orientation': 'Self_Describe_Sexual_Orientation',
                'Are you open about your sexual orientation "At home"?': 'Sexual_Orientation_Openness_At_Home',
                'Are you open about your sexual orientation "With your manager"?': 'Sexual_Orientation_Openness_With_Manager',
                'Are you open about your sexual orientation "With colleagues"?': 'Sexual_Orientation_Openness_With_Colleagues',
                'Are you open about your sexual orientation "At work generally"?': 'Sexual_Orientation_Openness_At_Work_Generally',
                'Are you open about your sexual orientation "Prefer not to say"?': 'Sexual_Orientation_Openness_PNTS',
                'Do you have any dependants or caring responsibilities of "Nobody"?': 'Has_Caring_Responsibility',
                'Do you have any dependants or caring responsibilities of "a child/children (under 18)"?': 'Dependents_Children_Under_18',
                'Do you have any dependants or caring responsibilities of "a disabled child/children (under 18)"?': 'Dependents_Disabled_Children_Under_18',
                'Do you have any dependants or caring responsibilities of "of a disabled adult (18 and over)"?': 'Dependents_Disabled_Adult_18_and_Over',
                'Do you have any dependants or caring responsibilities of "an older person/people (65 and over)"?': 'Dependents_Older_Person_65_and_Over',
                'Do you have any dependants or caring responsibilities "Prefer not to say"?': 'Dependents_Prefer_Not_to_Say', ##
                'What is your religion?': 'Religion',
                'Please tell us your religion below if you do not personally identify with any of the options above.': 'Religion_Not_in_List',
                'Are you serving or have you ever served in the Armed Forces': 'Armed_Forces_Service',
                'How often do you feel worried, nervous or anxious?': 'How_often_feeling_worried_nervous_anxious',
                'Thinking about the last time you felt worried, nervous or anxious, how would you describe the level of these feelings?':'Level_of_last_worrying_anxiety_nervousness',
                'How often do you feel depressed?': 'How_often_feeling_depressed',
                'Thinking about the last time you felt depressed, how depressed did you feel?': 'Level_of_last_depression',
                'What was the occupation of your main household earner when you were aged about 14?': 'Main_Earner_Occupation_At_14',
                'Did the main household earner in your house work as an employee or were they self employed when you were aged about 14?': 'Main_Earner_Employee_or_SelfEmployed_At_14',######
                'Which type of school did you attend for most of the time between the ages of 11 and 16?': 'School_Type_Ages_11_16',
                'If you finished school after 1980, were you eligible for Free School Meals at any point during your school years?': 'Eligibility_For_Free_School_Meals',
                'Did either of your parents attend university by the time you were 18?': 'Parents_University_Attendance_By_18',
                'What is the highest level of qualification achieved by either of your parent(s) by the time you were 18?': 'Parents_Highest_Level_of_Qualification_By_18',######
                'Compared to people in general in the UK, would you describe yourself as coming from a lower socio-economic background?': 'Lower_Socio_Economic_Background',##########
                'How much of a priority is EDI in the company to "The senior leadership team"?': 'EDI_Priority_Senior_Leadership',
                'How much of a priority is EDI in the company to "Your line manager"?': 'EDI_Priority_Line_Manager',
                'How much of a priority is EDI in the company to "Your peers "?': 'EDI_Priority_Peers',
                'How much of a priority is EDI in the company to "Yourself "?': 'EDI_Priority_YourSelf',
                'What advice do you have for the Senior Leadership Team regarding EDI at this company?': 'Advice_for_Senior_Leadership_Team_re_EDI',
                'I feel like I belong at this organisation': 'I feel like I belong at business',
                'It feels like I do not belong at the business when something negative happens to me at work': 'I feel that I might not belong at business when something negative happens to me at work',
                'I can voice a contrary opinion without fear of negative consequences': 'I can voice a contrary opinion without fear of negative consequences',
                'I often worry I do not have things in common with others at the company': 'I often worry I do not have things in common with others at business',
                'I feel like my colleagues understand who I really am': 'I feel like my colleagues understand who I really am',
                'I feel respected and valued by my colleagues at the company': 'I feel respected and valued by my colleagues at business',
                'I feel respected and valued by my manager': 'I feel respected and valued by my manager',
                'I feel confident I can develop my career at the company': 'I feel confident I can develop my career at at business',
                'When I speak up at work, my opinion is valued': 'When I speak up at work, my opinion is valued',
                'Admin or routine daily tasks that don’t have a specific owner are divided fairly at the company': 'Administrative tasks that don’t have a specific owner, are divided fairly at business',
                'Promotion decisions are fair at the company': 'Promotion decisions are fair at business',
                'My job performance is assessed fairly': 'My job performance is evaluated fairly',
                'The company believes that people can always greatly improve their talents and abilities': 'Business believes that people can always improve their talents and abilities',
                'The company believes that people have a certain amount of talent, and they can’t do much to change it': 'Business believes that people have a certain amount of talent, and they can’t do much to change it',
                'Working here is important to the way that I think of myself as a person': 'Working at Business is important to the way that I think of myself as a person',
                'The information and resources I need to do my job effectively are readily available': 'The information and resources I need to do my job effectively are available',
                'The company hires people from diverse backgrounds': 'Business hires people from diverse backgrounds',
                'Which of the following statements best describes how you feel in your team': 'Which of the following statements best describes how you feel in your team',
                'Is there anything this organisation can do to recruit a more diverse group of employees?': 'What ONE thing do you think the business should be doing to recruit a diverse range of employees?',
                'What things do you think the organisation does well in terms of creating a diverse and inclusive workplace?': 'What ONE thing do you think the business does well in terms of creating a diverse and inclusive workplace?',
                'What things do you think the organisation should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?': 'What ONE thing do you think the business should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?',
                'In what ways can this organisation ensure that everyone is treated fairly and can thrive?': 'In what ways can this organisation ensure that everyone is treated fairly and can thrive?', ## Is it the same as above question???
                'We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?': 'We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?',
                'Would you agree that you are able to reach your full potential at work?': 'Would you agree that you are able to reach your full potential at work?',
                'How likely is it that you would recommend this company as an inclusive place to work to a friend or colleague?': 'How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?',
                'In the next 6 months, are you considering leaving this organisation because you do not feel respected or that you belong?': 'Considering_Leaveing_in_Next_6_Months',
                'Are there any other areas related to inclusion and diversity where you feel this organisation could do better, that have not been covered here?': 'What other comments would you like to make in relation to D&I at this organisation?'

                }

standard_questions = ['How long have you worked at the organization?',
                    'What is your annual salary',
                    'Which department do you work in?',
                    'Which part of the business do you work in?',
                    'Which function do you work in?',
                    'Which grade do you work at?',
                    'Which brand do you work in?',
                    'What is your role in the organisation?',
                    'Job Share',
                    'Flexibility with start and finish times',
                    'Working from home',
                    'Flexible Hours Based on Output',
                    'Remote Working',
                    'Condensed Hours',
                    'School Hours',
                    'Term Time',
                    'None of the above',
                    'Prefer not to say',###
                    'Flexible Working Comments',
                    'What is your age?',
                    'What generation were you born into?',
                    'What best describes your gender?',
                    'Prefer to self describe your gender',
                    'Is your gender identity the same as the sex you were assigned at birth?',
                    'Do you have a disability or long term health condition?',
                    'Do you have difficulty seeing, even if wearing glasses?',
                    'Do you have difficulty hearing, even if using a hearing aid?',
                    'Do you have difficulty walking or climbing steps?',
                    'Do you have difficulty remembering or concentrating?',
                    'Do you have difficulty (with self-care such as) washing all over or dressing?',
                    'Using your usual language, do you have difficulty communicating?',
                    'Do you have difficulty raising a 2 litre bottle of water or soda from waist to eye level?',
                    'Do you have difficulty using your hands and fingers?',
                    'Elaborate on how your difficulty above affect you at work',
                    'Do you consider yourself to be neuro-divergent',
                    'Is your work schedule or work tasks arranged to account for difficulties you have in doing certain activities?',
                    'Has your workplace been modified to account for difficulties you have in doing certain activities?',
                    'Would you describe your national identity as English?',
                    'Would you describe your national identity as Welsh?',
                    'Would you describe your national identity as Scottish?',
                    'Would you describe your national identity as Northern Irish?',
                    'Would you describe your national identity as British?',
                    'Would you prefer not to say your national identity?',
                    'would you describe your national identity yourself?',
                    'What is your ethnicity?',
                    'Please tell us your ethnicity below if you do not personally identify with any of the options above.',
                    'What is your first language?',
                    'What is your sexual orientation?',
                    'Prefer to self describe your sexual orientation',
                    'Are you open about your sexual orientation "At home"?',
                    'Are you open about your sexual orientation "With your manager"?',
                    'Are you open about your sexual orientation "With colleagues"?',
                    'Are you open about your sexual orientation "At work generally"?',
                    'Are you open about your sexual orientation "Prefer not to say"?',
                    'Do you have any dependants or caring responsibilities of "Nobody"?',##
                    'Do you have any dependants or caring responsibilities of "a child/children (under 18)"?',
                    'Do you have any dependants or caring responsibilities of "a disabled child/children (under 18)"?',
                    'Do you have any dependants or caring responsibilities of "of a disabled adult (18 and over)"?',
                    'Do you have any dependants or caring responsibilities of "an older person/people (65 and over)"?',
                    'Do you have any dependants or caring responsibilities "Prefer not to say"?', ##
                    'What is your religion?',
                    'Please tell us your religion below if you do not personally identify with any of the options above.',
                    'Are you serving or have you ever served in the Armed Forces',
                    'How often do you feel worried, nervous or anxious?',
                    'Thinking about the last time you felt worried, nervous or anxious, how would you describe the level of these feelings?',
                    'How often do you feel depressed?',
                    'Thinking about the last time you felt depressed, how depressed did you feel?',
                    'What was the occupation of your main household earner when you were aged about 14?',
                    'Did the main household earner in your house work as an employee or were they self employed when you were aged about 14?',
                    'Which type of school did you attend for most of the time between the ages of 11 and 16?',
                    'If you finished school after 1980, were you eligible for Free School Meals at any point during your school years?',
                    'Did either of your parents attend university by the time you were 18?',
                    'What is the highest level of qualification achieved by either of your parent(s) by the time you were 18?',
                    'Compared to people in general in the UK, would you describe yourself as coming from a lower socio-economic background?',
                    'How much of a priority is EDI in the company to "The senior leadership team"?',
                    'How much of a priority is EDI in the company to "Your line manager"?',
                    'How much of a priority is EDI in the company to "Your peers "?',
                    'How much of a priority is EDI in the company to "Yourself "?',
                    'What advice do you have for the Senior Leadership Team regarding EDI at this company?',
                    'I feel like I belong at this organisation',
                    'It feels like I do not belong at the business when something negative happens to me at work',
                    'I can voice a contrary opinion without fear of negative consequences',
                    'I often worry I do not have things in common with others at the company',
                    'I feel like my colleagues understand who I really am',
                    'I feel respected and valued by my colleagues at the company',
                    'I feel respected and valued by my manager',
                    'I feel confident I can develop my career at the company',
                    'When I speak up at work, my opinion is valued',
                    'Admin or routine daily tasks that don’t have a specific owner are divided fairly at the company',
                    'Promotion decisions are fair at the company',
                    'My job performance is assessed fairly',
                    'The company believes that people can always greatly improve their talents and abilities',
                    'The company believes that people have a certain amount of talent, and they can’t do much to change it',
                    'Working here is important to the way that I think of myself as a person',
                    'The information and resources I need to do my job effectively are readily available',
                    'The company hires people from diverse backgrounds',
                    'Which of the following statements best describes how you feel in your team',
                    'Is there anything this organisation can do to recruit a more diverse group of employees?',
                    'What things do you think the organisation does well in terms of creating a diverse and inclusive workplace?',
                    'What things do you think the organisation should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?',
                    'In what ways can this organisation ensure that everyone is treated fairly and can thrive?', ## Is it the same as above question???
                    'We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?',
                    'Would you agree that you are able to reach your full potential at work?',
                    'How likely is it that you would recommend this company as an inclusive place to work to a friend or colleague?',
                    'In the next 6 months, are you considering leaving this organisation because you do not feel respected or that you belong?',
                    'Are there any other areas related to inclusion and diversity where you feel this organisation could do better, that have not been covered here?'

                ]



if page == "Questions":
    st.header("Questions")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension.lower() == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension.lower() == 'xlsx':
            df = pd.read_excel(uploaded_file)

        st.session_state['df'] = df  # Save the main DataFrame in session state

    if st.checkbox('Please indicate if this is the first time you\'re uploading this dataset by checking the box below.'):
        
        if uploaded_file is None:
            st.write("Upload a file.")
        else:
            if 'df' in st.session_state:  # Check if the main DataFrame is in session state
                df = st.session_state['df']  # Retrieve the main DataFrame from session state

                st.markdown(f"###### This dataset includes {df.shape[0]}" + " rows and " + f"{df.shape[1]}" + " columns")
                # st.markdown(f"#### Number of Columns: {df.shape[1]}")

            
                # Define all replacements in a dictionary for readability and maintainability
                replacements = {
                    'Artex': 'this organization',
                    'artex': 'this organization',
                    'Plan': 'this organization',
                    'plan': 'this organization',
                    'Plan UK': 'this organization',
                    'plan uk': 'this organization',
                    'Plan International': 'this organization',
                    'plan international': 'this organization'
                }

                # Replace in column names
                df.columns = [col for col in df.columns]
                for old, new in replacements.items():
                    df.columns = [col.replace(old, new) for col in df.columns]

                # Replace in cell values
                def replace_values(x):
                    if isinstance(x, str):
                        for old, new in replacements.items():
                            x = x.replace(old, new)
                    return x

                df = df.applymap(replace_values)

                # Initialize an empty list for the new column names
                new_column_names = []
                previous_question = None

                # Iterate through each column name in the first row
                for col in df.columns:
                    if "Unnamed" not in col:
                        new_column_names.append(col)
                        previous_question = col
                        # Assume 'df' is your DataFrame and 'column_name' is the name of the column
                        idx = df.columns.get_loc(previous_question)

                    else:
                        # print(new_column_name)
                        sub_question = df.at[0, col]
                        sub_question = str(sub_question).strip()  # Ensure any string is stripped of whitespace
                        if  sub_question != '':
                            new_column_name = f"{previous_question}: {sub_question}"
                        else:
                            new_column_name = f"{previous_question}: Subpart"
                        new_column_names.append(new_column_name)
                        new_column_names[idx] = f"{previous_question}: {df.iloc[0, idx]}"

                
                if any('Unnamed' in column for column in df.columns):
                # Drop the first two rows that were used to identify the column names
                    df = df.drop(df.index[0:2]).reset_index(drop=True)

                # Now assign the new column names to the DataFrame
                df.columns = new_column_names

                st.session_state['df'] = df


            # Save the processed DataFrame to the session state
            # st.session_state['df'] = df
            # Display basic information about the data########################## show her or after preprocessing?
            # st.write("Data Overview:")
            # st.write(f"Number of Rows: {df.shape[0]}")
            # st.write(f"Number of Columns: {df.shape[1]}")
            # st.text("A brief explanation about the dataset...")  # You can change this later


                # rename_columns = {}
                

                # standard_questions = []

                # Initialize or retrieve mappings from session state
                if 'mappings' not in st.session_state:
                    st.session_state['mappings'] = {}


                st.markdown("#### Select the standard format for the questions:")
                actual_questions = df.columns.tolist()
                show_question_mapping_interface(actual_questions, standard_questions)

                # Button to finalize mappings and rename columns
                if st.button('Finalize Standard Formats'):
                    # Create a reverse mapping from selected standard questions to new column names
                    selected_standard_to_new = {
                        question: rename_columns[st.session_state['mappings'][question]]
                        # st.session_state['mappings'][question]: rename_columns[st.session_state['mappings'][question]] #[question]
                        for question in actual_questions
                        # if question in  st.session_state['mappings'].keys()

                        if st.session_state['mappings'][question] in rename_columns.keys()
                    }

                    # Apply the renaming based on the reverse mapping created above
                    df.rename(columns=selected_standard_to_new, inplace=True)

                    # Update the DataFrame in session state
                    st.session_state['df'] = df
                    st.session_state['question_mapping'] = selected_standard_to_new  # Save the mappings
                    st.success("Columns have been renamed to standard formats.")




                if 'df' in st.session_state:
                    df = st.session_state['df']

                    # Filling missing values as per the specified instructions
                    #if 'What other comments would you like to make in relation to D&I at this organisation?' in df.columns:
                        #df['What other comments would you like to make in relation to D&I at this organisation?'].fillna('No response', inplace=True)
                    columns_to_fillna = [
                    'Self_Describe_Gender', 'Self_Describe_Sexual_Orientation', 'Difficulty_Comment', 'Flexible_Working_Comments', 
                    'Advice_for_Senior_Leadership_Team_re_EDI', 'Religion_Not_in_List', 'Ethnicity_Not_in_List', 'National_Identity__Not_in_List',
                    'What other comments would you like to make in relation to D&I at this organisation?',
                    'What ONE thing do you think the business should be doing to recruit a diverse range of employees?',
                    'What ONE thing do you think the business does well in terms of creating a diverse and inclusive workplace?',
                    'What ONE thing do you think the business should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?',
                    'In what ways can this organisation ensure that everyone is treated fairly and can thrive?', ## Is it the same as above question??? 
                    'We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?'
                    ]

                    for col in columns_to_fillna:
                        if col in df.columns:
                            df[col].fillna('No response', inplace=True)

                    # string_columns = []
                    # for item in columns_to_fillna:
                    #     if item in df.columns:
                    #         string_columns.append(item)
                    # df[string_columns] = df[string_columns].astype(str)
                    # # Apply the anonymization functions to string columns
                    # df[string_columns] = df[string_columns].applymap(anonymize_text)
                    # df[string_columns] = df[string_columns].applymap(replace_named_entities)
                                     

                if 'Salary' in df.columns:
                    # Assuming the second column is named 'Salary_Range' and is formatted like '£30,000-£40,000'
                    # Create a new column 'Average_Salary' by splitting the 'Salary_Range', converting to integers, and taking the average
                    # Define a function to handle salary range conversion and averaging
                    df['Salary'] = df['Salary'].astype('category')
                  

                    # Apply the function to the 'Salary_Range' column
                    df['Average_Salary'] = df['Salary'].apply(average_salary)
                    # Update the DataFrame in session state
                    st.session_state['df'] = df

                if 'Religion' in df.columns:
                    df['Religion'].replace({'Christian (including Church of England, Catholic, Protestant and all other Christian denominations)': 'Christian'}, inplace=True)

                # Replacement in Parents_University_Attendance_By_18 column of the df DataFrame
                if 'Parents_University_Attendance_By_18' in df.columns:
                    df['Parents_University_Attendance_By_18'] = df['Parents_University_Attendance_By_18'].replace({'No, neither of my parents attended university': 'No, neither of my parents', 'Yes, one or both of my parents attended university': 'Yes, one or both'})

                mapping_dict = {
                    'I feel like a key component of my team with real influence over decisions': 'Key Component',
                    'I have some influence over decisions, but do not feel like a key component of my team': 'Some Influence',
                    'I feel safe voicing my views and opinions, but I have little influence over decisions': 'Safe Voicing',
                    'I am noticed by some people, but do not feel safe voicing my opinions': 'Unsafe Voicing',
                    'I am generally ignored by others': 'Ignored by Others',
                    'Prefer not to say': 'PNTS'
                }

                # Apply the mapping to create a new column with short descriptions
                if 'Which of the following statements best describes how you feel in your team' in df.columns:
                    df['Which of the following statements best describes how you feel in your team'] = df['Which of the following statements best describes how you feel in your team'].map(mapping_dict)


                # Apply the mapping to create a new column with short descriptions
                #mapping_dict1 = {
                #'Senior, middle or junior managers or administrators such as: finance manager, chief executive, large business owner, office manager, retail manager, bank manager, restaurant manager, warehouse manager.': 'Managers/Administrators',
                #'Long-term unemployed (claimed Jobseeker’s Allowance or earlier unemployment benefit for more than a year)': 'Long-term Unemployed',
                #'Technical and craft occupations such as: motor mechanic, plumber, printer, electrician, gardener, train driver.': 'Technical/Crafts',
                #'Routine, semi-routine manual and service occupations such as: postal worker, machine operative, security guard, caretaker, farm worker, catering assistant, sales assistant, HGV driver, cleaner, porter, packer, labourer, waiter/waitress, bar staff.': 'Manual and Service',
                #'Clerical and intermediate occupations such as: secretary, personal assistant, call centre agent, clerical worker, nursery nurse.': 'Clerical/Intermediate',
                #'Small business owners who employed less than 25 people such as: corner shop owners, small plumbing companies, retail shop owner, single restaurant or cafe owner, taxi owner, garage owner': 'Small Business Owners',
                #'Modern professional & traditional professional occupations such as: teacher, nurse, physiotherapist, social worker, musician, police officer (sergeant or above), software designer, accountant, solicitor, medical practitioner, scientist, civil / mechanical engineer.': 'Professionals',
                #'This question does not apply to me': 'Not Applicable',
                #'Prefer not to say': 'Prefer not to say'
                #}
                #if 'Main_Earner_Occupation_At_14' in df.columns:
                    #df['Main_Earner_Occupation_At_14'] = df['Main_Earner_Occupation_At_14'].map(mapping_dict1)

                # Filling missing values as per the specified instructions
                #if 'What other comments would you like to make in relation to D&I at this organisation?' in df.columns:
                    #df['What other comments would you like to make in relation to D&I at this organisation?'].fillna('No response', inplace=True)

                # # For simplicity, let's assume that columns with less than 20 unique values can be treated as categorical
                #categorical_columns = [col for col in df.columns if df[col].nunique() < 20]

                # # Converting these columns to 'category' data type
                #for col in categorical_columns:
                #df[col] = df[col].astype('category')
                categorical_columns = []
                for col in df.columns:
                    try:
                        if df[col].nunique() < 20:
                            categorical_columns.append(col) 
                    except Exception as e:
                        print(f"Error processing column '{col}': {e}")



                st.write("Download the cleaned data using the link provided to save time and avoid reprocessing this data in the future.")
                csv_filename = f"{uploaded_file.name.split('.')[-2]}_Processed.csv"
                df.to_csv(csv_filename, encoding='utf-8', index=False)

                # Download button for the file
                st.markdown(convert_df_to_csv_download_link(df, csv_filename), unsafe_allow_html=True)



                #******************************** New created sub dataframes *************************************#
                # Call the function to created sub dataframes 
                if 'df' in st.session_state and st.session_state['df'] is not None:
                    df_mental_health, df_LGBT, df_disabilities, df_women, df_minority_ethnicity, df_religious_beliefs, df_caring_responsibilities = create_sub_dataframes(df)

                #**********************************************************************************************
                # if st.checkbox('Show processed data'):
                #         # st.write(df.head())
                #         st.dataframe(df) 
                # Update the DataFrame in session state
                st.session_state['df'] = df


    if st.checkbox('Tick this box if you have already processed this dataset.'):
        if 'df' in st.session_state:  # Check if the main DataFrame is in session state
                df = st.session_state['df']  # Retrieve the main DataFrame from session state
                # if df.columns.tolist() in rename_columns.values():
                # if all(col in rename_columns.values() for col in df.columns):
                # if 'Average_Salary' in df.columns:
                #     df_drop_Avaerage_Salary = df.drop('Average_Salary', axis=1)
                # else:
                #     df_drop_Avaerage_Salary = df
                df_drop_Avaerage_Salary = df[df.columns.intersection(rename_columns.values())]
                if df_drop_Avaerage_Salary.columns.tolist()!=[] and all(col in rename_columns.values() for col in df_drop_Avaerage_Salary.columns):

                    #******************************** New created sub dataframes *************************************#
                    # Call the function to created sub dataframes 
                    if 'df' in st.session_state and st.session_state['df'] is not None:
                        df_mental_health, df_LGBT, df_disabilities, df_women, df_minority_ethnicity, df_religious_beliefs, df_caring_responsibilities = create_sub_dataframes(df)
                    #**********************************************************************************************
                        
                    if st.checkbox('Show data'):
                        # st.write(df.head())
                        st.dataframe(df) 
                    # Update the DataFrame in session state
                    st.write("You can now navigate between pages.")
                    st.session_state['df'] = df

                else:
                    st.write("It appears that the dataset has not been processed correctly. Please ensure it is processed again.")






# # Page content
elif page == "Demographic Analysis":
# if page == "Demographic Analysis":
    st.header("Demographic Analysis")
    st.subheader("Please select the desired analysis from the sidebar on the right.")
    
    # Ensure the data is loaded
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']

        # Define the options for the select slider with corresponding column names
        visualization_options = {
            'Language': ['Language'],
            'Neuro divergent': ['Do you consider yourself to be neuro-divergent?'],
            'Work Adaptation for Difficulties': ['Work_Adaptation_for_Difficulties'],
            'Workplace Modification for Difficulties': ['Workplace_Modification_for_Difficulties'],
            'Length of Service': ['Service_Length'],
            'Salary': ['Salary'],
            'Seniority in the Organisation': ['Organisational_Role'],
            'Departments': ['Department'],
            'Different Parts in Organization': ['Organization_Part'],
            'Function in Organization': ['Organization_Function'],
            'Grade in Organization': ['Grade'],
            'Brand in Organization': ['Brand'],
            'Types of Flexible Working': ['Job_Share', 'Flexibility_with_start_and_finish_times', 'Working_from_home', 'Flexible_Hours_Based_on_Output',
                                            'Remote_Working', 'Condensed_Hours', 'School_Hours', 'Term_Time', 'None_of_the_above', 'PNTS'],
            'Age Distribution': ['Age'],
            'Generation Distribution': ['Generation'],
            'Gender and gender identity': ['Gender'],
            'Disability or long-term health conditions': ['Disability_or_Long_Term_Health_Condition'],
            'Disability and long-term health conditions': ['Seeing_Dificulty', 'Hearing_Dificulty', 'Walking_Dificulty', 'Remembering_Dificulty', 'SelfCare_Dificulty',
                                                            'Communicating_Dificulty', 'Raising_Water/Soda_Bottle_Dificulty', 'Picking_Up_Small_Objects_Dificulty'],
            'Mental health': ['How_often_feeling_worried_nervous_anxious', 'Level_of_last_worrying_anxiety_nervousness', 'How_often_feeling_depressed', 'Level_of_last_depression'],
            'Nationality': ['English','Welsh', 'Scottish', 'Northern_Irish', 'British', 'Prefer_Not_To_Say'], #, 'Other, please describe'],
            'Ethnicity': ['Ethnicity'],
            'Sexuality': ['Sexual_Orientation'],
            'Caring responsibilities': ['Has_Caring_Responsibility','Dependents_Children_Under_18','Dependents_Disabled_Children_Under_18','Dependents_Disabled_Adult_18_and_Over','Dependents_Older_Person_65_and_Over','Dependents_Prefer_Not_to_Say'],
            'Religion': ['Religion'],
            'Gender and Ethnicity Distribution Chart': ['Gender', 'Ethnicity'],
            'Gender and Caring Responsibility Distribution': ['Gender', 'Has_Caring_Responsibility'],
            'Service Length amongst LGBT+ Distribution': ['Service_Length', 'LgbT'],
            'Service Length amongst Disabled Employees': ['Service_Length', 'Has_Disability'],
            'Use of Flexible Working Options by Employees with Caring Responsibilities, Disabilities, Different Genders': ['Gender', 'Has_Caring_Responsibility', 'Has_Disability',
                                                                                                                            'Job_Share', 'Flexibility_with_start_and_finish_times', 'Working_from_home', 'Flexible_Hours_Based_on_Output',
                                                                                                                            'Remote_Working', 'Condensed_Hours', 'School_Hours', 'Term_Time', 'None_of_the_above', 'PNTS'],
            'Mental Health Status Distribution by Department': ['Department', 'has_mental_health']
            # 'Optional Filtering': ['Service_Length']
        }

        # Filter options based on the columns present in the DataFrame
        filtered_options = {
                            key: value
                            for key, value in visualization_options.items()
                            if all(col in df.columns for col in value)
                        }

        # Sidebar for selecting visualizations
        selected_visualizations = st.sidebar.multiselect(
            "Select Visualizations to Display",
            filtered_options.keys(),
            default=next(iter(filtered_options.keys()), None) # Default to the first available option or None
        )
        # Add 'Optional Filtering' to the sidbar
        selected_visualizations.append('Optional Filtering')

        # Assuming df is your main DataFrame
        group_dfs = update_group_dfs(df, groups_info)




        # Analysis and Visualization
        # Sample Plotly Visualization

        # Function to create and show the visualizations
        def show_visualization(visualization_key):
            ######################################################################
            #                         length_of_service                          #
            ######################################################################
            if visualization_key == "Length of Service":
                column_name, chart_title = 'Service_Length', 'Length of Service at Company'
                plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #                                language                            #
            ######################################################################
            elif visualization_key == "Language":
                plot_horizontal_bar_chart(df, 'Language', 'Language Distribution', 'Percentage', 'Language')
                pass
            ######################################################################
            #                           neuro-divergent                          #
            ######################################################################
            elif visualization_key == "Neuro divergent":
                column_name, chart_title = 'Do you consider yourself to be neuro-divergent?', 'Do you consider yourself to be neuro-divergent?'
                plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #                  Work_Adaptation_for_Difficulties                  #
            ######################################################################
            elif visualization_key == "Work Adaptation for Difficulties":
                column_name, chart_title = 'Work_Adaptation_for_Difficulties', 'Is your work schedule or work tasks arranged to account for difficulties you have in doing certain activities?'
                plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #             Workplace_Modification_for_Difficulties                #
            ######################################################################
            elif visualization_key == "Workplace Modification for Difficulties":
                column_name, chart_title = 'Workplace_Modification_for_Difficulties', 'Has your workplace been modified to account for difficulties you have in doing certain activities?'
                plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #                                 Salary                             #
            ######################################################################
            elif visualization_key == "Salary":
                plot_horizontal_bar_chart(df, 'Salary', 'Salary of Employees', 'Percentage', 'Salary')
                pass
            ######################################################################
            #                           seniority_in_org                         #
            ######################################################################
            elif visualization_key == "Seniority in the Organisation":
                plot_horizontal_bar_chart(df, 'Organisational_Role', 'Seniority in the Organisation', 'Percentage', 'Role')
                pass
            ######################################################################
            #                               Department                           #
            ######################################################################
            elif visualization_key == "Departments":
                plot_horizontal_bar_chart(df, 'Department', 'Department Distribution in the Organisation', 'Percentage', 'Department')
                pass
            ######################################################################
            #                               Department                           #
            ######################################################################
            elif visualization_key == "Different Parts in Organization":
                plot_horizontal_bar_chart(df, 'Organization_Part', 'Different Parts Distribution in the Organisation', 'Percentage', 'Part')
                pass
            ######################################################################
            #                               Department                           #
            ######################################################################
            elif visualization_key == "Function in Organization":
                plot_horizontal_bar_chart(df, 'Organization_Function', 'Function Distribution in the Organisation', 'Percentage', 'Function')
                pass
            ######################################################################
            #                               Department                           #
            ######################################################################
            elif visualization_key == "Grade in Organization":
                plot_horizontal_bar_chart(df, 'Grade', 'Grade Distribution in the Organisation', 'Percentage', 'Grade')
                pass
            ######################################################################
            #                               Department                           #
            ######################################################################
            elif visualization_key == "Brand in Organization":
                plot_horizontal_bar_chart(df, 'Brand', 'Brand Distribution in the Organisation', 'Percentage', 'Brand')
                pass
            ######################################################################
            #               Types of flexlible working                           #
            ######################################################################

            elif visualization_key == "Types of Flexible Working":

                # Replace the following dictionary keys with your actual DataFrame column names.
                flexible_options = {
                    'Job Share': 'Job_Share',
                    'Flexibility with start and finish times': 'Flexibility_with_start_and_finish_times',
                    'Working from home': 'Working_from_home',
                    'Flexible Hours Based on Output': 'Flexible_Hours_Based_on_Output',
                    'Remote Working': 'Remote_Working',
                    'Condensed Hours': 'Condensed_Hours',
                    'School Hours': 'School_Hours',
                    'Term Time': 'Term_Time',
                    'None of the above': 'None_of_the_above',
                    'Prefer not to say': 'PNTS'
                }
                # Convert 'Yes' to 1 and 'No' to 0
                binary_data = df.replace({'Yes': 1, 'No': 0})
                # Convert the columns to numeric (if they are still categorical)
                for column in flexible_options.values():
                    binary_data[column] = pd.to_numeric(binary_data[column], errors='coerce')


                # Calculate the percentage of 'Yes' responses for each flexible working option
                flex_work_percentages = {}
                total_responses = len(df)

                for option, column_name in flexible_options.items():
                    flex_work_percentages[option] = binary_data[column_name].sum() / total_responses * 100

                # Create a DataFrame for plotting
                flex_work_df = pd.DataFrame(list(flex_work_percentages.items()), columns=['Flexible Option', 'Percentage'])

                # Create a horizontal bar chart using Plotly
                fig = px.bar(flex_work_df, x='Percentage', y='Flexible Option', orientation='h', text='Percentage')

                # Update layout for better readability
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    title_text='Types of Flexible Working',
                    xaxis_title='Percentage of Employees',
                    yaxis_title='Flexible Working Option',
                    uniformtext_minsize=8,
                    uniformtext_mode='hide'
                )
                # Show the figure
                st.plotly_chart(fig)
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                ########################### Add text analysis #######################################
                # Perform text summarization
                if 'Flexible_Working_Comments' in df.columns:

                # Initialize a list in session state for storing analysis results if it doesn't exist
                  if 'flexible_analysis_results' not in st.session_state:
                      st.session_state['flexible_analysis_results'] = {}

                      # Filter out NaN values and entries with '-'
                      filtered_comments = process_text_data(df, 'Flexible_Working_Comments')
                      filtered_comments = df['Flexible_Working_Comments'][~df['Flexible_Working_Comments'].isin([np.nan, '-', ''])]
                      joined_text = ' '.join(filtered_comments)
                      #st.write(joined_text)
                      Flexible_Working_Comments_Analysis = summarize_text(joined_text)
                      #flexible_analysis_results.append(Flexible_Working_Comments_Analysis)
                      st.session_state['flexible_analysis_results']['Flexible_Working_Comments'] = Flexible_Working_Comments_Analysis
                      #st.write(Flexible_Working_Comments_Analysis)

                      st.markdown("### Summary of Comments for Flexible Working")
                      st.markdown("The bullet points below are a concise summary of the key points made by individuals regarding flexible working.")
                      #display_summary(flexible_analysis_results[0])
                      display_summary(st.session_state['flexible_analysis_results']['Flexible_Working_Comments'])

                  else:
                      st.markdown("### Summary of Comments for Flexible Working")
                      st.markdown("The bullet points below are a concise summary of the key points made by individuals regarding flexible working.")
                      #display_summary(flexible_analysis_results[0])
                      display_summary(st.session_state['flexible_analysis_results']['Flexible_Working_Comments'])

                pass
            ######################################################################
            #                               Age                                  #
            ######################################################################
            elif visualization_key == "Age Distribution":
                plot_horizontal_bar_chart(df, 'Age', 'Age Distribution in the Organisation', 'Percentage', 'Age Range')
                pass
            ######################################################################
            #                               Generation                           #
            ######################################################################
            elif visualization_key == "Generation Distribution":
                plot_horizontal_bar_chart(df, 'Generation', 'Generation Distribution in the Organisation', 'Percentage', 'Generation Range')
                pass
            ######################################################################
            #                    Gender & Gender Identity                        #
            ######################################################################
            elif visualization_key == "Gender and gender identity":
                column_name, chart_title = 'Gender', 'Gender'
                plot_pie_chart(df, column_name, chart_title)
                #---------------------------------------------------------------------------------------------------
                if 'Self_Describe_Gender' in df.columns:
                    df['Self_Describe_Gender'] = df['Self_Describe_Gender'].str.lower()
                    # Filter out NaN values and entries with '-'
                    filtered_descriptions = df['Self_Describe_Gender'][~df['Self_Describe_Gender'].isin([np.nan, '-', ''])]
                    # filtered_comments = process_text_data(df, 'Self_Describe_Gender')

                    # Quantitative Analysis: Count unique self-described genders
                    # self_describe_counts = df['Self_Describe_Gender'].value_counts(dropna=True)
                    self_describe_counts = filtered_descriptions.value_counts()
                    st.write("Frequency of each unique self-described gender:\n", self_describe_counts)
                #-----------------------------------------------------------------------------------------------------
                if 'Gender_Identification_Same_as_Birth' in df.columns:
                    # Count the occurrences of each response
                    gender_identity_counts = df['Gender_Identification_Same_as_Birth'].value_counts()

                    # Calculate the percentage of respondents with the same gender identity as assigned at birth
                    # identity_same_as_assigned_at_birth_percentage = (gender_identity_counts['Yes'] / df.shape[0]) * 100
                    # st.write(f"Percentage of respondents with the same gender identity as assigned at birth: {identity_same_as_assigned_at_birth_percentage:.2f}%")
                    # Calculate the percentage of respondents with the same gender identity as assigned at birth
                    total_respondents = df.shape[0]
                    if 'Yes' in gender_identity_counts:
                        identity_same_as_assigned_at_birth_percentage = (gender_identity_counts['Yes'] / total_respondents) * 100
                        st.write(f"Percentage of respondents with the same gender identity as assigned at birth: {identity_same_as_assigned_at_birth_percentage:.2f}%")
                    else:
                        st.write("No respondents with 'Yes' answer for gender identity as assigned at birth in the dataset.")

                    pass
                # else:
                #     st.write("Data for Gender is not available.")
            ######################################################################
            #            Disability or Long Term Health Condetion                #
            ######################################################################
            elif visualization_key == "Disability or long-term health conditions":
                column_name, chart_title = 'Disability_or_Long_Term_Health_Condition', 'Disability or Long Term Health Condition'
                plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #            Disability & Long Term Health Condetion                 #
            ######################################################################
            elif visualization_key == "Disability and long-term health conditions":
                # Check for 'dificulties' columns before analysis
                difficulty_columns = ['Seeing_Dificulty', 'Hearing_Dificulty', 'Walking_Dificulty', 'Remembering_Dificulty', 'SelfCare_Dificulty', 'Communicating_Dificulty', 'Raising_Water/Soda_Bottle_Dificulty', 'Picking_Up_Small_Objects_Dificulty']
                # if all(column in df.columns for column in difficulty_columns):
                    # Prepare the data by counting the number of responses in each category for each question
                difficulty_data = {
                    'Seeing': df['Seeing_Dificulty'].value_counts(normalize=True),
                    'Hearing': df['Hearing_Dificulty'].value_counts(normalize=True),
                    'Walking': df['Walking_Dificulty'].value_counts(normalize=True),
                    'Remembering': df['Remembering_Dificulty'].value_counts(normalize=True),
                    'Self-care': df['SelfCare_Dificulty'].value_counts(normalize=True),
                    'Communicating': df['Communicating_Dificulty'].value_counts(normalize=True),
                    'Raising water/soda bottle': df['Raising_Water/Soda_Bottle_Dificulty'].value_counts(normalize=True),
                    'Picking up small objects': df['Picking_Up_Small_Objects_Dificulty'].value_counts(normalize=True),
                }

                # Create the figure
                fig = go.Figure()

                # Categories and colors
                categories = ['No, no difficulty', 'Yes, some difficulty', 'Yes, a lot of difficulty', 'Cannot do it at all', 'Prefer not to say']
                colors = ['blue', 'orange', 'green', 'red', 'purple']

                # Add each category as a separate trace
                for idx, category in enumerate(categories):
                    fig.add_trace(go.Bar(
                        name=category,
                        y=list(difficulty_data.keys()),
                        x=[difficulty_data[key].get(category, 0) for key in difficulty_data],
                        orientation='h',
                        marker_color=colors[idx]
                    ))

                # Update layout for stacked bar chart
                fig.update_layout(
                    barmode='stack',
                    title='Disability and Long-term Health Conditions',
                    xaxis_title='Percent',
                    yaxis_title='Condition',
                    legend_title='Difficulty Level'
                )

                # Show the figure
                st.plotly_chart(fig)
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                ########################### Add text analysis #######################################
                # Perform text summarization
                if 'Difficulty_Comment' in df.columns:

                  # Initialize a list in session state for storing analysis results if it doesn't exist
                  if 'difficulty_analysis_results' not in st.session_state:
                    st.session_state['difficulty_analysis_results'] = {}

                    # Filter out NaN values and entries with '-'
                    filtered_comments = process_text_data(df, 'Difficulty_Comment')
                    # filtered_comments = df['Difficulty_Comment'][~df['Difficulty_Comment'].isin([np.nan, '-', ''])]
                    joined_text = ' '.join(filtered_comments)
                    #st.write(joined_text)
                    # Difficulty_Comment_Analysis = df['Difficulty_Comment'].apply(summarize_text)
                    Difficulty_Comment_Analysis = summarize_text(joined_text)
                    #difficulty_analysis_results.append(Difficulty_Comment_Analysis)
                    st.session_state['difficulty_analysis_results']['Difficulty_Comment'] = Difficulty_Comment_Analysis
                    #st.write(Difficulty_Comment_Analysis)

                    st.markdown("### Summary of Comments on Difficulties")
                    st.markdown("The bullet points below are a concise summary of the key points made by individuals regarding their difficulties.")
                    display_summary(st.session_state['difficulty_analysis_results']['Difficulty_Comment'])

                  else:
                    st.markdown("### Summary of Comments on Difficulties")
                    st.markdown("The bullet points below are a concise summary of the key points made by individuals regarding their difficulties.")
                    display_summary(st.session_state['difficulty_analysis_results']['Difficulty_Comment'])
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                #--------------------------------------------------------------------------------------------------------------------------------------------------
                # Analyze work adaptation for each difficulty
                if 'Work_Adaptation_for_Difficulties' in df.columns:
                    st.subheader("Work Adaptation Analysis for Different Difficulties")
                    for difficulty_col in difficulty_columns[:-1]:  # Exclude 'Work_Adaptation_for_Difficulties' from the loop###########################################
                        ######################################## -1, chek if we should exclude -1 in difficulty_columns##################################################
                        # Filter the dataframe for employees with a specific difficulty
                        df_filtered = df[df[difficulty_col] != 'No, no difficulty']  # Assuming 'No, no difficulty' means no issues

                        if not df_filtered.empty:
                            # Analysis for Work Adaptation for this specific difficulty
                            column_name, chart_title = 'Work_Adaptation_for_Difficulties', f'Work Adaptation for {difficulty_col}'
                            plot_pie_chart(df, column_name, chart_title)
                            # Analysis for workplace modification for this specific difficulty
                            column_name, chart_title = 'Workplace_Modification_for_Difficulties', f'Workplace modification for {difficulty_col}'
                            plot_pie_chart(df, column_name, chart_title)
                #----------------------------------------------------------------------------------------------------------------------------------------------------
                pass
            ######################################################################
            #                         Mental Health                              #
            ######################################################################
            elif visualization_key == "Mental health":
                # Check for 'How_often_feeling_worried_nervous_anxious' column before analysis
                if 'How_often_feeling_worried_nervous_anxious' in df.columns:
                    #******************** feeling_worried_nervous_anxious ***************#
                    worry_counts = df['How_often_feeling_worried_nervous_anxious'].value_counts()
                    total_responses = worry_counts.sum()  # Total number of non-null responses

                    # Calculate the percentage for each response category
                    worry_percentages = (worry_counts / total_responses) * 100

                    # Create a DataFrame for plotting
                    worry_data = pd.DataFrame({'Frequency': worry_counts.index, 'Percentage': worry_percentages})

                    # Create a bar chart using Plotly
                    fig = px.bar(worry_data, x='Frequency', y='Percentage', text='Percentage')

                    # Update layout for better readability
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        title_text='Frequency of Feeling Worried, Nervous or Anxious',
                        xaxis_title='Frequency',
                        yaxis_title='Percentage'
                    )
                    # Show the figure
                    st.plotly_chart(fig)
                    #-------------------------------------------------------------------------------------
                    # Calculate and print the most common frequency of feeling worried, nervous or anxious
                    most_common_frequency = worry_counts.idxmax()
                    most_common_count = worry_counts.max()
                    st.write(f"The most common frequency of feeling worried, nervous or anxious is: {most_common_frequency} with {most_common_count} responses.")
                else:
                    st.write("Data for Feeling Worried, Nervous or Anxious is not available.")
                #-------------------------------------------------------------------------------------
                # Check for 'Levels of Worry, Nervousness, or Anxiety' column before analysis
                if 'Level_of_last_worrying_anxiety_nervousness' in df.columns:
                    column_name, chart_title = 'Level_of_last_worrying_anxiety_nervousness', 'Levels of Worry, Nervousness, or Anxiety'
                    plot_pie_chart(df, column_name, chart_title)

                #************************ feeling_depressed ********************#
                # Check for 'Frequency of Feeling Depressed' column before analysis
                if 'How_often_feeling_depressed' in df.columns:
                    depression_counts = df['How_often_feeling_depressed'].value_counts()
                    total_responses = depression_counts.sum()  # Total number of non-null responses

                    # Calculate the percentage for each response category
                    depression_percentages = (depression_counts / total_responses) * 100

                    # Create a DataFrame for plotting
                    depression_data = pd.DataFrame({'Frequency': depression_counts.index, 'Percentage': depression_percentages})

                    # Create a bar chart using Plotly
                    fig = px.bar(depression_data, x='Frequency', y='Percentage', text='Percentage')

                    # Update layout for better readability
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        title_text='Frequency of Feeling Depressed',
                        xaxis_title='Frequency',
                        yaxis_title='Percentage'
                    )

                    # Show the figure
                    st.plotly_chart(fig)
                    #-------------------------------------------------------------------------------------
                    # Calculate and print the most common frequency of feeling depressed
                    most_common_frequency = depression_counts.idxmax()
                    most_common_count = depression_counts.max()
                    st.write(f"The most common frequency of feeling depressed is: {most_common_frequency} with {most_common_count} responses.")

                else:
                    st.write("Data for Frequency of Feeling Depressed is not available.")
                #-------------------------------------------------------------------------------------
                    # Check for 'Levels of Depression' column before analysis
                if 'Level_of_last_depression' in df.columns:
                    column_name, chart_title = 'Level_of_last_depression', 'Levels of Depression'
                    plot_pie_chart(df, column_name, chart_title)
            ######################################################################
            #                             Nationality                            #
            ######################################################################

            elif visualization_key == "Nationality":

                # Replace the following dictionary keys with your actual DataFrame column names.
                national_options = ['English','Welsh', 'Scottish', 'Northern_Irish', 'British', 'Prefer_Not_To_Say']

                # Check for 'Types of Flexible Working' column before analysis
                # if national_options in df.columns:
                if all(option in df.columns for option in national_options):
                    # Assuming 'df' is your DataFrame and it has columns for each flexible working option
                    # with values 'Yes' or 'No'. We will create a dictionary to hold the percentage data.

                    # Convert 'Yes' to 1 and 'No' to 0
                    binary_data = df.replace({'Yes': 1, 'No': 0})
                    # Convert the columns to numeric (if they are still categorical)
                    for column in national_options:
                        binary_data[column] = pd.to_numeric(binary_data[column], errors='coerce')


                    # Calculate the percentage of 'Yes' responses for each flexible working option
                    national_percentages = {}
                    total_responses = len(df)

                    for column_name in national_options:
                        national_percentages[column_name] = binary_data[column_name].sum() / total_responses * 100

                    # Create a DataFrame for plotting
                    nationality_df = pd.DataFrame(list(national_percentages.items()), columns=['Nationality', 'Percentage'])

                    # Create a horizontal bar chart using Plotly
                    fig = px.bar(nationality_df, x='Percentage', y='Nationality', orientation='h', text='Percentage')

                    # Update layout for better readability
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig.update_layout(
                        title_text='Nationality of Employees',
                        xaxis_title='Percentage of Employees',
                        yaxis_title='Nationality',
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )
                    # Show the figure
                    st.plotly_chart(fig)
                    #---------------------------------------------------------------------------------------------------
                    if 'National_Identity__Not_in_List' in df.columns:
                        df['National_Identity__Not_in_List'] = df['National_Identity__Not_in_List'].str.lower()
                        # Filter out NaN values and entries with '-'
                        filtered_descriptions = df['National_Identity__Not_in_List'][~df['National_Identity__Not_in_List'].isin([np.nan, '-', ''])]
                        # filtered_comments = process_text_data(df, 'National_Identity__Not_in_List')

                        # Quantitative Analysis: Count unique  National Identity which are not in the list
                        self_describe_counts = filtered_descriptions.value_counts()
                        st.write("Frequency of each unique National Identity which are not in the list:\n", self_describe_counts)


                    pass
            ######################################################################
            #                             Ethnicity                              #
            ######################################################################
            elif visualization_key == "Ethnicity":
                plot_horizontal_bar_chart(df, 'Ethnicity', 'Ethnicity Distribution in the Organisation', 'Percentage', 'Ethnicity')
                #-----------------------------------------------------------------------------------------------------------
                if 'Ethnicity_Not_in_List' in df.columns:
                    df['Ethnicity_Not_in_List'] = df['Ethnicity_Not_in_List'].str.lower()
                    # Filter out NaN values and entries with '-'
                    filtered_descriptions = df['Ethnicity_Not_in_List'][~df['Ethnicity_Not_in_List'].isin([np.nan, '-', ''])]
                    # filtered_comments = process_text_data(df, 'Ethnicity_Not_in_List')

                    # Quantitative Analysis: Count unique ethnicity which are not in the list
                    self_describe_counts = filtered_descriptions.value_counts()
                    st.write("Frequency of each unique ethnicity which are not in the list:\n", self_describe_counts)

                pass
            ######################################################################
            #                             Sexuality                              #
            ######################################################################
            elif visualization_key == "Sexuality":
                column_name, chart_title = 'Sexuality', 'Sexual Orientation'
                plot_pie_chart(df, column_name, chart_title)
                #-------------------------------------------------------------------------------------------
                if 'Self_Describe_Sexual_Orientation' in df.columns:
                    df['Self_Describe_Sexual_Orientation'] = df['Self_Describe_Sexual_Orientation'].str.lower()
                    # Filter out NaN values and entries with '-'
                    filtered_descriptions = df['Self_Describe_Sexual_Orientation'][~df['Self_Describe_Sexual_Orientation'].isin([np.nan, '-', ''])]
                    # filtered_comments = process_text_data(df, 'Self_Describe_Sexual_Orientation')

                    # Quantitative Analysis: Count unique self describe sexual orientation
                    self_describe_counts = filtered_descriptions.value_counts()
                    st.write("Frequency of each unique self describe sexual orientation:\n", self_describe_counts)

                #**************************Openness About Sexual Orientation****************************
                contexts = ['At home', 'With your manager', 'With colleagues', 'At work generally', 'Prefer not to say']
                columns = ['Sexual_Orientation_Openness_At_Home', 'Sexual_Orientation_Openness_With_Manager', 'Sexual_Orientation_Openness_With_Colleagues', 'Sexual_Orientation_Openness_At_Work_Generally', 'Sexual_Orientation_Openness_PNTS']
                if all(item in df.columns for item in columns):       
                    # Prepare the data: count the responses for each category in each context
                    openness_data = {context: df[column].value_counts(normalize=True) * 100 for context, column in zip(contexts, columns)}

                    # Categories and colors
                    categories = df['Sexual_Orientation_Openness_At_Home'].unique()  # Assuming all columns have the same unique values
                    colors = ['blue', 'orange'] 

                    # Creating the figure
                    fig = go.Figure()

                    # Add each category as a separate trace
                    for category, color in zip(categories, colors):
                        fig.add_trace(go.Bar(
                            name=category,
                            x=contexts,
                            y=[openness_data[context].get(category, 0) for context in contexts],
                            marker_color=color
                        ))

                    # Update layout for stacked bar chart
                    fig.update_layout(
                        barmode='stack',
                        title='Openness About Sexual Orientation',
                        xaxis_title='Context',
                        yaxis_title='Percentage',
                        legend_title='Response'
                    )
                    # Show the figure
                    st.plotly_chart(fig)
                    
                    #*********************LGBT+_Openness About Sexual Orientation*************************
                    contexts = ['At home', 'With your manager', 'With colleagues', 'At work generally', 'Prefer not to say']
                    columns = ['Sexual_Orientation_Openness_At_Home', 'Sexual_Orientation_Openness_With_Manager', 'Sexual_Orientation_Openness_With_Colleagues', 'Sexual_Orientation_Openness_At_Work_Generally', 'Sexual_Orientation_Openness_PNTS']

                    # Prepare the data: count the responses for each category in each context
                    LGBT_openness_data = {context: df_LGBT[column].value_counts(normalize=True) * 100 for context, column in zip(contexts, columns)}

                    # Categories and colors
                    categories = df_LGBT['Sexual_Orientation_Openness_At_Home'].unique()  # Assuming all columns have the same unique values
                    colors = ['blue', 'orange'] 

                    # Creating the figure
                    fig = go.Figure()

                    # Add each category as a separate trace
                    for category, color in zip(categories, colors):
                        fig.add_trace(go.Bar(
                            name=category,
                            x=contexts,
                            y=[LGBT_openness_data[context].get(category, 0) for context in contexts],
                            marker_color=color
                        ))

                    # Update layout for stacked bar chart
                    fig.update_layout(
                        barmode='stack',
                        title='LGBT+_Openness About Sexual Orientation',
                        xaxis_title='Context',
                        yaxis_title='Percentage',
                        legend_title='Response'
                    )

                    # Show the figure
                    st.plotly_chart(fig)

                pass
            ######################################################################
            #                     Caring Responsibility                          #
            ######################################################################
            elif visualization_key == "Caring responsibilities":

                categories = [
                    'No',
                    'Yes, of a child/children (under 18)',
                    'Yes, of a disabled child/children (under 18)',
                    'Yes, of a disabled adult (18 and over)',
                    'Yes, of an older person/people (65 and over)',
                    'Prefer not to say'
                ]
                columns = ['Has_Caring_Responsibility','Dependents_Children_Under_18','Dependents_Disabled_Children_Under_18','Dependents_Disabled_Adult_18_and_Over','Dependents_Older_Person_65_and_Over','Dependents_Prefer_Not_to_Say']

                # Count the 'Yes' responses for each category
                caring_counts = df[columns].apply(lambda x: x == 'Yes').sum()

                # Calculate the percentage of 'Yes' responses for each category
                total_respondents = len(df)
                caring_percentages = (caring_counts / total_respondents) * 100

                # Create a DataFrame for plotting
                caring_data = pd.DataFrame({'Category': categories, 'Percentage': caring_percentages})

                # Create a horizontal bar chart using Plotly
                fig = px.bar(caring_data, y='Category', x='Percentage', orientation='h', text='Percentage')

                # Update layout for better readability
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    title_text='Caring Responsibilities',
                    xaxis_title='Percentage',
                    yaxis_title='Type of Caring Responsibility'
                )
                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #                              Religion                              #
            ######################################################################
            elif visualization_key == "Religion":
                column_name, chart_title = 'Religion', 'Religious Affiliation Distribution'
                plot_pie_chart(df, column_name, chart_title)
                #-------------------------------------------------------------------------------------------
                if 'Religion_Not_in_List' in df.columns:
                    df['Religion_Not_in_List'] = df['Religion_Not_in_List'].str.lower()
                    # Filter out NaN values and entries with '-'
                    filtered_descriptions = df['Religion_Not_in_List'][~df['Religion_Not_in_List'].isin([np.nan, '-', ''])]
                    # filtered_comments = process_text_data(df, 'Religion_Not_in_List')

                    # Quantitative Analysis: Count unique religion not in list
                    self_describe_counts = filtered_descriptions.value_counts()
                    st.write("Frequency of each unique religion which are not in the list:\n", self_describe_counts)


                pass
            ######################################################################
            #             Gender and Ethnicity Distribution Chart                #
            ######################################################################
            elif visualization_key == "Gender and Ethnicity Distribution Chart":

                # Create a bar chart with Plotly
                fig = px.bar(df, x='Gender', color='Ethnicity', barmode='group')

                # Update layout for better readability
                fig.update_layout(
                    title_text='Gender and Ethnicity Distribution',
                    xaxis_title='Gender',
                    yaxis_title='Count',
                    legend_title='Ethnicity'
                )

                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #          Gender and Caring Responsibility Distribution             #
            ######################################################################
            elif visualization_key == "Gender and Caring Responsibility Distribution":

                # Create a bar chart with Plotly
                fig = px.bar(df, x='Gender', color='Has_Caring_Responsibility', barmode='group')

                # Update layout for better readability
                fig.update_layout(
                    title_text='Gender and Caring Responsibility Distribution',
                    xaxis_title='Gender',
                    yaxis_title='Count',
                    legend_title='Caring Responsibility'
                )

                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #             Service Length amongst LGBT+ Distribution              #
            ######################################################################
            elif visualization_key == "Service Length amongst LGBT+ Distribution":

                # Create a bar chart with Plotly
                fig = px.bar(df, x='Service_Length', color='LGBT', barmode='group')

                # Update layout for better readability
                fig.update_layout(
                    title_text='Service Length amongst LGBT+ Distribution',
                    xaxis_title='Service_Length',
                    yaxis_title='Count',
                    legend_title='LGBT+'
                )

                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #            Service Length amongst Disableld Employees              #
            ######################################################################
            elif visualization_key == "Service Length amongst Disabled Employees":

                # Create a bar chart with Plotly
                fig = px.bar(df, x='Service_Length', color='Has_Disability', barmode='group')

                # Update layout for better readability
                fig.update_layout(
                    title_text='Service Length amongst Disableld Distribution',
                    xaxis_title='Service_Length',
                    yaxis_title='Count',
                    legend_title='Disability'
                )

                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
                        # Use of Flexible Working Options by Employees with:
                        # 1. Caring Responsibilities
                        # 2. Disabilities
                        # 3. Different Genders
            ######################################################################
            elif visualization_key == "Use of Flexible Working Options by Employees with Caring Responsibilities, Disabilities, Different Genders":

                #*****Use of Flexible Working Options by Employees with Caring Responsibilities/Disabilities*****
                # Define the flexible working options columns
                flexible_options = [
                    'Job_Share', 'Flexibility_with_start_and_finish_times', 'Working_from_home',
                    'Flexible_Hours_Based_on_Output', 'Remote_Working', 'Condensed_Hours',
                    'School_Hours', 'Term_Time'
                ]

                # Dropdown options for selecting the group
                group_options = {
                    'Caring Responsibilities': df_caring_responsibilities,
                    'Disability': df_disabilities
                    # 'Gender': df,
                    # 'Ethnicity': df
                }

                # Streamlit dropdown for group selection
                selected_group = st.selectbox(
                    "Select a group:",
                    options=list(group_options.keys()),
                    index=0  # default index
                )

                # Update chart based on selected group
                def update_chart(selected_group):
                    df_selected_group = group_options[selected_group]

                    flex_work_data = {option: (df_selected_group[option].value_counts().get('Yes', 0) / len(df_selected_group)) * 100
                                    for option in flexible_options}

                    flex_work_df = pd.DataFrame.from_dict(flex_work_data, orient='index', columns=['Percentage']).reset_index()
                    flex_work_df.columns = ['Flexible Working Option', 'Percentage']

                    fig = px.bar(flex_work_df, x='Percentage', y='Flexible Working Option', orientation='h',
                                    title=f'Use of Flexible Working Options by Employees with {selected_group}',
                    text='Percentage',  # Display percentage value on bars
                    labels={'Percentage'})  # Label for the percentage axis

                    # Customize layout for displaying percentage outside of bars
                    fig.update_traces(textposition='outside', texttemplate='%{text:.2f}%')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


                    return fig

                # Display the chart
                st.plotly_chart(update_chart(selected_group))


                #**************Use of Flexible Working Options by Employees with Different Genders***************
                # Define the flexible working options columns
                flexible_options = [
                    'Job_Share', 'Flexibility_with_start_and_finish_times', 'Working_from_home',
                    'Flexible_Hours_Based_on_Output', 'Remote_Working', 'Condensed_Hours',
                    'School_Hours', 'Term_Time'
                ]

                # Assuming 'Gender' column exists in your DataFrame
                genders = df['Gender'].unique()

                # Data preparation for Plotly
                all_data = []

                for gender in genders:
                    data = []
                    text_values = []  # List to store text values for displaying percentages
                    for option in flexible_options:
                        # Filter DataFrame by gender and calculate percentage for each option
                        gender_df = df[df['Gender'] == gender]
                        count_yes = gender_df[option].value_counts().get('Yes', 0)
                        percent_yes = (count_yes / len(gender_df)) * 100
                        data.append(percent_yes)
                        text_values.append(f'{percent_yes:.2f}%')  # Format percentage with two decimal places

                    all_data.append(go.Bar(
                        name=gender,
                        y=flexible_options,
                        x=data,
                        orientation='h',  # Specify horizontal bar orientation
                        text=text_values,  # Specify text values for displaying percentages
                        textposition='outside',
                        # textposition='auto',  # Automatically position text
                        insidetextanchor='start'  # Anchor text to the start of bars
                    ))

                # Create the grouped bar chart
                fig = go.Figure(data=all_data)
                fig.update_layout(
                    barmode='group',  # Display bars beside each other
                    title='Distribution of Flexible Working Options by Gender',
                    xaxis={'title': 'Percentage'},
                    yaxis={'title': 'Flexible Working Option'},
                    legend_title='Gender'
                )
                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #           Mental Health Status Distribution by Department          #
            ######################################################################
            elif visualization_key == "Mental Health Status Distribution by Department":

                # We'll create a crosstab of department and mental health status
                cross_tab = pd.crosstab(df['Department'], df['has_mental_health'])

                # Normalize the crosstab to get the proportion within each department
                cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0)

                # Create a horizontal bar chart using Plotly Express
                fig = px.bar(cross_tab_normalized.reset_index(),
                            y='Department',
                            x=cross_tab_normalized.columns.tolist(),
                            title='Mental Health Status Distribution by Department',
                            orientation='h',
                            barmode='stack')

                # Show the figure
                st.plotly_chart(fig)

                pass
            ######################################################################
            #                         Other Filtering                            #
            ######################################################################
            elif visualization_key == "Optional Filtering":

                # List of demographics
                demographic = ['Service_Length', 'Age', 'Salary', 'Has_Disability', 'Organisational_Role', 'Gender', 'has_mental_health', 'Ethnicity', 'Sexual_Orientation', 'Has_Caring_Responsibility', 'Religion']

                # Streamlit widgets for user input
                selected_group = st.selectbox("Which group are you interested in?", options=list(group_dfs.keys()))

                # Filter the list to only include demographics present in df.columns
                available_demographics = [demo for demo in demographic if demo in df.columns]
                selected_demo = st.selectbox("Which demographic are you interested in?", options=available_demographics)


                # Function to plot based on demographic
                def plot_demographic(df, demographic):
                    if demographic == 'Service_Length' and 'Service_Length' in df.columns:
                        # fig = px.pie(df, names='Service_Length', title='Length of Service Distribution')
                        fig = px.pie(df, names='Service_Length', title= (f"Length of Service amongst {selected_group}"))

                    elif demographic == 'Age' and 'Age' in df.columns:
                        fig = px.bar(df, y='Age', orientation='h', title= (f"Age of {selected_group}"))

                    elif demographic == 'Salary' and 'Salary' in df.columns:
                        fig = px.bar(df, y='Salary', orientation='h', title=(f"Ethnicity of {selected_group}"))

                    elif demographic == 'Has_Disability' and 'Has_Disability' in df.columns:
                        fig = px.bar(df, y='Has_Disability', orientation='h', title=(f"Disability amongst {selected_group}"))

                    elif demographic == 'Organisational_Role' and 'Organisational_Role' in df.columns:
                        fig = px.bar(df, x='Organisational_Role', title=(f"Role of {selected_group}"))

                    elif demographic == 'Gender' and 'Gender' in df.columns:
                        fig = px.pie(df, names='Gender', title= (f"Gender of {selected_group}"))

                    elif demographic == 'has_mental_health' and 'has_mental_health' in df.columns:
                        fig = px.pie(df, names='has_mental_health', title= (f"Mental Health Status amongst {selected_group}"))

                    elif demographic == 'Ethnicity' and 'Ethnicity' in df.columns:
                        fig = px.bar(df, y='Ethnicity', orientation='h', title=(f"Ethnicity of {selected_group}"))

                    elif demographic == 'Sexual_Orientation' and 'Sexual_Orientation' in df.columns:
                        fig = px.pie(df, names='Sexual_Orientation', title=(f"Sexual Orientation of {selected_group}"))

                    elif demographic == 'Has_Caring_Responsibility' and 'Has_Caring_Responsibility' in df.columns:
                        fig = px.bar(df, y='Has_Caring_Responsibility', orientation='h', title= (f"Caring Responsibility amongst {selected_group}"))

                    elif demographic == 'Religion' and 'Religion' in df.columns:
                        fig = px.pie(df, names='Religion', title=(f"Religion {selected_group}"))

                    else:
                        fig = None
                        st.write("Demographic not recognized for plotting.")


                    # if fig is not None:
                    #     fig.show()
                    return fig

                # Display the plot
                if selected_group in group_dfs and selected_demo in demographic:
                    fig = plot_demographic(group_dfs[selected_group], selected_demo)
                    st.plotly_chart(fig)
                else:
                    st.write("Invalid group or demographic selected.")

                pass

        # Loop through selected visualizations and display them
        for viz_title in selected_visualizations:
            st.subheader(viz_title)
            show_visualization(viz_title)

    else:
        st.write("Please upload and preprocess the data in the 'Data Preprocessing' page first.")







elif page == "Social Mobility Analysis":
    st.header("Social Mobility Analysis")
    # Ensure the data is loaded
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        # Assuming df is your main DataFrame
        group_dfs = update_group_dfs(df, groups_info)

        ######################################################################
        #              Parents' University Attendance by Age 18              #
        ######################################################################
        # Specify the column related to parents' university attendance by the time the respondent was 18
        column_name = 'Parents_University_Attendance_By_18'
        if 'Parents_University_Attendance_By_18' in df.columns:
            column_name, chart_title = 'Parents_University_Attendance_By_18', 'Parents\' University Attendance by Age 18'
            plot_pie_chart(df, column_name, chart_title)
        ######################################################################
        #          Type of School Attended Between Ages 11 and 16            #
        ######################################################################
        if 'School_Type_Ages_11_16' in df.columns:
            column_name, chart_title = 'School_Type_Ages_11_16', 'Type of School Attended Between Ages 11 and 16'
            plot_pie_chart(df, column_name, chart_title)
        ######################################################################
        #                 Eligibility for Free School Meals                  #
        ######################################################################
        if 'Eligibility_For_Free_School_Meals' in df.columns:
            column_name, chart_title = 'Eligibility_For_Free_School_Meals', 'Eligibility for Free School Meals'
            plot_pie_chart(df, column_name, chart_title)
        ######################################################################
        #                     Serving  in the Armed Forces                   #
        ######################################################################
        if 'Armed_Forces_Service' in df.columns:
            column_name, chart_title = 'Armed_Forces_Service', 'Serving  in the Armed Forces'
            plot_pie_chart(df, column_name, chart_title)
        ######################################################################
        #                  Lower_Socio_Economic_Background                   #
        ######################################################################
        if 'Lower_Socio_Economic_Background' in df.columns:
            column_name, chart_title = 'Lower_Socio_Economic_Background', 'Describe Yourself as Coming from a Lower Socio-Economic Background'
            plot_pie_chart(df, column_name, chart_title)
        ######################################################################
        #          Parents_Highest_Level_of_Qualification_By_18              #
        ######################################################################
        column_name = 'Parents_Highest_Level_of_Qualification_By_18'
        if 'Parents_Highest_Level_of_Qualification_By_18' in df.columns:
            # Check for 'Parents_Highest_Level_of_Qualification_By_18' column before analysis
            # qualification_counts = df['Parents_Highest_Level_of_Qualification_By_18'].value_counts(normalize=True) * 100
            # fig = px.pie(values=qualification_counts.values, names=qualification_counts.index, title='Highest Level of Parents\' Qualification by age 18')
            # st.plotly_chart(fig)
            # pass
            plot_horizontal_bar_chart(df, 'Parents_Highest_Level_of_Qualification_By_18', 'Highest Level of Parents\' Qualification by age 18', 'Percentage', 'Qualification')
           
        ######################################################################
        #          Main_Earner_Employee_or_SelfEmployed_At_14                #
        ######################################################################
        column_name = 'Main_Earner_Employee_or_SelfEmployed_At_14'
        if 'Main_Earner_Employee_or_SelfEmployed_At_14' in df.columns:
            # Check for 'Main_Earner_Employee_or_SelfEmployed_At_14' column before analysis
            # employee_status_counts = df['Main_Earner_Employee_or_SelfEmployed_At_14'].value_counts(normalize=True) * 100
            # fig = px.pie(values=employee_status_counts.values, names=employee_status_counts.index, title='Employment Status of Main Erner by Age 14')
            # st.plotly_chart(fig)
            # pass
            plot_horizontal_bar_chart(df, 'Main_Earner_Employee_or_SelfEmployed_At_14', 'Employment Status of Main Erner by Age 14', 'Percentage', 'Employment Status')
        ######################################################################
        #          Occupation of Main Household Earner at Age 14             #
        ######################################################################
        column_name = 'Main_Earner_Occupation_At_14'
        if 'Main_Earner_Occupation_At_14' in df.columns:
            plot_horizontal_bar_chart(df, 'Main_Earner_Occupation_At_14', 'Occupation of Main Household Earner at Age 14', 'Percentage', 'Occupation')
        ######################################################################
        #     Role Distribution by Parental University Attendance            #
        ######################################################################
        cols = ['Organisational_Role', 'Parents_University_Attendance_By_18']
        if all(col in df.columns for col in cols):
            # We'll create a cross-tabulation of current role and parental university attendance
            cross_tab = pd.crosstab(df['Organisational_Role'],
                                    df['Parents_University_Attendance_By_18'])

            # Normalize the cross-tabulation to get the proportion of each role where parents did/did not attend university
            cross_tab_normalized = cross_tab.div(cross_tab.sum(axis=1), axis=0)

            # Create a bar chart using Plotly Express
            fig = px.bar(cross_tab_normalized.reset_index(),
                        x='Organisational_Role',
                        y=cross_tab_normalized.columns.tolist(),
                        title='Role Distribution by Parental University Attendance',
                        barmode='group')
            # Show the figure
            st.plotly_chart(fig)
        ######################################################################
        #            Role Distribution by Type of School Attended            #
        ######################################################################
        cols = ['Organisational_Role', 'School_Type_Ages_11_16']
        if all(col in df.columns for col in cols):
            # Create a crosstab for question 21 and roles
            cross_tab_q21 = pd.crosstab(df['Organisational_Role'],
                                        df['School_Type_Ages_11_16'])

            # Normalize the crosstab
            cross_tab_normalized_q21 = cross_tab_q21.div(cross_tab_q21.sum(axis=1), axis=0)

            # Create a bar chart for question 21
            fig = px.bar(cross_tab_normalized_q21.reset_index(),
                        x='Organisational_Role',
                        y=cross_tab_normalized_q21.columns.tolist(),
                        title='Role Distribution by Type of School Attended',
                        barmode='group')

            # Show the figure
            st.plotly_chart(fig)




elif page == "Inclusion Analysis":
    st.header("Inclusion Analysis")
    st.subheader("Please select the desired analysis from the the dropdown box.")
    
    # Ensure the data is loaded
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        # Assuming df is your main DataFrame
        group_dfs = update_group_dfs(df, groups_info)

        # Response categories and updated colors
        response_categories = ['Strongly agree', 'Agree', 'Neither agree nor disagree', 'Disagree', 'Strongly disagree']
        colors = {
            'Strongly agree': '#2ca02c',  # Darker green
            'Agree': '#98df8a',  # Light green
            'Neither agree nor disagree': '#ffbb78',  # Neutral orange
            'Disagree': '#ff7f0e',  # Light red/orange
            'Strongly disagree': '#d62728'  # Darker red
        }

        # Define the list of questions
        questions = [
            'In your opinion, how much of a priority is diversity and inclusion in the business to',
            'I feel like I belong at business',
            'I feel that I might not belong at business when something negative happens to me at work',
            'I can voice a contrary opinion without fear of negative consequences',
            'I often worry I do not have things in common with others at business',
            'I feel like my colleagues understand who I really am',
            'I feel respected and valued by my colleagues at business',
            'I feel respected and valued by my manager',
            'I feel confident I can develop my career at at business',
            'When I speak up at work, my opinion is valued',
            'Administrative tasks that don’t have a specific owner, are divided fairly at business',
            'Promotion decisions are fair at business',
            'My job performance is evaluated fairly',
            'Business believes that people can always improve their talents and abilities',
            'Business believes that people have a certain amount of talent, and they can’t do much to change it',
            'Working at Business is important to the way that I think of myself as a person',
            'The information and resources I need to do my job effectively are available',
            'Business hires people from diverse backgrounds',
            'Would you agree that you are able to reach your full potential at work?', ##
            'Which of the following statements best describes how you feel in your team',
            'How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?',
            'In the next 6 months, are you considering leaving this organisation because you do not feel respected or that you belong?'

        ]

        # Filter options based on the columns present in the DataFrame
        # filtered_question = {question for question in questions if question in df.columns}
        filtered_question = {question for question in questions if question in df.columns}##########################The one in COLAB doesn't have this
        if all(item in df.columns for item in ['EDI_Priority_Senior_Leadership', 'EDI_Priority_Line_Manager', 'EDI_Priority_Peers', 'EDI_Priority_YourSelf']):
            filtered_question.add('In your opinion, how much of a priority is diversity and inclusion in the business to')
        if 'Considering_Leaveing_in_Next_6_Months' in df.columns:
            filtered_question.add('In the next 6 months, are you considering leaving this organisation because you do not feel respected or that you belong?')
        # Create the dropdown box
        selected_question = st.selectbox("Which question are you interested in?", filtered_question)



        # Assuming `selected_visualizations` contains the questions you want to visualize,
        # `response_categories` is a list of all possible responses,
        # `group_dfs` is your dictionary of groups and their dataframes,
        # and `colors` is a dictionary mapping responses to colors.

            

        ##############################################################################################
        #                          Considering_Leaveing_in_Next_6_Months                             # 
        ##############################################################################################
        if selected_question == "Considering_Leaveing_in_Next_6_Months":
            column_name, chart_title = 'Considering_Leaveing_in_Next_6_Months', 'Leaving the Company in the Next 6 Month'
            plot_pie_chart(df, column_name, chart_title)
            #---------------------------------------------------------------------------------------------------------------------            
            # Function to calculate turnover costs
            def calculate_turnover_costs(days_to_fill_position, days_to_ramp_up, percent_salary_cost_to_find_talent):
                # Dictionary to hold the results for each group
                turnover_costs_by_group = {}
                # Iterate over each group and perform calculations
                for group_name, group_df in group_dfs.items():
                    # Convert 'Average_Salary' to a numeric type (float) before computing the mean
                    group_df['Average_Salary'] = pd.to_numeric(group_df['Average_Salary'], errors='coerce')
                    # Calculate the number of employees considering leaving (including 'Maybe')
                    employees_leaving = group_df[(group_df['Considering_Leaveing_in_Next_6_Months'] == 'Yes') | 
                                                (group_df['Considering_Leaveing_in_Next_6_Months'] == 'Maybe')].shape[0]

                    # Calculate average salary for those considering leaving (including 'Maybe')
                    average_salary_leaving = round(group_df[(group_df['Considering_Leaveing_in_Next_6_Months'] == 'Yes') | 
                                                    (group_df['Considering_Leaveing_in_Next_6_Months'] == 'Maybe')]['Average_Salary'].mean())

                    # Calculate each component of turnover cost
                    recruitment_cost = employees_leaving * percent_salary_cost_to_find_talent * average_salary_leaving
                    unfilled_role_opportunity_cost = employees_leaving * days_to_fill_position * (2 * average_salary_leaving / 365)
                    ramp_up_time_opportunity_cost = employees_leaving * days_to_ramp_up * (average_salary_leaving / 365)

                    # Total cost of turnover
                    total_turnover_cost = recruitment_cost + unfilled_role_opportunity_cost + ramp_up_time_opportunity_cost
                    
                    # Add the results to the dictionary
                    turnover_costs_by_group[group_name] = {
                        'Cost of Attrition': total_turnover_cost,
                        'Number of Potential Leavers': employees_leaving,
                        'Average Salary': average_salary_leaving
                    }


                    # Calculate the number of employees considering leaving (Yes Only)
                    employees_leaving_yes = group_df[group_df['Considering_Leaveing_in_Next_6_Months'] == 'Yes'].shape[0]

                    # Calculate average salary for those considering leaving (Yes Only)
                    average_salary_leaving_yes = round(group_df[group_df['Considering_Leaveing_in_Next_6_Months'] == 'Yes']['Average_Salary'].mean())

                    # Calculate each component of turnover cost (Yes Only)
                    recruitment_cost_yes = employees_leaving_yes * percent_salary_cost_to_find_talent * average_salary_leaving_yes
                    unfilled_role_opportunity_cost_yes = employees_leaving_yes * days_to_fill_position * (2 * average_salary_leaving_yes / 365)
                    ramp_up_time_opportunity_cost_yes = employees_leaving_yes * days_to_ramp_up * (average_salary_leaving_yes / 365)

                    # Total cost of turnover (Yes Only)
                    total_turnover_cost_yes = recruitment_cost_yes + unfilled_role_opportunity_cost_yes + ramp_up_time_opportunity_cost_yes
                    
                    # Add the results to the dictionary (including Yes Only)
                    turnover_costs_by_group[group_name] = {
                        'Cost of Attrition (Yes & Maybe)': total_turnover_cost,
                        'Number of Potential Leavers (Yes & Maybe)': employees_leaving,
                        'Average Salary (Yes & Maybe)': average_salary_leaving,
                        'Cost of Attrition (Yes Only)': total_turnover_cost_yes,  
                        'Number of Potential Leavers (Yes Only)': employees_leaving_yes,
                        'Average Salary (Yes Only)': average_salary_leaving_yes
                    }

                # Convert the dictionary to a DataFrame for display
                turnover_costs_df = pd.DataFrame.from_dict(turnover_costs_by_group, orient='index')
                turnover_costs_df.reset_index(inplace=True)
                turnover_costs_df.rename(columns={'index': 'Group'}, inplace=True)

                # Format the 'Cost of Attrition' as currency
                turnover_costs_df['Cost of Attrition (Yes & Maybe)'] = turnover_costs_df['Cost of Attrition (Yes & Maybe)'].apply(lambda x: f"£ {x:,.2f}")
                turnover_costs_df['Cost of Attrition (Yes Only)'] = turnover_costs_df['Cost of Attrition (Yes Only)'].apply(lambda x: f"£ {x:,.2f}")
                turnover_costs_df['Average Salary (Yes & Maybe)'] = turnover_costs_df['Average Salary (Yes & Maybe)'].apply(lambda x: f"£ {x:,.2f}")
                turnover_costs_df['Average Salary (Yes Only)'] = turnover_costs_df['Average Salary (Yes Only)'].apply(lambda x: f"£ {x:,.2f}")

                return turnover_costs_df

            # Streamlit interface
            st.title("Turnover Costs Calculator")

            # Sliders
            days_to_fill_position = st.slider('Days to Fill Position', 0, 100, 37)
            days_to_ramp_up = st.slider('Days to Ramp Up New Hire', 0, 100, 60)
            percent_salary_cost = st.slider('Percentage of Salary Cost to Find Talent', 0.0, 2.0, 0.15)

            # Button
            if st.button('Calculate Turnover Costs'):
                # Call the calculate function with the slider values
                turnover_costs_df = calculate_turnover_costs(days_to_fill_position, days_to_ramp_up, percent_salary_cost)
                # Display the DataFrame
                st.dataframe(turnover_costs_df)
            pass
        # Now, based on the selected question, you can display relevant analysis or information
        #############################################################################################################
        #         In your opinion, how much of a priority is diversity and inclusion in the business to             #
        #############################################################################################################
        elif selected_question == "In your opinion, how much of a priority is diversity and inclusion in the business to": ######################change to 'if' if it's the first one
            
            # Streamlit dropdown
            selected_group = st.selectbox(
                "Select a group",
                options=list(group_dfs.keys()),
                index=0
            )

            # Function to update the figure
            def update_figure(selected_group):
                df_group = group_dfs[selected_group]

                # Ensure consistent priority levels across all groups
                priority_levels = ['Extremely important', 'Very important', 'Somewhat important', 'Not so important', 'Not at all important', 'Prefer not to say']

                # Prepare data for Plotly
                data = []
                for level in priority_levels:
                    values = [df_group[col].value_counts(normalize=True).get(level, 0) * 100 for col in ['EDI_Priority_Senior_Leadership', 'EDI_Priority_Line_Manager', 'EDI_Priority_Peers', 'EDI_Priority_YourSelf']]
                    data.append(go.Bar(
                        x=['Senior Leadership', 'Line Manager', 'Peers', 'Yourself'],
                        y=values,
                        name=level
                    ))

                fig = go.Figure(data=data)
                fig.update_layout(
                    title='Diversity and Inclusion Priority Level by Group',
                    barmode='group',
                    xaxis={'title': 'Group'},
                    yaxis={'title': 'Percentage'},
                    legend_title='Priority Level'
                )
                return fig

            # Display the chart
            st.plotly_chart(update_figure(selected_group))
            #--------------------------------------------------------------------------------------------------------------------------------------------------
            ########################### Add text analysis #######################################
            # Perform text summarization
            if 'Advice_for_Senior_Leadership_Team_re_EDI' in df.columns:

              # Initialize a list in session state for storing analysis results if it doesn't exist
              if 'edi_priority_analysis_results' not in st.session_state:
                  st.session_state['edi_priority_analysis_results'] = {}

                  # Filter out NaN values and entries with '-'
                  filtered_comments = process_text_data(df, 'Advice_for_Senior_Leadership_Team_re_EDI')
                  # filtered_comments = df['Advice_for_Senior_Leadership_Team_re_EDI'][~df['Advice_for_Senior_Leadership_Team_re_EDI'].isin([np.nan, '-', ''])]
                  joined_text = ' '.join(filtered_comments)
                  #st.write(joined_text)
                  Advice_for_Senior_Leadership_Team_re_EDI_Analysis = summarize_text(joined_text)
                  #edi_priority_analysis_results.append(Advice_for_Senior_Leadership_Team_re_EDI_Analysis)
                  st.session_state['edi_priority_analysis_results']['Advice_for_Senior_Leadership_Team_re_EDI'] = Advice_for_Senior_Leadership_Team_re_EDI_Analysis
                  #st.write(Advice_for_Senior_Leadership_Team_re_EDI_Analysis)

                  st.markdown("### Summary of Advice for Senior Leadership Team Regarding Diversity and Inclusion")
                  st.markdown("The bullet points below are a concise summary of the key points made by individuals.")
                  display_summary(st.session_state['edi_priority_analysis_results']['Advice_for_Senior_Leadership_Team_re_EDI'])

              else:
                st.markdown("### Summary of Advice for Senior Leadership Team Regarding Diversity and Inclusion")
                st.markdown("The bullet points below are a concise summary of the key points made by individuals.")
                display_summary(st.session_state['edi_priority_analysis_results']['Advice_for_Senior_Leadership_Team_re_EDI'])
            pass
        ######################################################################
        #              I feel like I belong at business                      #
        ######################################################################
        elif selected_question == "I feel like I belong at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ################################################################################################
        #   I feel that I might not belong at business when something negative happens to me at work   #
        ################################################################################################
        if selected_question == "I feel that I might not belong at business when something negative happens to me at work":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ##########################################################################
        #  I can voice a contrary opinion without fear of negative consequences  #
        ##########################################################################
        if selected_question == "I can voice a contrary opinion without fear of negative consequences":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ########################################################################
        # I often worry I do not have things in common with others at business #
        ########################################################################
        if selected_question == "I often worry I do not have things in common with others at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #         I feel like my colleagues understand who I really am       #
        ######################################################################
        if selected_question == "I feel like my colleagues understand who I really am":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #     I feel respected and valued by my colleagues at business       #
        ######################################################################
        if selected_question == "I feel respected and valued by my colleagues at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #             I feel respected and valued by my manager              #
        ######################################################################
        if selected_question == "I feel respected and valued by my manager":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ########################################################################
        #       I feel confident I can develop my career at at business        #
        ########################################################################
        if selected_question == "I feel confident I can develop my career at at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #           When I speak up at work, my opinion is valued            #
        ######################################################################
        if selected_question == "When I speak up at work, my opinion is valued":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        #################################################################################################
        #      Administrative tasks that don’t have a specific owner, are divided fairly at business    #
        #################################################################################################
        if selected_question == "Administrative tasks that don’t have a specific owner, are divided fairly at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #             Promotion decisions are fair at business               #
        ######################################################################
        if selected_question == "Promotion decisions are fair at business":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #                My job performance is evaluated fairly              #
        ######################################################################
        if selected_question == "My job performance is evaluated fairly":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ################################################################################################
        #         Business believes that people can always improve their talents and abilities         #
        ################################################################################################
        if selected_question == "Business believes that people can always improve their talents and abilities":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################################################################
        #          Business believes that people have a certain amount of talent, and they can’t do much to change it        #
        ######################################################################################################################
        if selected_question == "Business believes that people have a certain amount of talent, and they can’t do much to change it":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################################################
        #           Working at Business is important to the way that I think of myself as a person           #
        ######################################################################################################
        if selected_question == "Working at Business is important to the way that I think of myself as a person":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################################################
        #            The information and resources I need to do my job effectively are available             #
        ######################################################################################################
        if selected_question == "The information and resources I need to do my job effectively are available":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################
        #          Business hires people from diverse backgrounds            #
        ######################################################################
        if selected_question == "Business hires people from diverse backgrounds":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ########################################################################################
        #       Would you agree that you are able to reach your full potential at work?        #
        ########################################################################################
        if selected_question == "Would you agree that you are able to reach your full potential at work?":
            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ###################################################################################################
        #           Which of the following statements best describes how you feel in your team            #
        ###################################################################################################
        if selected_question == "Which of the following statements best describes how you feel in your team":
            response_categories = ['Key Component', 'Some Influence', 'Safe Voicing', 'Unsafe Voicing', 'Ignored by Others']

            colors = {
                'Key Component': '#2ca02c',  # Darker green
                'Some Influence': '#98df8a',  # Light green
                'Safe Voicing': '#ffbb78',  # Neutral orange
                'Unsafe Voicing': '#ff7f0e',  # Light red/orange
                'Ignored by Others': '#d62728'  # Darker red
            }


            plot_group_responses(selected_question, response_categories, group_dfs, colors)

        ######################################################################################################################################
        #           How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?          #
        ######################################################################################################################################
        if selected_question == "How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?":
            # Function to calculate NPS and percentages
            def calculate_nps_details(scores):
                promoters = (scores >= 9).sum()
                passives = ((scores >= 7) & (scores < 9)).sum()
                detractors = (scores <= 6).sum()

                # Calculate Net Promoter Score
                nps = ((promoters - detractors) / len(scores)) * 100
                nps_formatted = f'Net Promoter Score: {nps.round(2)}'

                # Calculate percentages
                promoters_percentage = f'{((promoters / len(scores)) * 100).round(2)}% Promoters'
                passives_percentage = f'{((passives / len(scores)) * 100).round(2)}% Passives'
                detractors_percentage = f'{((detractors / len(scores)) * 100).round(2)}% Detractors'
                
                return nps, nps_formatted, promoters_percentage, passives_percentage, detractors_percentage

            # # Sample scores from DataFrame
            # scores = df['How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?']
            # Convert the 'scores' column to numeric
            scores = pd.to_numeric(df['How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?'], errors='coerce')

            # Calculate NPS and other details
            nps_score, nps_formatted, promoters, passives, detractors = calculate_nps_details(scores)

            # Display NPS and other details in Streamlit
            st.write(nps_formatted)
            st.write(promoters)
            st.write(passives)
            st.write(detractors)

            # Create and display a gauge chart for NPS
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = nps_score,
                title = {'text': "Net Promoter Score"},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "lightblue"},
                    'steps' : [
                        {'range': [-100, 0], 'color': "lightgray"},
                        {'range': [0, 100], 'color': "gray"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': nps_score}
                }
            ))
            st.plotly_chart(fig)

            # Column name for scores
            column_name = 'How likely is it that you would recommend this business as an inclusive place to work to a friend or colleague?'

            # Calculate NPS for each group
            nps_scores = {group: calculate_nps_details(pd.to_numeric(dataframe[column_name]))[0] for group, dataframe in group_dfs.items()}

            # Convert NPS scores to DataFrame and display in Streamlit
            nps_table = pd.DataFrame(list(nps_scores.items()), columns=['Group', 'NPS Score'])
            st.dataframe(nps_table)

            pass



################################
################################
################################
elif page == "Text Analysis":
    st.header("Text Analysis Results")
    st.subheader("Please choose the open-ended question that you want to analyze.")
    torch.cuda.empty_cache()
    # Check if data is loaded
    if 'df' in st.session_state and st.session_state['df'] is not None:
      df = st.session_state['df']
    text_columns = ['What other comments would you like to make in relation to D&I at this organisation?',
                    'What ONE thing do you think the business should be doing to recruit a diverse range of employees?',
                    'What ONE thing do you think the business does well in terms of creating a diverse and inclusive workplace?',
                    'What ONE thing do you think the business should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?',
                    'In what ways can this organisation ensure that everyone is treated fairly and can thrive?', ## Is it the same as above question???
                    'We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?']

    # Check if any of the specified text columns are present in the DataFrame
    if not any(col in df.columns for col in text_columns):
        st.write("There's no text data to be analysed.")
    else:

        


        # # Check if data is loaded
        # if 'df' in st.session_state and st.session_state['df'] is not None:
        #   df = st.session_state['df']

        # Initialize a dictionary in session state for storing analysis results if it doesn't exist
        if 'analysis_results' not in st.session_state:
           st.session_state['analysis_results'] = {}




          # Mapping of question names to column names
          #questions = {
              #"Recruiting Diverse Employees": "What ONE thing do you think the business should be doing to recruit a diverse range of employees?",
              #"Creating a Diverse Workplace": "What ONE thing do you think the business does well in terms of creating a diverse and inclusive workplace?",
              #"Respectful Work Environment": "What ONE thing do you think the business should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?",
              #"Comments on D&I": "What other comments would you like to make in relation to D&I at this organisation?"
          #}
          # User selects a question
          #selected_question = st.selectbox("Select a Question", list(questions.values()))

        questions = {
              "Recruiting Diverse Employees": "What ONE thing do you think the business should be doing to recruit a diverse range of employees?",
              "Creating a Diverse Workplace": "What ONE thing do you think the business does well in terms of creating a diverse and inclusive workplace?",
              "Respectful Work Environment": "What ONE thing do you think the business should be doing to create a work environment where everyone is respected and can thrive regardless of personal circumstances or background?",

              "Fairly Treated Work Environment": "In what ways can this organisation ensure that everyone is treated fairly and can thrive?",
              "REG Interest Establishing": "We want to support employees in setting up networks for our people if there is demand. What ERG would you be interested in us establishing?",

              "Comments on D&I": "What other comments would you like to make in relation to D&I at this organisation?"
              }




        # Filter the values to only include those that are present in df.columns
        available_questions = [q for q in questions.values() if q in df.columns]
        options = ['Select...'] + available_questions

        # User selects a question from the available options with 'Select...' as the default
        selected_question = st.selectbox("Select a Question", options, index=0)

        if selected_question != 'Select...':





        # # Filter the values to only include those that are present in df.columns
        # available_questions = [q for q in questions.values() if q in df.columns]

        # # User selects a question from the available options
        # selected_question = st.selectbox("Select a Question", available_questions)


            # Check if selected question's analysis is already stored
            if selected_question in st.session_state['analysis_results']:
                # Retrieve and display the stored analysis results
                topic_info = st.session_state['analysis_results'][selected_question]
                display_topics_and_docs(topic_info, selected_question)


            else:


                # Process text data for the selected question
                # abstracts = process_text_data(df, questions[selected_question])
                abstracts = process_text_data(df, selected_question)

                # Perform topic modeling
                topic_info = perform_topic_modeling(abstracts)               


                # Perform text summarization
                topic_info['Summary'] = topic_info['Representative_Docs'].apply(summarize_text)


                # Store the new analysis results in the session state
                st.session_state['analysis_results'][selected_question] = topic_info


                # Display topic info and topics with representative documents
                #st.subheader("Text Analysis Results")
                # display_topics_and_docs(topic_info, questions[selected_question])
                display_topics_and_docs(topic_info, selected_question)

                # Save results to CSV
                # def get_key_for_value(my_dict, value_to_find):
                  # for key, value in my_dict.items():
                      # if value == value_to_find:
                          # return key
                  # return None
                question = get_key_for_value(questions, selected_question)

                csv_filename = f"{question.replace(' ', '_').lower()}_summarized.csv"
                topic_info.to_csv(csv_filename, encoding='utf-8', index=False)

                # Download button for the file
                st.markdown(convert_df_to_csv_download_link(topic_info, csv_filename), unsafe_allow_html=True)

else:
  st.write("Please upload and preprocess the data in the 'Data Preprocessing' page first.")
