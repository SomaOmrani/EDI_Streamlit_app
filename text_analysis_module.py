import streamlit as st
import pandas as pd
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

#---------------------------------------------------------

#################################################################################
#################################################################################
#Text analysis here
#################################################################################
def process_text_data(df, column_name):
    # Load and preprocess data
    Q = df[[column_name]]
    #Q.replace('-', pd.NA, inplace=True)
    Q.replace('-', np.nan, inplace=True)
    Q.dropna(inplace=True)
    Q = Q.reset_index(drop=True)
    Q[column_name] = Q[column_name].str.lower().apply(lambda x: re.sub(r'\W+', ' ', x))
    return Q[column_name]

def perform_topic_modeling(abstracts):
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    topics, probs = topic_model.fit_transform(abstracts, embeddings)
    return topic_model.get_topic_info()

def summarize_text(text):
  return llm_chain.run(text)

def display_summary(summary_text):
    # Split the summary text into bullet points based on the "•" character
    # Assuming each bullet point is separated by " • " (space, bullet, space)
    bullet_points = summary_text.split(" • ")
    # Display each bullet point on a separate line
    for bullet_point in bullet_points:
        # Check if bullet_point is not empty to avoid printing empty lines
        if bullet_point.strip():
            st.markdown(f"- {bullet_point.strip()}")

# Function to convert a DataFrame to a CSV download link
def convert_df_to_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">Download {filename}</a>'
    return href

# Function to display topics and their representative documents
def display_topics_and_docs(df, question):
    st.subheader(f"{question}")
    for index, row in df.iterrows():
        topic = row['Llama2'][0]
        representative_docs = row['Representative_Docs']
        summary = row['Summary']
        st.markdown(f"**Topic: {topic}**")
        
        st.markdown("##### Key Insights in this cluster:")
        # Use regex to find bullet points
        bullet_points = re.split(r'\n*[-*]\s|\n*\d+\.\s|\n+', summary)
        for point in bullet_points:
            point = point.strip()  # Remove leading/trailing whitespace
            if point:  # Check if the point is not empty
                st.markdown(f"- {point}")

        st.markdown("##### Examples:")
        st.write(representative_docs)

# Save results to CSV
def get_key_for_value(my_dict, value_to_find):
    for key, value in my_dict.items():
        if value == value_to_find:
            return key
    return None
##########################################################################################
#                                   Topic Modeling                                       #
##########################################################################################
model_id = 'meta-llama/Llama-2-13b-chat-hf'
#-------------------------------------------
# Logging to hugging face
login("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")
#----------------------------------------------
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library

#bnb_config = transformers.BitsAndBytesConfig(
    #  load_in_4bit=True,  # 4-bit quantization
    #  bnb_4bit_quant_type='nf4',  # Normalized float 4
    #  bnb_4bit_use_double_quant=True,  # Second quantization after the first
    #  bnb_4bit_compute_dtype=bfloat16  # Computation type
# )

bnb_config = transformers.BitsAndBytesConfig(
load_in_4bit=True,  # 4-bit quantization
bnb_4bit_quant_type='nf4',  # Normalized float 4
bnb_4bit_use_double_quant=True,  # Second quantization after the first
bnb_4bit_compute_dtype=torch.bfloat16,  # Computation type
load_in_8bit_fp32_cpu_offload=True  # Enable offloading to CPU
)
#---------------------------------------------------------------------------------
# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)##

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()
#------------------------------------------------------------------------------
# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    # max_new_tokens=500,
    max_new_tokens=300,
    repetition_penalty=1.1
)
#---------------------------------------------------------------------------
# System prompt describes information given to all conversations
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

# Example prompt demonstrating the output we are looking for
example_prompt = """
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

[/INST] Environmental impacts of eating meat
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""

prompt_topics = system_prompt + example_prompt + main_prompt
#--------------------------------------------------------------------------------------------
# Pre-calculate embeddings
embedding_model = SentenceTransformer("BAAI/bge-small-en")
#----------------------------------------------------------------------------------------------
umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
#----------------------------------------------------------------------------------------------------------------------

# Text generation with Llama 2
llama2 = TextGeneration(generator, prompt=prompt_topics)

# All representation models
representation_model = {
    "Llama2": llama2
}
#-----------------------------------------------------------------------------------
topic_model = BERTopic(

    # Sub-models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,

    # Hyperparameters
    top_n_words=5,
    verbose=True
)

##########################################################################################
#                                 Text Summarization                                     #
##########################################################################################
# model = "meta-llama/Llama-2-7b-chat-hf"

torch.cuda.empty_cache()

#-------------------------------------------
# Logging to hugging face
login("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")
#----------------------------------------------


tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
          """

prompt_summary = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt_summary, llm=llm)



# Function to anonymize people and org names using spaCy's NER
def replace_named_entities(text):
    if isinstance(text, str):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                text =text.replace(ent.text, f"[{ent.label_}]")
    return text


