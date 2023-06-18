import openai
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import ast

openai.api_key = "your API key"

print('1')
def find_best_match(embedding, df):
    embedding_2d = np.array(embedding).reshape(1, -1)

    distances = cdist(embedding_2d, df['embedding'].tolist(), 'cosine')
    best_match_index = np.argmin(distances)
    return df.iloc[best_match_index]

def process_dataframe(df):
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))


    return df

def generate_text_embedding(text):
    response = openai.Embedding.create(input=text, model="model")
    return response["data"][0]["embedding"]


def generate_response(input_text, df):
    input_embedding = generate_text_embedding(input_text)
    matched_content = find_best_match(input_embedding, df)
    response = openai.Completion.create(
        model="model",
        prompt=f"""You are a knowledgeable AI developed by OpenAI.
                        - question: "{input_text}"
                        - Content: "{matched_content['content']}"
                        - AI's Answer: """,
        max_tokens=600,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def main():
    df = pd.read_csv("The path of the csv file with the embeddings of the contents.")
    df = process_dataframe(df)
    print('2')
    st.markdown("<h1 style='text-align: center; color: #fec301;'>CHATBOT</h1>",
                unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            .stTextInput>div>div>input {
                background-color: #FFFFFF;
                color: #000000;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    print('3')

    user_input = st.text_input("Ask question:")

    if user_input:
        response = generate_response(user_input, df)
        st.write(response)

if __name__ == "__main__":
    main()
