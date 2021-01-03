# Core Pkgs
import streamlit as st 
# NLP Pkgs
import spacy_streamlit
import spacy
#nlp = spacy.load('en')
import os
from PIL import Image
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords

import trafilatura
import pdfplumber



def main():
   """A Simple NLP app with Spacy-Streamlit"""
   st.title("Text processing app for biological scientific papers")
   menu = ["Home","NER","Summarization","Trainer"]
   choice = st.sidebar.selectbox("Menu",menu)
   if choice == "Home":
      
      link = '[GitHub page](https://github.com/fm1320/IC_NLP)'
      st.write("""This application was made as part of a postgradute program at Imeprial College London. The details about the traning of the models, data and the techniques can be found at my personal github page provided below.""")
      st.markdown(link, unsafe_allow_html=True)
      st.markdown("![Alt Text](https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Imperial_College_London_monotone_logo.jpg/320px-Imperial_College_London_monotone_logo.jpg)")
      st.write("""*Make sure the requirements are installed before running the website""")
      
      #st.subheader("Tokenization")
      #raw_text = st.text_area("Your Text","Enter Text Here")
      #docx = nlp(raw_text)
      #if st.button("Tokenize"):
      #   spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
      
   elif choice == "NER":
      st.subheader("Named Entity Recognition")
      # Add a selectbox to the sidebar:
      sel = st.sidebar.selectbox("Which NER model would you like to use ?", ["SciSpacy", "BiAssay", "Spacy core en"])
      
      if sel== "SciSpacy":
         #import scispacy
         nlp = spacy.load("en_core_sci_sm")
      elif sel=="BiAssay":
         nlp = spacy.load(r'C:\Users\Filip\Desktop\BiA') #Location of directory of spacy model
      elif sel=="Spacy core en":
         import en_core_web_sm
         nlp = en_core_web_sm.load() 
      
      method = st.sidebar.selectbox("Choose input method (recommended:text box)", ["Text box", "URL"])   

      
      if method == "Text box":
         raw_text = st.text_area("Enter text for entity recognition","Text here")   
         docx = nlp(raw_text)
         spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

      if method == "URL":
         user_input = st.text_input("Enter page URL of an HTML file")
         if user_input is not None:
            downloaded = trafilatura.fetch_url(user_input)
            raw_text=trafilatura.extract(downloaded)
            raw_text=str(raw_text)
            docx = nlp(raw_text)
            spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
   
   elif choice == "Summarization":
      #Textbox for text user is entering
      st.subheader("Enter the text you'd like to summarize.")
      raw_text = st.text_input('Enter text') #text is stored in this variable
      summWords = summarize(raw_text)
      st.subheader("Summary")
      #st.write(summWords)

if __name__ == '__main__':
   main()