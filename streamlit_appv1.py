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
#import pdfplumber
import en_core_web_md
#import zipfile
#!python -m spacy download en_core_web_lg
@st.cache(suppress_st_warning=True)
def model_loader(link,foldername):
  """
  returns path of zipped folder with trained spacy model
  
  """
  import requests
  import zipfile
  import tempfile
  import spacy

  dir=tempfile.gettempdir()


  #link= "https://github.com/fm1320/IC_NLP/releases/download/V3/V3-20210203T001829Z-001.zip"

  results = requests.get(link)
  #with open(dir, 'wb') as f:
  fp = tempfile.TemporaryFile()  
  fp.write(results.content)


  file = zipfile.ZipFile(fp)
  with tempfile.TemporaryDirectory() as tmpdirname:
    file.extractall(path=dir)

  #print(dir)
  end_path=os.path.join(dir, foldername)
  files = os.listdir(end_path)
  #for file in files:
    #print(file)
  return end_path


def finder(text,user_assay):    
    import re
    import pandas as pd
    file_path="./all_assays.csv"
    assay=[]
    df = pd.read_csv(file_path,dtype= str, encoding='latin1') # READ CSV AS STRING !!!
    assay = df['1H NMR'].values.tolist()
    assay = list(map(''.join, assay)) # convert list of lists to list of strings 
    nuovo=[]
    pattern1=r'[^.?!]*(?<=[.?\s!])%s(?=[\s.?!])[^.?!]*[.?!]' #extracts full sentences that contain a word 
    pattern2=r'\b%s\b' #extracts a given word 
    index=[]
    sentc=[]
    for i in range(len(assay)):
      tmp=re.findall(pattern2 %assay[i],text, flags=re.IGNORECASE)  
      if (len(tmp)>0):
       index.append(i)
       nuovo.append(tmp)
      tmp1=re.findall(pattern1 %assay[i],text, flags=re.IGNORECASE)
      #st.write("Sentences that have the assay:" ,tmp1)
      sentc.append(tmp1)       
    res_list = [assay[j] for j in index]
    #print("Nuovo:", nuovo)
    st.write("The assays mentioned are: \n ", sentc)
    return sentc

def main():
   """A Simple NLP app with Spacy-Streamlit"""
   st.title("Text processing app for biological scientific papers")
   menu = ["Home","NER","Summarization"]
   choice = st.sidebar.selectbox("Menu",menu)
   if choice == "Home":
      
      link = '[GitHub page](https://github.com/fm1320/IC_NLP)'
      st.write("""This application was made as part of a postgradute program at Imeprial College London. The details about the traning of the models, data and the techniques can be found at my personal github page provided below.""")
      st.markdown(link, unsafe_allow_html=True)
      st.write("""<---- Choose and try out one of the NLP tasks available from the drop down menu on the left""")
      st.markdown("![Alt Text](https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Imperial_College_London_monotone_logo.jpg/320px-Imperial_College_London_monotone_logo.jpg)")
      
      
      #st.subheader("Tokenization")
      #raw_text = st.text_area("Your Text","Enter Text Here")
      #docx = nlp(raw_text)
      #if st.button("Tokenize"):
      #   spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
      
   elif choice == "NER":
      st.subheader("Named Entity Recognition")
      # Add a selectbox to the sidebar:
      sel = st.sidebar.selectbox("Which NER model would you like to use ?", ["SciSpacy", "DL small", "Spacy core en","DL medium","Regex"])
      
      if sel== "SciSpacy":
         #import scispacy
         nlp = spacy.load("en_core_sci_sm")
      elif sel=="DL small":
         nlp = spacy.load('./BiA') #Location of directory of spacy model
      elif sel=="Spacy core en":
         import en_core_web_sm
         nlp = en_core_web_sm.load() 
      elif sel=="DL medium":
         path=model_loader("https://github.com/fm1320/IC_NLP/releases/download/V3/V3-20210203T001829Z-001.zip", "V3")   
         nlp = spacy.load(path)
      elif sel=="Regex":
         r_text = st.text_area("Enter text for entity recognition with Regex","Text here")
         iz=finder(r_text,"")
		 st.write("Sentences with ASSAY:",iz)
      method = st.sidebar.selectbox("Choose input method (recommended:text box)", ["Text box", "URL"])   

      
      if method == "Text box" and sel !="Regex":
         raw_text = st.text_area("Enter text for entity recognition","Text here")   
         docx = nlp(raw_text)
         spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

      if method == "URL" and sel !="Regex":
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
      st.write(summWords)

if __name__ == '__main__':
   main()