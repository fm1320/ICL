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

##################################################################################################
# '''
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# import numpy as np
# import contextlib
# import plotly.express as px
# import pandas as pd
# from PIL import Image
# import datetime
# import os
# import psutil

# MODEL_DESC = {
    # 'Bart MNLI': """Bart with a classification head trained on MNLI.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
    # 'Bart MNLI + Yahoo Answers': """Bart with a classification head trained on MNLI and then further fine-tuned on Yahoo Answers topic classification.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
    # 'XLM Roberta XNLI (cross-lingual)': """XLM Roberta, a cross-lingual model, with a classification head trained on XNLI. Supported languages include: _English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili, and Urdu_.
# Note that this model seems to be less reliable than the English-only models when classifying longer sequences.
# Examples were automatically translated and may contain grammatical mistakes.
# Sequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
# }

# ZSL_DESC = """Recently, the NLP science community has begun to pay increasing attention to zero-shot and few-shot applications, such as in the [paper from OpenAI](https://arxiv.org/abs/2005.14165) introducing GPT-3. This demo shows how 🤗 Transformers can be used for zero-shot topic classification, the task of predicting a topic that the model has not been trained on."""

# CODE_DESC = """```python
# from transformers import pipeline
# classifier = pipeline('zero-shot-classification',
                      # model='{}')
# hypothesis_template = 'This text is about {{}}.' # the template used in this demo
# classifier(sequence, labels,
           # hypothesis_template=hypothesis_template,
           # multi_class=multi_class)
#{{'sequence' ..., 'labels': ..., 'scores': ...}}
# ```"""

# model_ids = {
    # 'Bart MNLI': 'facebook/bart-large-mnli',
    # 'Bart MNLI + Yahoo Answers': 'joeddav/bart-large-mnli-yahoo-answers',
    # 'XLM Roberta XNLI (cross-lingual)': 'joeddav/xlm-roberta-large-xnli'
# }



# device = 0 if torch.cuda.is_available() else -1

# @st.cache(allow_output_mutation=True)
# def load_models():
    # return {id: AutoModelForSequenceClassification.from_pretrained(id) for id in model_ids.values()}

# models = load_models()


# @st.cache(allow_output_mutation=True, show_spinner=False)
# def load_tokenizer(tok_id):
    # return AutoTokenizer.from_pretrained(tok_id)

# @st.cache(allow_output_mutation=True, show_spinner=False)
# def get_most_likely(nli_model_id, sequence, labels, hypothesis_template, multi_class, do_print_code):
    # classifier = pipeline('zero-shot-classification', model=models[nli_model_id], tokenizer=load_tokenizer(nli_model_id), device=device)
    # outputs = classifier(sequence, labels, hypothesis_template, multi_class)
    # return outputs['labels'], outputs['scores']

# def load_examples(model_id):
    # model_id_stripped = model_id.split('/')[-1]
    # df = pd.read_json(f'texts-{model_id_stripped}.json')
    # names = df.name.values.tolist()
    # mapping = {df['name'].iloc[i]: (df['text'].iloc[i], df['labels'].iloc[i]) for i in range(len(names))}
    # names.append('Custom')
    # mapping['Custom'] = ('', '')
    # return names, mapping

# def plot_result(top_topics, scores):
    # top_topics = np.array(top_topics)
    # scores = np.array(scores)
    # scores *= 100
    # fig = px.bar(x=scores, y=top_topics, orientation='h', 
                 # labels={'x': 'Confidence', 'y': 'Label'},
                 # text=scores,
                 # range_x=(0,115),
                 # title='Top Predictions',
                 # color=np.linspace(0,1,len(scores)),
                 # color_continuous_scale='GnBu')
    # fig.update(layout_coloraxis_showscale=False)
    # fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    # st.plotly_chart(fig)
# '''
#################################################################################################
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
      if (len(tmp1)>0):
        for k in range(len(tmp1)-1):
            sentc.append(tmp1[k])
    res_list = [assay[j] for j in index]
    #print("Nuovo:", nuovo)
    res_list=list(set(res_list))
    st.write("The assays mentioned are: \n ", res_list)
    sentc=list(set(sentc))
    st.write("Some sentences that mention an assay:", sentc)
    #st.write("Here are some sentences that mention an assay:")
    return sentc


def main():
   """A Simple NLP app with Spacy-Streamlit"""
   st.title("Text processing app for biological scientific papers")
   menu = ["Home","NER","Summarization","Zero shot learning"]
   choice = st.sidebar.selectbox("Menu",menu)
   if choice == "Home":
      
      link = '[GitHub page](https://github.com/fm1320/IC_NLP)'
      st.write("""This application was made as part of a postgradute program at Imeprial College London. The details about the traning of the models, data and the techniques can be found at my personal github page provided below.""")
      st.markdown(link, unsafe_allow_html=True)
      st.write("""<---- Choose and try out one of the NLP tasks available from the drop down menu on the left""")
      st.markdown("![Alt Text](https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Imperial_College_London_monotone_logo.jpg/320px-Imperial_College_London_monotone_logo.jpg)")
      st.write("*Text examples source: Garcia-Perez, I., Posma, J.M., Serrano-Contreras, J.I. et al. Identifying unknown metabolites using NMR-based metabolic profiling techniques. Nat Protoc 15, 2538–2567 (2020). https://doi.org/10.1038/s41596-020-0343-3")
      
      #st.subheader("Tokenization")
      #raw_text = st.text_area("Your Text","Enter Text Here")
      #docx = nlp(raw_text)
      #if st.button("Tokenize"):
      #   spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
      
   elif choice == "NER":
      st.subheader("Named Entity Recognition")
      # Add a selectbox to the sidebar:
      sel = st.sidebar.selectbox("Which NER model would you like to use ?", ["Spacy core en default","SpaCy Bloom embedding DL","String/Regex matching"])
      
      # if sel== "SciSpacy":
         #import scispacy
         # nlp = spacy.load("en_core_sci_sm")
      # elif sel=="DL small":
         # nlp = spacy.load('./BiA') #Location of directory of spacy model
      if sel=="SpaCy Bloom embedding DL":
         path=model_loader("https://github.com/fm1320/IC_NLP/releases/download/V3/V3-20210203T001829Z-001.zip", "V3")   
         nlp = spacy.load(path)
      elif sel=="Spacy core en default":
         import en_core_web_sm
         nlp = en_core_web_sm.load() 
      elif sel=="String/Regex matching":
         import en_core_web_sm
         nlp = en_core_web_sm.load() 
         #r_text = st.text_area("Enter text for entity recognition with Regex","Text here")
         r_text = st.text_area("Enter text for entity recognition with Regex","However, it is very challenging to elucidate the structure of all metabolites present in biofluid samples. The large number of unknown or unidentified metabolites with high dynamic concentration range, extensive chemical diversity and different physical properties poses a substantial analytical challenge. Metabolic profiling studies are often geared toward finding differences in the levels of metabolites that are statistically correlated with a clinical outcome, dietary intervention or toxic exposure when compared to a control group. The chemical assignment of this reduced panel of biologically relevant metabolites is possible using statistical spectroscopic tools9–11, two-dimensional (2D) NMR spectroscopic analysis12–14, separation and pre-concentration techniques11, various chromatographic and mass spectroscopy (MS)-based analytical platforms.")
         iz=finder(r_text,"")
         ######################################
         # '''
         # model_id = model_ids[model_desc]
         # ex_names, ex_map = load_examples(model_id)

         # st.title('Zero Shot Topic Classification')
         # sequence = st.text_area('Text', ex_map[example][0], key='sequence', height=height)
         # labels = st.text_input('Possible topics (separated by `,`)', ex_map[example][1], max_chars=1000)
         # multi_class = st.checkbox('Allow multiple correct topics', value=True)
         # hypothesis_template = "This text is about {}."
         # labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))
         # if len(labels) == 0 or len(sequence) == 0:
            # st.write('Enter some text and at least one possible topic to see predictions.')
            # return
         # if do_print_code:
            # st.markdown(CODE_DESC.format(model_id))

         # with st.spinner('Classifying...'):
            # top_topics, scores = get_most_likely(model_id, sequence, labels, hypothesis_template, multi_class, do_print_code)

         # plot_result(top_topics[::-1][-10:], scores[::-1][-10:])

         # if "socat" not in [p.name() for p in psutil.process_iter()]:
            # os.system('socat tcp-listen:8000,reuseaddr,fork tcp:localhost:8001 &')     
         # '''   
         ##########################################
      method = st.sidebar.selectbox("Choose input method (recommended:text box)", ["Text box", "URL"])   

      
      if method == "Text box" and sel !="Regex":
         raw_text = st.text_area("Enter text for entity recognition","However, it is very challenging to elucidate the structure of all metabolites present in biofluid samples. The large number of unknown or unidentified metabolites with high dynamic concentration range, extensive chemical diversity and different physical properties poses a substantial analytical challenge. Metabolic profiling studies are often geared toward finding differences in the levels of metabolites that are statistically correlated with a clinical outcome, dietary intervention or toxic exposure when compared to a control group. The chemical assignment of this reduced panel of biologically relevant metabolites is possible using statistical spectroscopic tools9–11, two-dimensional (2D) NMR spectroscopic analysis12–14, separation and pre-concentration techniques11, various chromatographic and mass spectroscopy (MS)-based analytical platforms.")   
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
      st.subheader("Enter the text you'd like to summarize (Here is an example that can be pasted in the text box!)")
      raw_text = st.text_area('''
    For over three decades, NMR spectroscopy has been widely applied in metabolic profiling and phenotyping1,2,3. The technology allows for accurate high-throughput screening of thousands of metabolites (small molecular species <1 kDa) present in a biological sample4,5,6,7, such as urine, plasma, feces, saliva and multiple types of tissues, as well as food8 and plant extracts. NMR spectroscopy provides robust multi-metabolite fingerprints of hundreds of metabolites in many biofluids, many of which are listed in spectral databases, particularly for common biofluids in urine and blood.
    However, it is very challenging to elucidate the structure of all metabolites present in biofluid samples. The large number of unknown or unidentified metabolites with high dynamic concentration range, extensive chemical diversity and different physical properties poses a substantial analytical challenge. Metabolic profiling studies are often geared toward finding differences in the levels of metabolites that are statistically correlated with a clinical outcome, dietary intervention or toxic exposure when compared to a control group. The chemical assignment of this reduced panel of biologically relevant metabolites is possible using statistical spectroscopic tools9,10,11, two-dimensional (2D) NMR spectroscopic analysis12,13,14, separation and pre-concentration techniques11, various chromatographic and mass spectroscopy 
    (MS)-based analytical platforms15,16 and existing spectral databases. However, the structural elucidation of NMR resonances relating to unknown molecules remains a major bottleneck in metabolic profiling studies. As a result, many published NMR-based metabolic profiling studies still continue to include putatively identified metabolites and unknown features without providing unequivocal proof of assignment, or they simply label peaks as ‘unknown’, thereby potentially missing key mechanistic information.
    To avoid the problem of multiple entries for the same compound in databases under different names, a community-wide effort is underway to develop better, faster and more standardized metabolite identification strategies, such as implementing standard nomenclature for newly identified metabolites using the International Chemical Identifier (InChI)17. Sumner et al. proposed a four-level system18 for assigning a confidence level to newly identified metabolites in metabolic profiling studies: 1) positively identified compounds (with a name, a known structure, a CAS number or an InChI); 2) putatively annotated compounds using spectral similarity with databases but without chemical reference standard; 3) putatively identified chemicals within a compound class; and 4) unknown compounds. Wishart et al. proposed a further distinction for those metabolites: the ‘known unknowns’ and the ‘unknown unknowns’19.
    A ‘known unknown’ corresponds to a metabolite that has not yet been identified in the sample of interest but that has been previously described in a database or in the literature, whereas a truly new compound, an ‘unknown unknown’, has never been described or formally identified.
    Commercial packages, such as Bruker’s AMIX TM software, and open-source software20, such as COLMAR (http://spinportal.magnet.fsu.edu/), can help with identifying these ‘known unknowns’, and some of these software applications are capable of automatically or semi-automatically annotating a limited number of compounds in a biological sample. However, even with automated annotation, the software still requires manual revision and can be prone to inconsistent interpretation and assignment by different individuals19. Most software packages and databases do not support identification of ‘unknown unknowns’, although a few platforms, such as AMIX, include prediction software to aid the identification of new compounds.
    Open-access databases have been created for researchers to deposit information relating to newly identified compounds. Most of the available databases, such as the Human Metabolome Database (HMDB)21, the BioMagResBank (BMRB)22, PRIMe server23, COLMAR 1H(13C)-TOCCATA and Bruker-AMIX (http://www.bruker-biospin.com/amix.html), contain chemical shift values, relative intensity and peak shape information for 1H-NMR and often 13C-NMR data to support metabolite identification. However, all databases contain inherent errors, such as incorrect structures for the metabolites, incorrect names and incorrect assigments. This problem is compounded further by the effect that experimental conditions, such as the pH or ionic content of the sample, can have on the chemical shift of a metabolite.
    Some of these databases, such as HMDB, provide complementary information, including MS assignments, which can be useful for checking potential errors in assignments of NMR peaks. However, although there are resources available to aid assignment of candidate biomarkers, there is no panacea for accurate metabolite identification, and there remains a clear unmet need for improved strategies for metabolite identification and curation for NMR spectral profiling.  
      ''') #text is stored in this variable
      summWords = summarize(raw_text)
      st.subheader("Summary")
      st.write(summWords)
      
      
   elif choice == "Zero shot learning":
      st.write("""Due to resource constraints, this demo is moved to the link below:""")    
      link = '[Zero shot learning for NER demo](https://colab.research.google.com/drive/1zKDbjLo9vyEuSRotSSVwFLyaA61o1ceG#scrollTo=hkfE6NRA0Dzy)'
      st.markdown(link, unsafe_allow_html=True) 
    

if __name__ == '__main__':
   main()