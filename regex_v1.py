# Beta V 1.0
'''
This code reads into a list all the assay methods from a csv file.
A function takes a text as an input an a python list of assays. 
The function performs matching of all sentences that contain a word which 
is in the list. This is done using regular expression, the word is found 
and then the expression looks for puntcuation signs left from the word and 
right from the word so that it can extract the sentence which contains the word.

The user can choose the given assay when running the function, 
or if he doesn't input a parameter it will go through the list of all assays in the file.

'''
import re #time complexity issues of finite state machines ?
import pandas as pd
#file_path="/content/drive/My Drive/Imperial college London/all_assays.csv"
file_path="./all_assays.csv"
assay=[]
df = pd.read_csv(file_path,dtype= str) # READ CSV AS STRING !!!
assay = df.values.tolist()
output=[]

def finder(text,user_assay=None):    
    if (user_assay==None): 
      for i in range(len(assay)):
       out2=re.findall(r'[^.?!]*(?<=[.?\s!])%s(?=[\s.?!])[^.?!]*[.?!]' % assay[i] , text ,  flags=re.IGNORECASE )
       print('\n'.join(out2))
       output.append(out2) # just a list with all the results
      return 0
    # if the user chooses a custom assay not from the list
    else:  
     out2=re.findall(r'[^.?!]*(?<=[.?\s!])%s(?=[\s.?!])[^.?!]*[.?!]' % user_assay , text )
     if (len(out2) == 0):
      print("No matching assays")
     else:
      output.append(out2) 
      print("At least one match was found :")
      print('\n'.join(out2))

# Run the program outside of the function    
text = ('Metabolomics is used to determine the metabolic profile of biological samples, identify specific biomarkers,' 
       'and explore possible metabolic pathways. It has been used during drug development [1], '
       'and in clinical disease research [2, 3], pathology [4], toxicology [5] and nutrition studies [6].' 
       'Metabolomics mainly utilizes NMR spectroscopy [7], liquid chromatography (LC)–mass spectrometry [8] and' 
       'gas chromatography–mass spectrometry [9] to analyze and evaluate biological specimens. '
       'Each analytical technique has its own advantages and shortcomings. none of them can be used individually' 
       'to systematically and accurately identify metabolites in complex biological matrices. '
       'Since accurate metabolite identification directly determines the usefulness of the metabolomic analysis,' 
       'metabolite identification has gained increased attention from the metabolomics research community. '
       '1H NMR spectroscopy is often used for metabolomics research. As all 1H nucleuses have the same sensitivity,' 
       'the reproducibility of NMR spectroscopy is typically high.' )   
x=finder(text, "") # You should write "" instead of None for the input of the function
