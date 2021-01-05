import os
import json
from bs4 import BeautifulSoup
doc_dir_name = r"D:\MWAStar\maintext"
abb_dir_name = r"D:\MWAStar\abbreviations"
output_path = r"D:\MWAStar\out"
'''
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".mat"):
        os.remove(os.path.join(dir_name, item))
'''		

def replace_abb():
	abb_dict={}
	for filename in os.listdir(abb_dir_name):
		if filename.endswith(".json"):
			fullpath = os.path.join(abb_dir_name, filename)
			with open(fullpath, "rb") as read_file:   # read json file
				data= json.load(read_file) 
				temp=data.get("all_abbreviations")
				abb_dict.update(temp)
				json.dump( abb_dict, open( r"D:\MWAStar\out\abbs.json", 'w' ) )
	# Serialize data into file:
	#json.dump( abb_dict, open( "file_name.json", 'w' ) )		
			
def extract(): 
	train=[]
	temp=[]
	raw_text=[]
	for i, filename in enumerate(os.listdir(doc_dir_name)):
		if filename.endswith(".json"):
			fullpath = os.path.join(doc_dir_name, filename)
			with open(fullpath, "rb") as read_file:   # read json file
				data= json.load(read_file) 	
				temp.append(data.get("paragraphs"))
				#print(i, data.get("title"))
				# index out of range, da ne interferira so drugiot for ?
	
	#{d['body']:d['IAO_term'] for d in temp}
	for i in range (0, len(temp)):
		for j in range (0, len(temp[i])):
			if ("materials" in (str(temp[i][j].get("IAO_term"))) ) or ("methods" in (str(temp[i][j].get("IAO_term")))) or ("conclusion" in (str(temp[i][j].get("IAO_term")))):
				raw_text.append(str(temp[i][j].get("body")))

	print (len(raw_text))
	#print("section" in iao[2])
	#[(d[j]['body'],d[j]['IAO_term']) for d[j] in temp[j]]
	#print(temp[1344].get("IAO_term")
	#print((temp[-1][1]))
	#print(temp[number of article][element in list])
	#print((temp[0][2].get("IAO_term")[0]))
	 
	 #[0]First article 
	#print(type(temp[1].get("IAO_term")))	
"""				
	for i in range (1,len(temp)):
		if (temp[i].get("IAO_term") in "materials")|(temp[i].get("IAO_term") in "methods")|(temp[i].get("IAO_term") in "conclusion"): 
			train.append(temp[i].get("body")) #train is a list of all body sections							
			with open(r'D:\MWAStar\out\your_file.txt', 'w') as f:
				for item in my_list:
					f.write("%s\n" % item)
								
								
				#for d in thisismylist:  # AKO E LIST OF DICIONAIRES OVA 
				#	print d['Name']				
	"""		     
			
	


#replace_abb()
extract()
print("Working")


#extension = filename.split("_")[0]




#print(data_train,type(data_train))
#print((train))  