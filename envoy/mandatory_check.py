import yaml, json
import pandas as pd

# Read the configuration file
with open('mandatory_fields.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Opening JSON file
f = open('map_1.json')

# returns JSON object as 
# a dictionary
data = json.load(f)

# Iterating through the json
list1=[]
mappings="{"
for i in data['mapping']:
           mylocal=json.dumps(i['local']).strip('"')
           myglobal=json.dumps(i['global']).strip('"')
           list1+=[mylocal]
           mappings+='"'+mylocal+'":"'+myglobal+'",'

mappings=mappings[:-1]
mappings+="}"

df = pd.read_csv("./dataset_feature_mapping/train_1_172.csv",header=0,usecols=list1) 
df.rename(columns=json.loads(mappings),inplace=True)
f.close()
        
# Check if all mandatory fields are present
if not all(field in df.columns for field in config['mandatory_fields']):
    raise ValueError('Some mandatory fields are missing')
    

