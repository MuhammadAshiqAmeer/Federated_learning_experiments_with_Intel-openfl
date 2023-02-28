# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Shard Descriptor template.

It is recommended to perform tensor manipulations using numpy.
"""

#Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""
from typing import List, Tuple
import logging
import pandas as pd
import json,yaml
import os
from typing import List
import tensorflow as tf
import numpy as np
import requests
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor

logger = logging.getLogger(__name__)

class neuroblastomaShardDataset(ShardDataset):
    """Lem-Mel Shard dataset class."""

    def __init__(self, x, y, data_type):
        """Pick rank-specific subset of (x, y)"""
        self.data_type = data_type
        self.x=x
        self.y=y
        #self.x = x[self.rank - 1::self.worldsize]
        #self.y = y[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index], self.y[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)
    
class neuroblastomaShardDescriptor(ShardDescriptor):
    """Lem-Mel Shard descriptor class."""

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = self.load_prepare_data()

        self.data_by_type = {
            'train': (x_train, y_train),
            'val': (x_test, y_test)
        }


    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return neuroblastomaShardDataset(
            *self.data_by_type[dataset_type],
            data_type=dataset_type
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['9']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @staticmethod
    def load_prepare_data() -> Tuple[tf.data.Dataset]:
        """Load and prepare dataset."""
                  
        # Opening JSON file for mapping
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
    
        
        # Preprocessing
        #df["Gender"]=df["Gender"].replace({"Male":1,"Female":0})
        df["Vital Status"]=df["Vital Status"].replace({"Dead":0,"Alive":1})
        df["INSS Stage"]=df["INSS Stage"].replace({"Stage 1":0,"Stage 2a":1,"Stage 2b":2,"Stage 3":3,"Stage 4":4,"Stage 4s":5,})
        df["MYCN status"]=df["MYCN status"].replace({"Not Amplified":0,"Amplified":1})
        df["Histology"]=df["Histology"].replace({"Favorable":0,"Unfavorable":1})
        df["MKI"]=df["MKI"].replace({"Low":0,"High":1,"Intermediate":2})
        df["COG Risk Group"]=df["COG Risk Group"].replace({"Low Risk":0,"Intermediate Risk":1,"High Risk":2})
        df["Overall Survival Time in Years"] = round(df["Overall Survival Time in Days"]/365,0)
        df["Age at Diagnosis in years"] = round(df["Age at Diagnosis in Days"]/365,0)
        df["Event Free Survival Time in years"] = round(df["Event Free Survival Time in Days"]/365,0)
        x=df.drop(['Vital Status','Overall Survival Time in Days','Event Free Survival Time in Days','Age at Diagnosis in Days'],axis=1)
        y=df['Vital Status']

        df_t= pd.read_csv("./dataset_feature_mapping/test_1_74.csv",header=0,usecols=list1) 
        df_t.rename(columns=json.loads(mappings),inplace=True)
        f.close()

        df_t["Vital Status"]=df_t["Vital Status"].replace({"Dead":0,"Alive":1})
        df_t["INSS Stage"]=df_t["INSS Stage"].replace({"Stage 1":0,"Stage 2a":1,"Stage 2b":2,"Stage 3":3,"Stage 4":4,"Stage 4s":5,})
        df_t["MYCN status"]=df_t["MYCN status"].replace({"Not Amplified":0,"Amplified":1})
        df_t["Histology"]=df_t["Histology"].replace({"Favorable":0,"Unfavorable":1})
        #df_t["Gender"]=df_t["Gender"].replace({"Male":1,"Female":0})
        df_t["MKI"]=df_t["MKI"].replace({"Low":0,"High":1,"Intermediate":2})
        df_t["COG Risk Group"]=df_t["COG Risk Group"].replace({"Low Risk":0,"Intermediate Risk":1,"High Risk":2})
        df_t["Overall Survival Time in Years"] = round(df_t["Overall Survival Time in Days"]/365,0)
        df_t["Age at Diagnosis in years"] = round(df_t["Age at Diagnosis in Days"]/365,0)
        df_t["Event Free Survival Time in years"] = round(df_t["Event Free Survival Time in Days"]/365,0)
        x_t=df_t.drop(['Vital Status','Overall Survival Time in Days','Event Free Survival Time in Days','Age at Diagnosis in Days'],axis=1)
        y_t=df_t['Vital Status']

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(x)
        test_scaled = scaler.fit_transform(x_t)
       
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_scaled, test_scaled, y, y_t
        y_train=np.asarray(y_train).reshape((-1,1))
        y_test=np.asarray(y_test).reshape((-1,1))
   
   
        print('Data was loaded!')
        return (x_train, y_train), (x_test, y_test)
