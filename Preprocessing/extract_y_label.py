#!/usr/bin/env python
# coding: utf-8

# In[1]:


def extract_y_label(df):
    #get dummies for each tag
    tmp_df = pd.get_dummies(df['NAME'])
    #create label column
    tmp_df['label'] = ''
    tmp_df.label = tmp_df.values.tolist()
    df_new = pd.concat([df,tmp_df],axis=1)    
    return df_new


# In[ ]:




