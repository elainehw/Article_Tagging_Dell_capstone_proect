import pandas as pd
from sklearn.model_selection import train_test_split
from text_analysis import get_article_length




#filters FinalData by language
#df1: type DataFrame
#langauge: type string or None
#return: type DataFrame

def filter_language(df1, language=None):
	if(language==None):
		return df1
	else:
		return df1[df1['LOCALEID']==language]



#joins FinalData and XML, filters by langauge, removes columns
#df1: type DataFrame, FinalData dataset
#df2: type DataFrame, XML dataset
#language: type string or None
#return: type DataFrame

def data_merge(df1, df2, language=None):

	df1 = df1.drop(['Unnamed: 0'], axis=1)
	df1 = df1.drop_duplicates(subset='RECORDID') #keep one record of article

	df1 = filter_language(df1, language) #filter by language

	df_f = pd.merge(df1, df2, how='left', on='RECORDID') #join XML and FinalData2
	
	if(language==None): #filter DataFrame by language
		df_f = df_f[['RECORDID', 'LOCALEID', 'Tags', 'INDEXMASTERIDENTIFIERS', 'XML']]
	else:
		df_f = df_f[['RECORDID', 'Tags', 'INDEXMASTERIDENTIFIERS', 'XML']] #for language filtered, remove LOCALEID
	
	return df_f



#converts Tag field to list, used with .apply() method
#s: type string
#return: type list

def tags_to_list(s):
	l = s.split(',') #split Tag string by comma

	for i in range(len(l)):
		l[i] = l[i].lstrip().rstrip()

	return l



#removes rows with tags greater than a threshold, saves eliminated rows to csv
#df: type DataFrame, full dataset
#tag_count_max: type int or None, in range [0,inf)
#return: type DataFrame

def remove_high_tags(df, tag_count_max):
	if tag_count_max == None:
		return df
	else:
		df['Tag_Count'] = df['Tags'].apply(get_article_length)
		df[df['Tag_Count']>tag_count_max].to_csv('high_count_tags.csv', index=False)
		return df[df['Tag_Count']<=tag_count_max]



#remove tags requested to not be included in the model, used with .apply() method
#l: type list
#unwanted_tags: type list
#return: type list

def remove_unwanted_tags(l, unwanted_tags):
	new_tags = []

	for i in l:
		if(i not in unwanted_tags):
			new_tags.append(i)

	return new_tags



#remove tags not in a provided list
#l: type list
#wanted_tags: type list
#return: type list

def keep_wanted_tags(l, wanted_tags):
	new_tags = []

	for i in l:
		if(i in wanted_tags):
			new_tags.append(i)

	return new_tags



#get list of unique expired tags
#df3: type DataFrame, XML dataset with DISPLAYENDDATE field
#return: type list, unique values

def get_expired_tags(df3): #get list of expired tags
	df3 = df3[['NAME', 'DISPLAYENDDATE']]
	df3 = df3[df3['DISPLAYENDDATE']!='31-DEC-99 12.00.00.000000000 AM']
	return list(set(df3['NAME']))



#returns list of most occurring tags, limited by tag_freq_ceil
#df_train: type DataFrame, list of tags determined by training dataset
#tag_freq_ceil: type int or None
#return: type list

def get_tag_frequencies(df_train, tag_freq_ceil):
	tags_d = {}

	for tag_list in range(len(df_train)): #create counter dictionary
		for tag in df_train['Tags'].iloc[tag_list]:
			if not tag in tags_d:
				tags_d[tag] = 1
			else:
				tags_d[tag] += 1

	tags = list(tags_d.keys())
	counts = list(tags_d.values())

	if tag_freq_ceil==None: #if no limit, return all tags
		return tags
	else:                   #if limit, return the highest occuring tags
		df_tags = pd.DataFrame({'Tag': tags, 'Count': counts}).sort_values(by='Count', ascending=False).head(tag_freq_ceil)
		return list(df_tags['Tag'])



#vectorize tag field, each element binary, used with .apply(method)
#l: type list
#train_tags: type list, the set of tags to be used in model
#return: type list

def vectorize_tags(l, train_tags): #vectorize tag field
	vectorized = [0]*len(train_tags)

	for i in range(len(train_tags)):
		if train_tags[i] in l:
			vectorized[i] = 1

	return vectorized



#create csv file containing tag data
#df: type DataFrame, training dataset
#train_tags: type list, the set of tags to be used in model

def store_tag_info(df, train_tags): #create csv with tag metadata
	tag_count = []

	for i in range(len(train_tags)):
		sum = 0
		for j in range(len(df)):
			sum += df['Vec_Tags'].iloc[j][i]
		tag_count.append(sum/len(df))

	tag_info = pd.DataFrame({'Tag': train_tags, 'Frequency': tag_count})
	tag_info.to_csv('Training_Tag_Data.csv', index=False)



#call function to run script: joins FinalData and XML, creates training and test datasets, filters tags, creates csv with tag data
#df1: type DataFrame, FinalData dataset
#df2: type DataFrame, XML dataset
#df3: type DataFrame, XML dataset with dates
#unwanted_tags: type list, list of tags to be removed from model
#language: type string or None
#train_prop: type float, in range (0,1)
#tag_freq_ceil: type int or None, in range (0, inf)
#store_tag_data: boolean
#return:
	#1): type DataFrame, training dataset
	#2): type DataFrame, test dataset

def shape_data(df1, df2, df3, unwanted_tags, language=None, train_prop=0.8, tag_count_max = None, tag_freq_ceil=None, store_tag_data=False):
	print('Shaping data and transforming target variable...')
	df_f = data_merge(df1, df2, language) #merge FinalData and XML

	df_f['Tags'] = df_f['Tags'].apply(tags_to_list) #convert tag field (string) to list

	df_f = remove_high_tags(df_f, tag_count_max)

	X = df_f.drop(['Tags'], axis=1)
	y = df_f['Tags']

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop, random_state = 311) #split dataset into train and test
	df_train = X_train.join(y_train)
	df_test = X_test.join(y_test)

	

	df_train['Tags'] = df_train['Tags'].apply(remove_unwanted_tags, args=(unwanted_tags,)) #remove discussed tags from model

	#Issue Here: too many high frequency tags removed by expiration; leads to data sparsity
	expired_tags = get_expired_tags(df3)
	df_train['Tags'] = df_train['Tags'].apply(remove_unwanted_tags, args=(expired_tags,)) #remove expired  tags from model

	train_tags = get_tag_frequencies(df_train, tag_freq_ceil) #get highest frequency tags
	df_train['Tags'] = df_train['Tags'].apply(keep_wanted_tags, args=(train_tags,)) #include only high frequency tags in training set
	df_test['Tags'] = df_test['Tags'].apply(keep_wanted_tags, args=(train_tags,)) #include only high frequency training tags in test set

	df_train['Vec_Tags'] = df_train['Tags'].apply(vectorize_tags, args=(train_tags,)) #vectorize tags
	df_test['Vec_Tags'] = df_test['Tags'].apply(vectorize_tags, args=(train_tags,)) #vectorize tags

	print('Shaping Complete')
	print()

	if(store_tag_data):
		print('Producing Tag Data...')
		store_tag_info(df_train, train_tags)
		print('Tag data sucessfully output to csv.')
		print()

	return df_train, df_test

