#required libraries: pandas, sklearn, bs4, nltk, collections, re

#run the following four lines if the below objects are not present in your environment
# import nltk
# nltk.download('tagsets')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

import pandas as pd
import shape_data
import text_analysis
import get_counts_and_frequencies



#either removes missing values or inserts '' for string and 0 for numeric
#df: type DataFrame
#remove: type = boolean
#return: type DataFrame

def missing_values(df, remove=False):
	df['Missing_Value'] = 0

	df_c = df[df['Body'].notna()]
	df_m = df[df['Body'].isna()]

	if(remove):
		df_c = df_c.drop(['Missing_Value'], axis = 1)
		df_c = df_c.fillna(0)
		df_m = df_m.fillna(0)
		df_m.to_csv('missing_article_bodies.csv', index=False)
		return df_c
	else:
		df_m['Body'] = ''
		df_m['Body_Lemm'] = ''
		df_m['Body_Lemm_lower'] = ''
		df_m['Body_Length'] = 0
		df_m.iloc[:, 12:] = 0
		df_m['Missing_Value'] = 1
		df_c = df_c.fillna(0)
		df_m = df_m.fillna(0)
		return pd.concat([df_c, df_m])


print('Importing data...') 
df1 = pd.read_csv('Prod_FinalData2.csv')
df2 = pd.read_excel('XML.xlsx', header=1)
df3 = pd.read_csv('PublicArticlesWithEndDate.csv', encoding="ISO-8859-1")
print('Import Complete')
print()

unwanted_tags = ['Bulk Archive', 'External Content Archive', 'Not Searchable in FAST', 'Client / Commercial']
#unwanted_tags = ['Bulk Archive', 'External Content Archive', 'Not Searchable in FAST', 'Client / Commercial', 'IPS / PG', 'PSQN']

train, test = shape_data.shape_data(df1, df2, df3, unwanted_tags, language = 'en_US', tag_count_max = 20, train_prop = 0.8, tag_freq_ceil = 200, store_tag_data=True)

print('Training Set:')
train = text_analysis.text_analysis(train)
train = missing_values(train, remove = True)

print('Test Set:')
test = text_analysis.text_analysis(test)
test = missing_values(test, remove = True)

#Keywords
by_tag = True
if(by_tag):
	article_word_list, title_word_list = get_counts_and_frequencies.get_keywords_by_tag(train, df1, article_word_count=500, title_word_count=100)
else:
	article_word_list, title_word_list = get_counts_and_frequencies.get_keywords_by_article(train, article_word_count=500, title_word_count=100)
train = get_counts_and_frequencies.get_counts_and_frequencies(train, article_keywords=article_word_list, title_keywords=title_word_list)
test = get_counts_and_frequencies.get_counts_and_frequencies(test, article_keywords=article_word_list, title_keywords=title_word_list)

train.to_csv('train_tfidf_by_article.csv', index=False)
test.to_csv('test_tfidf_by_article.csv', index=False)
