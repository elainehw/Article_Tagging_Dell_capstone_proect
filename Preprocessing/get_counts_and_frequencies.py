import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import text_analysis




#get list of highest tfidf scores in dataset by article body or tag
#df: type DataFrame
#list_size: type int, in range [0,inf)
#article: type boolean
#return: type list

def get_tfidf(df, list_size, article=True):

	# train vectorizer using all the documents
	vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
	if (article):
		X = vectorizer.fit_transform(df['Body_Lemm_lower'].dropna())
		# combine all the body words into a single list
		wordstr = ' '.join(str(word) for word in df['Body_Token'])
	else:
		X = vectorizer.fit_transform(df['Title_Lemm_lower'].dropna())
		# combine all the title words into a single list
		wordstr = ' '.join(str(word) for word in df['Title_Token'])

	# transform the whole list
	response = vectorizer.transform([wordstr])

	# create a df containing word and its tfidf score
	feature_names = vectorizer.get_feature_names()
	word_list = []
	score_list = []
	for col in response.nonzero()[1]:
		word_list.append(feature_names[col])
		score_list.append(response[0, col])
	df_tfidf = pd.DataFrame({'word': word_list, 'score': score_list})
	top_word_list = df_tfidf.sort_values(by='score', ascending=False)['word'].head(list_size).tolist()

	return top_word_list



#get lists of keywords in articles and titles using tfidf by tag and the set of tags being modeled
#train: type DataFrame, training dataset
#df: type DataFrame, Prod_FinalData
#article_word_count: type int, in range [0,inf)
#title_word_count: type int, in range [0, inf)
#return:
	#1): type list, keywords in article body
	#2): type list, keywords in article title

def get_keywords_by_tag(train, df, article_word_count, title_word_count):
	print('Identifying Keywords in Training Set by Tag...')

	df_tags = pd.read_csv('Training_Tag_Data.csv')

	#Filter Prod_FinalData by relevant tags
	tag_list = list(set(df_tags['Tag']))
	df = df[df['NAME'].isin(tag_list)]

	#merge training dataset and Prod_FinalData, clean text
	df_tfidf_by_tag = pd.merge(train[['RECORDID', 'Body_Lemm_lower', 'Body_Token', 'Title_Lemm_lower', 'Title_Token']], df[['RECORDID', 'NAME']], how='left', on='RECORDID')
	df_tfidf_by_tag['Body_Lemm_lower'] = df_tfidf_by_tag['Body_Lemm_lower'].astype(str)
	df_tfidf_by_tag['Title_Lemm_lower'] = df_tfidf_by_tag['Title_Lemm_lower'].astype(str)

	#aggregate article text by tag
	dft1 = df_tfidf_by_tag.groupby('NAME')['Body_Lemm_lower'].apply(' '.join).reset_index()
	dft2 = df_tfidf_by_tag.groupby('NAME')['Body_Token'].sum()
	df_tfidf_by_tag_a = pd.merge(dft1, dft2, how='inner', on='NAME')

	#aggregate title text by tag
	dft1 = df_tfidf_by_tag.groupby('NAME')['Title_Lemm_lower'].apply(' '.join).reset_index()
	dft2 = df_tfidf_by_tag.groupby('NAME')['Title_Token'].sum()
	df_tfidf_by_tag_t = pd.merge(dft1, dft2, how='inner', on='NAME')

	#get keyword lists
	article_word_list = get_tfidf(df_tfidf_by_tag_a, list_size=article_word_count, article=True)
	title_word_list = get_tfidf(df_tfidf_by_tag_t, list_size=title_word_count, article=False)

	#clean tags
	df_tags['Tag'] = df_tags['Tag'].apply(text_analysis.clean_text).apply(text_analysis.remove_stopwords_english, args = (nltk.corpus.stopwords.words('english'),))
	df_tags['Tag'] = df_tags['Tag'].str.lower()

	#concatenate tfidf keywords and tags in target variable
	article_word_list = article_word_list + list(df_tags['Tag'])
	title_word_list = title_word_list + list(df_tags['Tag'])

	print('Keywords Identified')
	print()

	#ensure list values unique using set()
	return list(set(article_word_list)), list(set(title_word_list))



#get lists of keywords in articles and titles using tfidf by article and the set of tags being modeled
#train: type DataFrame, training dataset
#article_word_count: type int, in range [0,inf)
#title_word_count: type int, in range [0, inf)
#return:
	#1): type list, keywords in article body
	#2): type list, keywords in article title

def get_keywords_by_article(train, article_word_count, title_word_count):
	print('Identifying Keywords in Training Set by Article...')

	#get keyword lists
	article_word_list = get_tfidf(train, list_size=article_word_count, article=True)
	title_word_list = get_tfidf(train, list_size=title_word_count, article=False)

	#get all training tags, clean text
	df_tags = pd.read_csv('Training_Tag_Data.csv')
	df_tags['Tag'] = df_tags['Tag'].apply(text_analysis.clean_text).apply(text_analysis.remove_stopwords_english, args = (nltk.corpus.stopwords.words('english'),))
	df_tags['Tag'] = df_tags['Tag'].str.lower()

	#concatenate tfidf keywords and tags in target variable
	article_word_list = article_word_list + list(df_tags['Tag'])
	title_word_list = title_word_list + list(df_tags['Tag'])

	print('Keywords Identified')
	print()

	#ensure list values unique using set()
	return list(set(article_word_list)), list(set(title_word_list))



#count the number of times a keyword appears in a string, used with the .apply() method
#wordstr: type String
#keyword: type String
#return: type int

def get_keyword_count(wordstr, keyword):
	return wordstr.count(keyword)



#gets counts and frequencies of keywords in article body and title
#df: type DataFrame, train or test
#article_keywords: type list
#title_keywords: type list
#return: type DataFrame

def get_counts_and_frequencies(df, article_keywords, title_keywords):
	print('Getting Counts and Frequencies of Keywords...')
	for i in article_keywords:
		df[i+"_a_c"] = df['Body_Lemm_lower'].astype(str).apply(get_keyword_count, args=(i,))
		df[i+"_a_f"] = df[df.columns[-1]]/df['Body_Length']

	for i in title_keywords:
		df[i+"_t_c"] = df['Title_Lemm_lower'].astype(str).apply(get_keyword_count, args=(i,))
		df[i+"_t_f"] = df[df.columns[-1]]/df['Title_Length']

	print('Calculations Complete')
	print()

	return df
