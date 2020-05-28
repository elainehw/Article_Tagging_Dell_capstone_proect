import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import vader
from collections import Counter
import re

replace_by_space_re = re.compile('[/(){}\[\]\[@,;]]')
bad_symbols_re = re.compile('[^0-9a-zA-Z #+_]')




#pull text from HTML, replace symbols and punctuation, used with .apply() method
#text: type string
#return: type string

def clean_text(text):
	text = str(text)
	text = BeautifulSoup(text, 'lxml').text
	text = replace_by_space_re.sub(' ', text)
	text = bad_symbols_re.sub(' ', text)
	text = text.lstrip().rstrip()

	return text



#clean title text, replace symbols and punctuation, used with .apply() method
#text: type string
#return: type string

def clean_title(text):
	text = str(text)
	text = replace_by_space_re.sub(' ', text)
	text = bad_symbols_re.sub(' ', text)
	text = text.lstrip().rstrip()

	return text


#remove stopwords from text, used with .apply() method
#text: type string
#sw: type list
#return: type string

def remove_stopwords_english(text, sw):
	return ' '.join([word for word in text.split() if word not in sw])



#pos_tag for lemmatization, output used in get_lemmatized_text()
#treebank_tag: type string
#return: nltk.corpus.wordnet object or None

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return nltk.corpus.wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return nltk.corpus.wordnet.VERB
	elif treebank_tag.startswith('N'):
		return nltk.corpus.wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return nltk.corpus.wordnet.ADV
	else:
		return None



#lemmatize text based on pos_tag
#text: type string
#return: type string

def get_lemmatized_text(text):
	text = nltk.tokenize.word_tokenize(text)
	tagged = nltk.pos_tag(text)

	lemmatizer = nltk.stem.WordNetLemmatizer()
	text = []

	for word, tag in tagged:
		wtag = get_wordnet_pos(tag)
		if wtag is None:
			text.append(lemmatizer.lemmatize(word))
		else:
			text.append(lemmatizer.lemmatize(word, pos=wtag))

	return ' '.join([word for word in text])



#apply adjustments to text preprocessing, used with .apply() method
#text: type string
#return: type string

def adjust_keywords(text):
	return text.replace(' e mail', ' email')



#tokenize text, used with .apply() method
#s: type string
#return: type list

def tokenize_article(s):
	return nltk.word_tokenize(s)



#get approximate wordcount of article, used with .apply() method
#l: type list
#return: type int, in range [0, inf)

def get_article_length(l):
	return len(l)



#extract text from html, remove stopwords, lemmatize, apply .lower() to title, body
#df: type DataFrame, merged dataset
#return: type DataFrame

def extract_text(df):

	df['Body'] = df['XML'].apply(clean_text)

	df['Body_sw']= df['Body'].apply(remove_stopwords_english, args = (nltk.corpus.stopwords.words('english'),))
	df['Body_Lemm'] = df['Body_sw'].apply(get_lemmatized_text).apply(adjust_keywords)
	df['Body_Lemm_lower'] = df['Body_Lemm'].str.lower()
	df['Body_Token'] = df['Body_Lemm_lower'].apply(tokenize_article)
	df['Body_Length'] = pd.to_numeric(df['Body_Token'].apply(get_article_length))


	df['Title'] = df['INDEXMASTERIDENTIFIERS'].apply(clean_title)

	df['Title_sw'] = df['Title'].apply(remove_stopwords_english, args = (nltk.corpus.stopwords.words('english'),))
	df['Title_Lemm'] = df['Title_sw'].apply(get_lemmatized_text).apply(adjust_keywords)
	df['Title_Lemm_lower'] = df['Title_Lemm'].str.lower()
	df['Title_Token'] = df['Title_Lemm_lower'].apply(tokenize_article)
	df['Title_Length'] = pd.to_numeric(df['Title_Token'].apply(get_article_length))

	df = df.drop(['Body_sw', 'Title_sw', 'INDEXMASTERIDENTIFIERS', 'XML'], axis = 1)

	return df
 


#returns dictionary of 4 sentiment scores, used with .apply() method
#document: type string
#return: type dictionary

def get_sentiment_score_dict(document):
	analyser = vader.SentimentIntensityAnalyzer()
	return analyser.polarity_scores(document)



#returns particular sentiment score, used with .apply() method
#d: type dictionary
#s: type string
#return: float, in range [-1, 1]

def get_sentiment_score(d, s):
	return d[s]



#Assigns compound, positive, neutral, negative sentiment scores to Title and Body
#df: type DataFrame, merged dataset
#return: type DataFrame

def sentiment_analysis(df):

	#sentiment analysis
	df['Title_Sent'] = df['Title_Lemm'].apply(get_sentiment_score_dict)
	df['Title_Com_Sent'] = df['Title_Sent'].apply(get_sentiment_score, args = ('compound',))
	df['Title_Pos_Sent'] = df['Title_Sent'].apply(get_sentiment_score, args = ('pos',))
	df['Title_Neu_Sent'] = df['Title_Sent'].apply(get_sentiment_score, args = ('neu',))
	df['Title_Neg_Sent'] = df['Title_Sent'].apply(get_sentiment_score, args = ('neg',))

	df['Body_Sent'] = df['Body_Lemm'].apply(get_sentiment_score_dict)
	df['Body_Com_Sent'] = df['Body_Sent'].apply(get_sentiment_score, args = ('compound',))
	df['Body_Pos_Sent'] = df['Body_Sent'].apply(get_sentiment_score, args = ('pos',))
	df['Body_Neu_Sent'] = df['Body_Sent'].apply(get_sentiment_score, args = ('neu',))
	df['Body_Neg_Sent'] = df['Body_Sent'].apply(get_sentiment_score, args = ('neg',))

	#drop unnecessary columns
	df = df.drop(['Title_Sent', 'Body_Sent'], axis=1)

	return df



#returns dictionary of counts of different parts of speech in article body, used with .apply() method
#s: type string
#return: type dict

def pos_count_to_dict(s):
	counts = Counter(tag for word, tag in nltk.pos_tag(nltk.Text(nltk.word_tokenize(s.lower()))))
	return dict((word, count) for word, count in counts.items())
	


#for a given set of POS tags, return the sum of their counts from the pos dictionary
#d: type dictionary
#pos_list: type list
#return: int, in range [0, inf)

def pos_count(d, pos_list):
	count = 0
	
	for i in pos_list:
		try:
			count = count + d[i]
		except KeyError:
			continue

	return count



#calculates proportion of POS tags in article body
#df: type DataFrame
#return: type DataFrame

def pos_proportions(df):
	
	noun = ['NN', 'NNS', 'RPR', 'WP'] #nouns, plural nouns, pronouns, wh-pronouns
	prop_noun = ['NNP', 'NNPS'] #proper nouns, plural proper nouns
	possessive = ['POS', 'PRP$', 'WP$'] #possessive nouns, pronouns, wh-pronouns
	verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] #all verbs except modal
	adverb = ['RB', 'RBR', 'RBS'] #all adverbs
	adjective = ['JJ', 'JJR', 'JJS'] #all adjectives
	conjunction = ['CC', 'IN'] #conjunctions, subordinating conjunctions, prepositions
	determiner = ['DT', 'PDT', 'WDT'] #determiners, predeterminers, wh-determiners
	expression = ['EX', 'RP', 'SYM', 'TO', 'UH'] #existence words, particles, symbols, to, interjections
	text_list = ['LS'] #any list notation. ex: 1)... 2)...
	modal = ['MD'] #would, could, should, etc.
	number = ['CD'] #cardinal number
	
	pos_tag_lists = [noun, prop_noun, possessive, verb, adjective, adverb, 
	 conjunction, determiner, expression, text_list, modal, number]

	pos_column_names = ['noun', 'prop_noun', 'possessive', 'verb', 'adjective', 'adverb', 
	'conjunction', 'determiner', 'expression', 'text_list', 'modal', 'number']
	

	#get POS dictionary
	df['Body_POS_dict'] = df['Body_Lemm_lower'].apply(pos_count_to_dict)
	
	#extract count for each POS from POS dictionary
	for i in range(len(pos_tag_lists)):
		df[pos_column_names[i]] = df['Body_POS_dict'].apply(pos_count, args = (pos_tag_lists[i],))
	
	#Normalize by sentence length
	df['Total_Hold'] = df.iloc[:, -len(pos_tag_lists):].sum(axis=1)
	
	for i in pos_column_names:
		df[i] = df[i]/df['Total_Hold']
	
	#drop unnecessary columns
	df = df.drop(['Body_POS_dict', 'Total_Hold'], axis=1)
	
	return df



#call to run script: extracts text from HTML, conducts sentiment analysis, calculates POS proportions
#df: type DataFrame
#return: type DataFrame

def text_analysis(df): #call to run script

	print('Extracting Text...')
	df = extract_text(df) #extract text from html, remove stopwords, lemmatize, etc.
	print('Text Extraction Complete')
	print()

	print('Conducting Sentiment Analysis...')
	df = sentiment_analysis(df) #Get sentiment scores of articles with POS tagging
	print('Sentiment Analysis Complete')
	print()

	print('Calculating POS Proportions...')
	df = pos_proportions(df) #Get proportions of parts of speech in article
	print('POS Tagging Complete')
	print()

	return df

