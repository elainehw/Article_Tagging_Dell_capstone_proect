import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_tfidf(df, list_size, article=True):

    # train vectorizer using all the documents
    vectorizer = TfidfVectorizer(stop_words='english')
    if (article):
        X = vectorizer.fit_transform(df['Body_Lemm_lower'].dropna())
        # combine all the body words into a single list
        wordstr = ' '.join(df['Body_Token'])
    else:
        X = vectorizer.fit_transform(df['Title_Lemm_lower'].dropna())
        # combine all the title words into a single list
        wordstr = ' '.join(df['Title_Token'])

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


def trans_pca(df, variance):

    scaler = StandardScaler()
    # fit on training set only.
    scaler.fit(df)
    # apply transform to both the training set and the test set.
    df = scaler.transform(df)

    # variance = 0.9 indicates retaining 90% of total variance
    pca = PCA(variance)
    pca.fit(df)
    # print(pca.n_components_)
    df = pca.transform(df)

    return df