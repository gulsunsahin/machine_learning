import numpy as np
import json
import nltk.stem
from sklearn.feature_extraction.text import *
from sklearn import preprocessing
from sklearn.pipeline import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.linear_model import *

english_stemmer = nltk.stem.SnowballStemmer('english')

def get_custom_pipe(num_comp=0, clf=None):
	# Get non-dim-reduced vectorizer
	pipe = get_vec_pipe(num_comp=0)

	# Add a logit on non-reduced tfidf, and ensemble on reduced tfidf
	clfs = ['rf', 'sgd', 'gbc']
	pipe.steps.append(
		('union', FeatureUnion([
			('featpipe', Pipeline([
				('svd', TruncatedSVD(num_comp)),
				('svd_norm', Normalizer(copy=False))
			]))
		]))
	)

	if clf:
		pipe.steps.append(('ensemblifier', clf))

	return pipe

def get_vec_pipe(num_comp=0, reducer='svd'):

    tfv = TfidfVectorizer(
        min_df=6, max_features=None, strip_accents='unicode',
        analyzer="word", token_pattern=r'\w{1,}', ngram_range=(1, 2),
        use_idf=1, smooth_idf=1, sublinear_tf=1)

    # Vectorizer
    vec_pipe = [
        ('vec', tfv)
    ]

    # Reduce dimensions of tfidf
    if num_comp > 0:
        if reducer == 'svd':
            vec_pipe.append(('dim_red', TruncatedSVD(num_comp)))
        elif reducer == 'kbest':
            vec_pipe.append(('dim_red', SelectKBest(chi2, k=num_comp)))
        elif reducer == 'percentile':
            vec_pipe.append(('dim_red', SelectPercentile(f_classif, percentile=num_comp)))

        vec_pipe.append(('norm', Normalizer()))

    return Pipeline(vec_pipe)

class StemmedCountVectorizer(TfidfVectorizer):
	  def build_analyzer(self):
		  analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		  #print("test edilen")
		  #print(english_stemmer.stem("difficulties"))
		  q = lambda doc:(english_stemmer.stem(w) for w in analyzer(doc.replace('fargo', ' ').replace('toyota', ' ').replace('saturday', ' ').replace('XXXX', ' ').replace('XX/XX/XXXX', ' ').replace('XXXX XXXX XXXX XXXX', '')) )
		  return q
		  
from sklearn.naive_bayes import *

topic = []
question = []

with open('data.csv') as f:
		for line in f:
			data = line.split(';')
			topic.append(data[0])
			question.append(data[1].replace('290', ' ').replace('99', ' ').replace('19', ' ').replace('citibank', ' ').replace('fargo', ' ').replace('X', ' ').replace('x', ' ').replace('XXXX', ' ').replace('00', ' ').replace('12000', ' ').replace('2015', ' ').replace('15', ' '))

unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]

Y = np.array(numeric_topics)

#vectorizer = StemmedCountVectorizer(
#                            min_df=1, 
#                             stop_words='english',
#							 ngram_range = (1,2))
#analyze =  vectorizer.build_analyzer()
#test = analyze(question[0])
#for x in test:
	#print(x)

clf = MultinomialNB(alpha=.01) 

pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

#pipe = get_custom_pipe(num_comp=0, clf=clf)

X_new=["have XXXX credit cards with Capital One.  had some financial difficulties and got a couple of months behind on all XXXX cards.  called Capital One to make arrangements on XXXX of the cards to pay {$60.00} over XXXX months to get caught up and gave them my bank account info to take the money out automatically. XXXX the other XXXX cards,  was able to pay them off in full.  called in to Capital One and made the payments over the phone. A few days later,  went online to check that the accounts reflected the new balance and found that each XXXX had a note that say the accounts had been suspended.  called in to Capital One to ask what this meant and they said that  could no longer use the accounts but they would remain open and  would still get a yearly charge for them but  could not use them again. They did not inform me when they took my money, nearly {$1000.00}, that  would not be able to use the cards anymore. Also, when  checked my credit, they reported me for being 120 days late which  was not, and never updated my balances to zero. They are still reporting the old balances."]
vectorizer = pipe.fit(question, Y)
z = vectorizer.predict(X_new)
print(z)
#vectors = vectorizer.fit_transform(question)

#feature_names = vectorizer.get_feature_names()

#from sklearn.feature_selection import SelectPercentile, chi2

#ch2 = SelectPercentile(score_func=chi2, percentile=1)
#results = ch2.fit_transform(vectors, Y)

#X_test = ch2.transform(X_test)
#if feature_names:
# 	keep selected feature names
#   feature_names = [feature_names[i] for i
#                         in ch2.get_support(indices=True)]
#   print(feature_names)
#   print(len(feature_names))
	
#print(feature_names)

#clf = MultinomialNB(alpha=.01) 
#pipe.fit(vectors, Y)

vectors_test_data=[]
vectors_test_expected_result=[]
with open('test_data.txt') as f:
    for line in f:
        data = line.split(';')
        vectors_test_expected_result.append(data[0])
        vectors_test_data.append(data[1])
        

numeric_result_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in vectors_test_expected_result]
numeric_result_topics = [float(i) for i in numeric_result_topics]

pred = pipe.predict(vectors_test_data)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != numeric_result_topics[index]):
		numberFalse=numberFalse+1;
	
print("200 veriden yanlis hesaplanan sayisi:")
print(numberFalse)

#vectors_test_data1=["have XXXX credit cards with Capital One.  had some financial difficulties and got a couple of months behind on all XXXX cards.  called Capital One to make arrangements on XXXX of the cards to pay {$60.00} over XXXX months to get caught up and gave them my bank account info to take the money out automatically. XXXX the other XXXX cards,  was able to pay them off in full.  called in to Capital One and made the payments over the phone. A few days later,  went online to check that the accounts reflected the new balance and found that each XXXX had a note that say the accounts had been suspended.  called in to Capital One to ask what this meant and they said that  could no longer use the accounts but they would remain open and  would still get a yearly charge for them but  could not use them again. They did not inform me when they took my money, nearly {$1000.00}, that  would not be able to use the cards anymore. Also, when  checked my credit, they reported me for being 120 days late which  was not, and never updated my balances to zero. They are still reporting the old balances."]
#vectors_test1 = vectorizer.transform(vectors_test_data1)
#pred1 = clf.predict(vectors_test1)

#print(pred1)
