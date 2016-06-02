import numpy as np
import json
import nltk.stem
from sklearn.feature_extraction.text import *
from sklearn import preprocessing
from sklearn.pipeline import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn import metrics

english_stemmer = nltk.stem.SnowballStemmer('english')

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

vectorizer = StemmedCountVectorizer(
                            min_df=3,
							max_df=0.95,
                            stop_words='english',
							ngram_range = (1,2))

pipe = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
])

X_new=["have XXXX credit cards with Capital One.  had some financial difficulties and got a couple of months behind on all XXXX cards.  called Capital One to make arrangements on XXXX of the cards to pay {$60.00} over XXXX months to get caught up and gave them my bank account info to take the money out automatically. XXXX the other XXXX cards,  was able to pay them off in full.  called in to Capital One and made the payments over the phone. A few days later,  went online to check that the accounts reflected the new balance and found that each XXXX had a note that say the accounts had been suspended.  called in to Capital One to ask what this meant and they said that  could no longer use the accounts but they would remain open and  would still get a yearly charge for them but  could not use them again. They did not inform me when they took my money, nearly {$1000.00}, that  would not be able to use the cards anymore. Also, when  checked my credit, they reported me for being 120 days late which  was not, and never updated my balances to zero. They are still reporting the old balances."]
vectorizer = pipe.fit(question, Y)
z = vectorizer.predict(X_new)
print(z)

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

# Print and plot the confusion matrix
#cm = metrics.confusion_matrix(vectors_test_data, vectors_test_expected_result)
#print(cm)


