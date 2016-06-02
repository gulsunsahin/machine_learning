# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np


topic = []
question = []


with open('data.csv') as f:
		for line in f:
			data = line.split(',')
			topic.append(data[0])
			question.append(data[1])

unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]


Y = np.array(numeric_topics)


model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(question, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
