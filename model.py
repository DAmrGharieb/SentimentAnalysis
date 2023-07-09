
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from textblob import TextBlob
from textblob import Word
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS


class TextModel:
    def __init__(self):
       pass


    def create_model(self,cv_train_reviews,tv_train_reviews,train_sentiments):
        #training the model
        mnb=MultinomialNB()
        #fitting the svm for bag of words
        mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
        #fitting the svm for tfidf features
        mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)

        return mnb_bow,mnb_tfidf,mnb

    def predict_model(self,mnb,cv_test_reviews,tv_test_reviews):
        #Predicting the model for bag of words
        mnb_bow_predict=mnb.predict(cv_test_reviews)
        #Predicting the model for tfidf features
        mnb_tfidf_predict=mnb.predict(tv_test_reviews)

        return mnb_bow_predict,mnb_tfidf_predict
    
    def measure_accuracy(self,test_sentiments,mnb_bow_predict,mnb_tfidf_predict):
        #Accuracy score for bag of words
        mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
        #Accuracy score for tfidf features
        mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)
        #Classification report for bag of words 
        mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
        #Classification report for tfidf features
        mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
        #confusion matrix for bag of words
        cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])
        #confusion matrix for tfidf features
        cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])

        return mnb_bow_report,mnb_tfidf_report,cm_bow,cm_tfidf,mnb_bow_score,mnb_tfidf_score
    

    def create_word_cloud(self,sentiment_txt):
        plt.figure(figsize=(10,10))
        text=sentiment_txt
        WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
        words=WC.generate(text)
        plt.imshow(words,interpolation='bilinear')
        plt.show