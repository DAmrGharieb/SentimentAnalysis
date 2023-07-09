from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from sklearn.preprocessing import LabelBinarizer

tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')


class TextProcessing:

    def __init__(self):
        pass

    def sentence_to_words(self,sentence):
        """
        Words are tokenized. To separate a statement into words, we utilise the word tokenize () method.
        """
        #Tokenization of text
        tokenizer=ToktokTokenizer()
        #Setting English stopwords
        return nltk.corpus.stopwords.words('english')


    def strip_html(self,text):
        """
        Removing the html strips

        """
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()


    def remove_between_square_brackets(self,text):
        """
        Removing the square brackets
        """
        return re.sub('\[[^]]*\]', '', text)


    def denoise_text(self,text):
        """
        Removing the noisy text

        """

        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        return text
    

    def remove_special_characters(self,text, remove_digits=True):
        '''
        Removing special characters and digits
        '''
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text


    def simple_stemmer(self,text):
        """
        Stemming the text
        """
        ps=nltk.porter.PorterStemmer()
        text= ' '.join([ps.stem(word) for word in text.split()])
        return text



    def remove_stopwords(self,text, is_lower_case=False):
        """
        Removing the stopwords
        """

        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text
    

    def create_bag_of_words(self,norm_train_reviews,norm_test_reviews):
        """
        Creating the bag of words
        """
        cv=CountVectorizer(min_df=0.0,max_df=1.0,binary=False,ngram_range=(1,3))
        #transformed train reviews
        cv_train_reviews=cv.fit_transform(norm_train_reviews)
        #transformed test reviews
        cv_test_reviews=cv.transform(norm_test_reviews)

        return cv_train_reviews,cv_test_reviews
  
    def create_tfidf(self,norm_train_reviews,norm_test_reviews):
        """
        Creating the tfidf
        """
        tv=TfidfVectorizer(min_df=0.0,max_df=1.0,use_idf=True,ngram_range=(1,3))
        #transformed train reviews
        tv_train_reviews=tv.fit_transform(norm_train_reviews)
        #transformed test reviews
        tv_test_reviews=tv.transform(norm_test_reviews)
        return tv_train_reviews,tv_test_reviews
    

    def label_sentiment(self,sentiment):
        """
        Labeling the sentiment data
        """
        lb=LabelBinarizer()
        #transformed sentiment data
        return lb.fit_transform(sentiment)


    