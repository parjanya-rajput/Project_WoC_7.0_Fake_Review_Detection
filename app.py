import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  transformed = []

  for word in text:
    if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
      transformed.append(ps.stem(word))


  return " ".join(transformed)

tfidf = pickle.load(open('checkpoint-2/vectorizer.pkl', 'rb'))
model = pickle.load(open('checkpoint-2/modelMNB.pkl', 'rb'))

st.title('Fake Review Detection System')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):

    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Vectorise the input message
    vectorised_sms = tfidf.transform([transformed_sms])

    # Make predictions
    result = model.predict(vectorised_sms)[0]

    # Display the predictions
    if(result == 1):
        st.header('Original Review')
    else:
        st.header('Fake Review')