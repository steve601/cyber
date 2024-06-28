from flask import Flask,request,render_template
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import re

app = Flask(__name__)

model = load_model('/var/app/current/models/twittModel.keras')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

def preprocess_text(sentence):
    tag_pattern = re.compile(r'<.*?>')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Lowercasing
    sent = sentence.lower()
    # Removal of HTML Tags
    sent = re.sub(tag_pattern, '', sent)
    # Removing Punctuation & Special Characters
    sent = re.sub('[^a-zA-Z]',' ',sent)
    # removing single character
    sent = re.sub(r"\s+[a-zA-Z]\s+",' ',sent)
    # removing multiple spaces
    sent = re.sub(r'\s+',' ',sent)
    # Removal of URLs
    sent = re.sub(url_pattern,'',sent)
    return sent

@app.route('/')
def homepage():
    return render_template('twitter.html')

@app.route('/classify',methods = ['POST','GET'])
def analyse_func():
    corpus = ""
    if request.method == 'POST':
        corpus = request.form['tweet']
        inp = []
        inp.append(preprocess_text(corpus))
        inp = tokenizer.texts_to_sequences(inp)
        inp = pad_sequences(inp,padding='pre',maxlen = 100)
    
        prediction = model.predict(inp)[0]
        predicted_class = np.argmax(prediction)
        predicted_class
    
        if predicted_class == 0:
            msg = 'This is probably a cyberbullying'
        if predicted_class == 1:
            msg = 'This is not a cyberbullying'
        if predicted_class == 2:
            msg = 'This is cyberbullying and is based on gender'
        if predicted_class == 3:
            msg = 'This is cyberbullying and is based on ethnicity'
        if predicted_class == 4:
            msg = 'This is cyberbullying and is based on age'
        if predicted_class == 5:
            msg = 'This is cyberbullying and is based on religion'
        
    return render_template('twitter.html',text = msg,corpus = corpus)


if __name__ == '__main__':
    app.run(host="0.0.0.0")