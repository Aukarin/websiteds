from flask import Flask,render_template,request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

import os, os.path
from wtforms.validators import InputRequired
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
from gensim.models.tfidfmodel import TfidfModel
import itertools
import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        
        
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        
    return render_template('index.html', form=form)


@app.route('/top')
def top():
    articles = []
    BOW=[]
    IF=[]
    DIR = './Wrbsite_As/static/files'
    num=(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    filesname = (([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    
    for i in range(num) :

        #f = open(f"./Wrbsite_As/static/files/wiki_article_{i}.txt", "r")
        f = open(f"./Wrbsite_As/static/files/" + filesname[i] , "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)  

    dictionary = Dictionary(articles)
   
    corpus = [dictionary.doc2bow(a) for a in articles]
    doc = corpus[0]
    bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
    total_word_count = defaultdict(int)
    for word_id, word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count
    sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
    for word_id, word_count in sorted_word_count[:5]:
        B=(dictionary.get(word_id), word_count)
        BOW.append(B)
    print(BOW)

    #TF
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
    for term_id, weight in sorted_tfidf_weights[:5]:
        I=(dictionary.get(term_id), weight)
        IF.append(I)


    return render_template("top.html",BO=BOW,IFI=IF)

@app.route('/search',methods=['GET',"POST"])
def search():
    articles = []
    searchs= request.form['searchs'];
    DIR = './Wrbsite_As/static/files'
    num=(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    filesname = (([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    
    for i in range(num) :
        #f = open(f"./Wrbsite_As/static/files/wiki_article_{i}.txt", "r")
        f = open(f"./Wrbsite_As/static/files/" + filesname[i] , "r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
        articles.append(lemmatized)  
   
    logtext = []
    logsheet = []
    logcount = 0
    n=0
   
    total = len(articles)
    for x1 in range(total):
        text = len(articles[x1])
    
        for x2 in range(text):
            scan = ((articles[x1][x2]))
            if(scan == searchs ):
                logcount = logcount+1
                logtext.append(x2)
                n = 1
        
        if(n == 1):
            logsheet.append([filesname[x1],logtext])
            logtext =[]
            n = 0

    return render_template("search.html",counlogs=logcount,scansheet1=logsheet,text=searchs)
  
@app.route('/spacy',methods=['GET',"POST"])
def spacy():
    textname =[]
    texttag =[]
    artall=""
    tesxt=''
    DIR = './Wrbsite_As/static/files'
    num=(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    filesname = (([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    
    for i in range(num) :
        #f = open(f"./Wrbsite_As/static/files/wiki_article_{i}.txt", "r")
        f = open(f"./Wrbsite_As/static/files/" + filesname[i] , "r")
        article = f.read()
        artall = artall + str(article)
    
        doc = nlp("Berlin is the capital of Germany")
        doc.ents  

        
   
        doc = nlp(article)
        
        for ent in doc.ents:
            
            #ORG  DATE  PERSON  WORK_OF_ART  NORP  CARDINAL  LANGUAGE  ORDINAL  GPE  EVENT   TIME   LOC  PRODUCT  LAW   FAC
            if(ent.label_ == "ORG"):
                textname.append(ent.label_ + "a$"+ent.text) 
            if(ent.label_ == "DATE"):
                textname.append(ent.label_ + "b$"+ent.text) 
            if(ent.label_ == "PERSON"):
                textname.append(ent.label_ + "c$"+ent.text) 
            if(ent.label_ == "WORK_OF_ART"):
                textname.append(ent.label_ + "d$"+ent.text) 
            if(ent.label_ == "NORP"):
                textname.append(ent.label_ + "e$"+ent.text) 
            if(ent.label_ == "CARDINAL"):
                textname.append(ent.label_ + "f$"+ent.text) 
            if(ent.label_ == "LANGUAGE"):
                textname.append(ent.label_ + "g$"+ent.text) 
            if(ent.label_ == "ORDINAL"):
                textname.append(ent.label_ + "h$"+ent.text) 
            if(ent.label_ == "GPE"):
                textname.append(ent.label_ + "i$"+ent.text) 
            if(ent.label_ == "EVENT"):
                textname.append(ent.label_ + "j$"+ent.text) 
            if(ent.label_ == "TIME"):
                textname.append(ent.label_ + "k$"+ent.text) 
            if(ent.label_ == "LOC"):
                textname.append(ent.label_ + "l$"+ent.text) 
            if(ent.label_ == "PRODUCT"):
                textname.append(ent.label_ + "m$"+ent.text) 
            if(ent.label_ == "LAW"):
                textname.append(ent.label_ + "n$"+ent.text) 
            if(ent.label_ == "FAC"):
                textname.append(ent.label_ + "o$"+ent.text) 
            texttag.append(ent.label_) 
        docs = nlp(artall)
        cltext = displacy.render(docs, style = "ent")
        
        
    
    return render_template("spacy.html",textp=textname,text0=cltext,tetag=texttag)
    
@app.route('/fakeN',methods=['GET',"POST"])
def fakeN():
    Det ="sdstest"
    model_path = "./Wrbsite_As/fake-news-bert-base-uncased"
    
    if request.method == 'POST':
        Det =request.form['DetF'];
        
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    real_news = Det
    print(real_news)



    def get_prediction(text, convert_to_label=False):
        # prepare our text into tokenized sequence
        inputs = tokenizer(text, padding=True, truncation=True, max_length = 512 ,return_tensors="pt")
        # perform inference to our model
        outputs = model(**inputs)
     # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
        d = {
            0: "reliable",
            1: "fake"
        }
        if convert_to_label:
            return d[int(probs.argmax())]
        else:
            return int(probs.argmax())
   
    if(real_news == "sdstest"):
        reTF =""
        realre=""
    else:
        realre=real_news
        reTF = get_prediction(real_news, convert_to_label=True)
    

    return render_template("fakeN.html",reF=reTF,reall=realre)

if __name__ == "__main__":
    app.run(debug=True)