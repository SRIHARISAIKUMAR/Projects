import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,request
import mysql.connector
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3500)


con=mysql.connector.connect(user="root",password="doddisriharisaikumar",database="spam")
cur=con.cursor()
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")



@app.route("/train")
def train():
    global X_train, X_test, y_train, y_test,model,model1
    messages=pd.read_csv("C:/A/DL project/SMSSpamCollection",sep="\t",names=['label','message'])
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer() 
    corpus = []
    for i in range(0, len(messages)):
        wn.ensure_loaded()
        review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()
        review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    X = cv.fit_transform(corpus).toarray()
    y = pd.get_dummies(messages['label'])
    y = y.iloc[:, 1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(X_train,y_train)

    filename = 'spam_sms_rfc_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    from sklearn import svm
    model1=svm.SVC()
    model1.fit(X_train,y_train)
    filename = 'spam_sms_svm_model.pkl'
    pickle.dump(model1, open(filename, 'wb'))
    return render_template("train.html")

@app.route("/accuracy")
def acc():
    y_pred=model.predict(X_test)
    xyz=model1.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print("Random Forest Classifier accuracy score:{0:0.4f}".format(accuracy_score(y_test,y_pred)))
    ac=accuracy_score(y_test,xyz)
    print("Support Vector Machine accuracy score:{0:0.4f}".format(accuracy_score(y_test,xyz)))
    return render_template("accuracy.html",ab=acc,bc=ac)

@app.route("/recommend")
def recommend():
    return render_template("predict.html")

@app.route("/predictspam",methods=['POST'])
def predictspam():
    v1=request.form['t1']
    data=[v1]
    vect=cv.transform(data).toarray()
    filename='spam_sms_rfc_model.pkl'
    model=pickle.load(open(filename,'rb'))
    svmfile = 'spam_sms_svm_model.pkl'
    model1=pickle.load(open(svmfile,'rb'))
    a=model.predict(vect)
    if a==0:
        a="ham"
    else:
        a="spam"
    print(a)
    b=model1.predict(vect)
    if b==0:
        b="ham"
    else:
        b="spam"
    print(b)

    return render_template("predictspam.html",rc=a,rd=b)

@app.route("/regDB",methods=['POST'])
def regDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    s="insert into sms(name,password) values('"+uname+"','"+pwd+"')"
    cur.execute(s)
    con.commit()
    return render_template("login.html")

@app.route("/loginDB",methods=['POST'])
def loginDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    s="select * from sms where name='"+uname+"' and password='"+pwd+"'"      
    cur.execute(s)
    data=cur.fetchall()
    if len(data)>0:
        return render_template("home.html")
    else:
        return render_template("login.html",msg="Please Check the Credentials")

@app.route("/logout")
def logout():
    return render_template("index.html")

app.run(debug=True)

