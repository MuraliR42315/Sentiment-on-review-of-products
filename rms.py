from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug import secure_filename
try:
	import MySQLdb
except:
	import pymysql as MySQLdb
import pandas as pd
import shutil
import os
import csv
from pathlib import Path
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import pandas as pd
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import backend as K
from keras import initializers, regularizers, constraints, optimizers, layers
from wordcloud import WordCloud
app = Flask(__name__, template_folder='templates')

stop_words=set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()

def clean_text(text):
	text=re.sub(r'[^\w\s]','',text,re.UNICODE)
	text=text.lower()
	text=[lemmatizer.lemmatize(token) for token in text.split(" ")]
	text=[lemmatizer.lemmatize(token,"v") for token in text]
	text=[word for word in text if not word in stop_words]
	text=" ".join(text)
	return text

@app.route("/model",methods=['POST','GET'])
def model():
	if request.method=='POST':
		file = request.files['csv_info']
		file.filename="text.csv"
		file.save(secure_filename(file.filename))
		x_text=pd.read_csv("text.csv",names=['review'])		
		x_text['processed_review']=x_text.review.apply(lambda x: clean_text(x))
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(x_text['processed_review'])
		list_tokenized_test = tokenizer.texts_to_sequences(x_text['processed_review'])
		maxlen=130
		x_t2= pad_sequences(list_tokenized_test, maxlen=maxlen)
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		#print("hi")
		loaded_model.load_weights("model.h5")
		print("Loaded model from disk")
		ynew=loaded_model.predict_classes(x_t2)
		print(ynew)
		df=pd.DataFrame(columns=['review','rating'])
		df['review']=x_text['review']
		rate=[]
		countp=0
		countn=0
		for i in ynew:			
			value=i[0]
			if value==0:
				rate.append("Negative")
				countn+=1
			else:
				rate.append("Positive")
				countp+=1
		rate=pd.Series(rate)
		df['rating'] = rate.T
		print(df)
#		del df['rating']
		df.to_csv("result.csv")
		f=open("result.csv",encoding="utf8")
		data=[{k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
		K.clear_session()
	return render_template('Results.html',data=data,cn=countn,cp=countp)
			

def graphgen(type):
	if(type=='mobile'):
		print("hi mobile")
		f=pd.read_csv("/home/muralidhar/Project1/csv file/cellphone_negative.csv",names=['rating','review'])
		f1=pd.read_csv("/home/muralidhar/Project1/csv file/cellphone_positive.csv",names=['rating','review'])
		f['rating']=f['rating'].map({4:1,5:1,1:0,2:0})
		f1['rating']=f1['rating'].map({4:1,5:1,1:0,2:0})
		f=f.head(5000)
		f1=f1.head(5000)
		f=pd.concat([f,f1])
		#print(f)
		f['processed_review']=f.review.apply(lambda x: clean_text(x))
		all_words=' '.join([text for text in f['processed_review']])
		wordcloud=WordCloud(width=300,height=250, random_state=21,max_font_size=110).generate(all_words)
		#print(all_words)
		plt.figure(figsize=(10,7))
		plt.imshow(wordcloud,interpolation="bilinear")
		plt.axis('off')
		name1="mobile1.jpg"
		destination="/".join(['static/',name1])
		#print(destination)	
		plt.savefig(destination)
		plt.close('all')
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(f['processed_review'])
		list_tokenized_train = tokenizer.texts_to_sequences(f['processed_review'])
		maxlen=130
		x_t= pad_sequences(list_tokenized_train, maxlen=maxlen)
		y=f['rating']
		embed_size=128
		model=Sequential()
		model.add(Embedding(max_features, embed_size))
		model.add(Bidirectional(LSTM(32, return_sequences = True)))
		model.add(GlobalMaxPool1D())
		model.add(Dense(20, activation="relu"))
		model.add(Dropout(0.05))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		batch_size = 100
		epochs = 4
		model.fit(x_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
    			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model.h5")
		K.clear_session()
		print("Saved model to disk")
	if(type=='laptop'):
		print("hi laptop")
		f=pd.read_csv("/home/muralidhar/Project1/csv file/laptop_negative.csv",names=['rating','review'])
		f1=pd.read_csv("/home/muralidhar/Project1/csv file/laptop_positive.csv",names=['rating','review'])
		f['rating']=f['rating'].map({4:1,5:1,1:0,2:0})
		f1['rating']=f1['rating'].map({4:1,5:1,1:0,2:0})
		f=f.head(5000)
		f1=f1.head(5000)
		f=pd.concat([f,f1])
		#print(f)
		f['processed_review']=f.review.apply(lambda x: clean_text(x))
		all_words=' '.join([text for text in f['processed_review']])
		wordcloud=WordCloud(width=300,height=250, random_state=21,max_font_size=110).generate(all_words)
		#print(all_words)
		plt.figure(figsize=(10,7))
		plt.imshow(wordcloud,interpolation="bilinear")
		plt.axis('off')
		name1="laptop1.jpg"
		destination="/".join(['static/',name1])
		#print(destination)	
		plt.savefig(destination)
		plt.close('all')
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(f['processed_review'])
		list_tokenized_train = tokenizer.texts_to_sequences(f['processed_review'])
		maxlen=130
		x_t= pad_sequences(list_tokenized_train, maxlen=maxlen)
		y=f['rating']
		embed_size=128
		model=Sequential()
		model.add(Embedding(max_features, embed_size))
		model.add(Bidirectional(LSTM(32, return_sequences = True)))
		model.add(GlobalMaxPool1D())
		model.add(Dense(20, activation="relu"))
		model.add(Dropout(0.05))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		batch_size = 100
		epochs = 4
		model.fit(x_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
    			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model.h5")
		K.clear_session()
		print("Saved model to disk")	
	if(type=='camera'):
		print("hi camera")
		f=pd.read_csv("/home/muralidhar/Project1/csv file/camera_negativet.csv",names=['rating','review'])
		f1=pd.read_csv("/home/muralidhar/Project1/csv file/camera_positive.csv",names=['rating','review'])
		f['rating']=f['rating'].map({4:1,5:1,1:0,2:0})
		f1['rating']=f1['rating'].map({4:1,5:1,1:0,2:0})
		f=f.head(5000)
		f1=f1.head(5000)
		f=pd.concat([f,f1])
		#print(f)
		f['processed_review']=f.review.apply(lambda x: clean_text(x))
		all_words=' '.join([text for text in f['processed_review']])
		wordcloud=WordCloud(width=300,height=250, random_state=21,max_font_size=110).generate(all_words)
		#print(all_words)
		plt.figure(figsize=(10,7))
		plt.imshow(wordcloud,interpolation="bilinear")
		plt.axis('off')
		name1="camera1.jpg"
		destination="/".join(['static/',name1])
		#print(destination)	
		plt.savefig(destination)
		plt.close('all')
		max_features=6000
		tokenizer=Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(f['processed_review'])
		list_tokenized_train = tokenizer.texts_to_sequences(f['processed_review'])
		maxlen=130
		x_t= pad_sequences(list_tokenized_train, maxlen=maxlen)
		y=f['rating']
		embed_size=128
		model=Sequential()
		model.add(Embedding(max_features, embed_size))
		model.add(Bidirectional(LSTM(32, return_sequences = True)))
		model.add(GlobalMaxPool1D())
		model.add(Dense(20, activation="relu"))
		model.add(Dropout(0.05))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		batch_size = 100
		epochs = 4
		model.fit(x_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
    			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model.h5")
		K.clear_session()
		print("Saved model to disk")	
	return name1

@app.route("/")
@app.route("/home")
def home():
    return render_template("Home.html")

@app.route("/review")
def review():
    return render_template("review.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/signin")
def signin():
    return render_template('signin.html')

@app.route("/login", methods=['POST','GET'])
def registration():
	if request.method=='POST':
		uname=request.form['usrnm']
		pwd=request.form['psw']
		print(uname)
		conn = MySQLdb.connect(host="localhost",
                           user = "root",
                           passwd = "root",
                           db = "rms")
		c = conn.cursor()
		register=("Insert into reg(uname,pwd) values(%s,%s)")
		c.execute(register,(uname,pwd))
		data=uname
		conn.commit()
		conn.close()	
		return render_template("product.html", data=data)


@app.route("/product",methods=['POST','GET'])
def product():
	if request.method=='POST':
		uname=request.form['usrnm']
		pwd=request.form['psw']
		conn = MySQLdb.connect(host="localhost",
	                           user = "root",
	                           passwd = "root",
        	                   db = "rms")
		c = conn.cursor()
		c.execute("SELECT * FROM reg WHERE uname =%s", [uname])
		data1=c.fetchone()				
		try:			
			pas=data1[1]
		except:
			return render_template('Home.html')
		if pwd.encode('utf-8')==pas.encode('utf-8'):
			app.logger.info('PassWord Matched')
			data=uname
#			data1=os.listdir('./static')
			c.close()
			return render_template("product.html", data=data)
		else:				
			return render_template("Home.html")	

@app.route("/back",methods=['POST','GET'])
def back():
	return render_template("product.html")

@app.route("/mobile",methods=['POST','GET'])
def mobile():
	f=pd.read_csv("/home/muralidhar/Project1/csv file/cellphone_negative.csv",header=None)
	f=f.head(2000)
	#print(type(f))
	f1=pd.read_csv("/home/muralidhar/Project1/csv file/cellphone_positive.csv",header=None)
	f1=f1.head(2000)
	#print(f1)	
	df2 = pd.concat([f,f1])
	#print(df2)
	df2.to_csv("temp.csv")
	f=open("temp.csv",encoding="utf8")
	#f1=open("rate.csv",encoding="utf8")
	data=[{k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
	#data1=[{k:v for k, v in row.items()} for row in csv.DictReader(f1, skipinitialspace=True)]
	return render_template("mobile.html",data=data)

@app.route("/laptop",methods=['POST','GET'])
def laptop():
	f=pd.read_csv("/home/muralidhar/Project1/csv file/laptop_negative.csv",header=None)
	#print(type(f))
	f=f.head(2000)
	f1=pd.read_csv("/home/muralidhar/Project1/csv file/laptop_positive.csv",header=None)
	#print(f1)
	f1=f1.head(2000)	
	df2 = pd.concat([f,f1])
	#print(df2)
	#df2.to_csv("rate.csv")
	df2.to_csv("temp.csv")
	f=open("temp.csv",encoding="utf8")
	#f1=open("rate.csv",encoding="utf8")
	data=[{k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
	return render_template("laptop.html",data=data)

@app.route("/camera",methods=['POST','GET'])
def camera():
	f=pd.read_csv("/home/muralidhar/Project1/csv file/camera_negativet.csv",header=None)
	#print(type(f))
	f=f.head(2000)
	f1=pd.read_csv("/home/muralidhar/Project1/csv file/camera_positive.csv",header=None)
	f1=f1.head(2000)	
	#print(f1)	
	df2 = pd.concat([f,f1])
	#print(df2)
	df2.to_csv("temp.csv")
	f=open("temp.csv",encoding="utf8")
	data=[{k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
	return render_template("camera.html",data=data)

@app.route("/graph/<type>",methods=['POST','GET'])
def graph(type):	
	data=graphgen(type)
	#plt.bar(f, performance, align='center', alpha=0.5)
	return render_template("graph.html",data=data)



if __name__ == "__main__":
    app.run(debug=True)
