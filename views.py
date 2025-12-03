from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import datetime
import ipfsApi
import os
import json
from web3 import Web3, HTTPProvider
from django.core.files.storage import FileSystemStorage
import pickle
import time
import pyaes, pbkdf2, binascii, os, secrets
import base64
import numpy as np

import pandas as pd # data processing
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

api = ipfsApi.Client(host='http://127.0.0.1', port=5001)
global details, username

stop_words = set(stopwords.words('english'))
df = pd.read_csv("Dataset/Crime_report.csv")
Y = df['Label'].ravel()
vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=450)
X = vectorizer.fit_transform(df['description']).toarray()
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], 15, 10, 3))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

model = Sequential()
model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
if os.path.exists("model/model_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, batch_size = 8, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/model_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/model_weights.hdf5")
predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
precision = precision_score(y_test1, predict,average='macro') * 100
recall = recall_score(y_test1, predict,average='macro') * 100
fscore = f1_score(y_test1, predict,average='macro') * 100

def TrainML(request):
    if request.method == 'GET':
        global acc, precision, recall, fscore
        algorithms = ['Tensorflow CNN Algorithm']
        output = '<table border="1" align="center" width="100%" ><tr><th><font size="" color="black">Algorithm Name</th>'
        output += '<th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output += '<th><font size="" color="black">Recall</th><th><font size="" color="black">FScore</th></tr>'
        for i in range(len(algorithms)):
            output+='<tr><td><font size="" color="black">'+algorithms[i]+'</td>'
            output+='<td><font size="" color="black">'+str(acc)+'</td>'
            output+='<td><font size="" color="black">'+str(precision)+'</td>'
            output+='<td><font size="" color="black">'+str(recall)+'</td>'
            output+='<td><font size="" color="black">'+str(fscore)+'</td></tr>'
        output+="</table><br/><br/><br/><br/><br/><br/>"
        context= {'data':output}
        return render(request, 'AuthorityScreen.html', context)    

def readDetails(contract_type):
    global details
    details = ""
    print(contract_type+"======================")
    blockchain_address = 'http://127.0.0.1:8545' #Blokchain connection IP
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'SmartContract.json' #Blockchain SmartContract calling code
    deployed_contract_address = '0x0F7dF25522A4Bb4d6C178Bc988aFD986c8DD57Bc' #hash address to access Shared Data contract
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi) #now calling contract to access data
    if contract_type == 'signup':
        details = contract.functions.getSignup().call()
    if contract_type == 'tips':
        details = contract.functions.getTips().call()    
    print(details)    

def saveDataBlockChain(currentData, contract_type):
    global details
    global contract
    details = ""
    blockchain_address = 'http://127.0.0.1:8545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'SmartContract.json' #Blockchain contract file
    deployed_contract_address = '0x0F7dF25522A4Bb4d6C178Bc988aFD986c8DD57Bc' #contract address
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)  # load contract info as JSON
        contract_abi = contract_json['abi']  # fetch contract's abi - necessary to call its functions
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    readDetails(contract_type)
    if contract_type == 'signup':
        details+=currentData
        msg = contract.functions.setSignup(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)
    if contract_type == 'tips':
        details+=currentData
        msg = contract.functions.setTips(details).transact()
        tx_receipt = web3.eth.waitForTransactionReceipt(msg)    

def getKey(): #generating AES key based on Diffie common secret shared key
    password = "s3cr3t*c0d3"
    passwordSalt = str("0986543")#get AES key using diffie
    key = pbkdf2.PBKDF2(password, passwordSalt).read(32)
    return key

def encrypt(plaintext): #AES data encryption
    aes = pyaes.AESModeOfOperationCTR(getKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    ciphertext = aes.encrypt(plaintext)
    return ciphertext

def decrypt(enc): #AES data decryption
    aes = pyaes.AESModeOfOperationCTR(getKey(), pyaes.Counter(31129547035000047302952433967654195398124239844566322884172163637846056248223))
    decrypted = aes.decrypt(enc)
    return decrypted

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def AuthorityLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == "admin" and password == "admin":
            context= {'data': "Welcome "+username}
            return render(request, 'AuthorityScreen.html', context)
        else:
            context= {'data': "Invalid Login"}
            return render(request, 'AuthorityLogin.html', context)
            

def AuthorityLogin(request):
    if request.method == 'GET':
       global username       
       return render(request, 'AuthorityLogin.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        readDetails('signup')
        arr = details.split("\n")
        status = "none"
        username = encrypt(username.encode())#call AES encrypt function to encrypt given user details
        username = base64.b64encode(username).decode()#convert AES binary data to string
        username1 = request.POST.get('t1', False)
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[1] == username and password == array[2]:
                status = "Welcome "+username1
                break
        if status != 'none':
            context= {'data':status}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)

def SubmitTip(request):
    if request.method == 'GET':
       global username       
       return render(request, 'SubmitTip.html', {})

def predictActivity(desc):
    global model, vectorizer
    model = load_model("model/model_weights.hdf5")
    temp = []
    temp.append(desc)
    test = vectorizer.transform(temp).toarray()
    print(test.shape)
    test = np.reshape(test, (test.shape[0], 15, 10, 3))
    predict = model.predict(test)
    predict = np.argmax(predict)
    output = "True Report Predicted"
    if predict == 1:
        output = "False Report Predicted"
    return output                      
        
def SubmitTipAction(request):
    if request.method == 'POST':
        global username
        address = request.POST.get('t1', False)
        activity = request.POST.get('t2', False)
        desc = request.POST.get('t3', False)
        filename = request.FILES['t4'].name
        myfile = request.FILES['t4'].read()
        myfiles = pickle.dumps(myfile)
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        user = username
        predict = predictActivity(desc)
        hashcode = api.add_pyobj(myfiles)
        data = user+"#"+address+"#"+activity+"#"+desc+"#"+str(hashcode)+"#"+str(current_time)+"#"+filename+"#"+predict+"\n"
        if os.path.exists("CrimeApp/static/files/"+filename):
            os.remove("CrimeApp/static/files/"+filename)
        with open("CrimeApp/static/files/"+filename, "wb") as file:
            file.write(myfile)
        file.close() 
        saveDataBlockChain(data,"tips")
        output = 'Suspicious Activity Saved in Blockchain using unique hashcode : '+str(hashcode)
        context= {'data':output}
        return render(request, 'SubmitTip.html', context)

def ViewTip(request):
    if request.method == 'GET':
        global username
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Suspicious Activity Address</th>'
        strdata+='<th><font size="" color="black">Type of Activity</th><th><font size="" color="black">Tip Description</th>'
        strdata+='<th><font size="" color="black">Hashcode</th>'
        strdata+='<th><font size="" color="black">Tip Date</th><th><font size="" color="black">Predicted Report Status</th><th><font size="" color="black">Tip Image</th></tr>'
        readDetails('tips')
        arr = details.split("\n")
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[0] == username:
                '''
                if os.path.exists("CrimeApp/static/files/"+array[6]):
                    os.remove("CrimeApp/static/files/"+array[6])
                content = api.get_pyobj(array[4])
                with open("CrimeApp/static/files/"+array[6], "wb") as file:
                    file.write(content)
                file.close()
                '''
                print(array[6])
                strdata+='<tr><td><font size="" color="black">'+str(array[0])+'</td><td><font size="" color="black">'+array[1]+'</td><td><font size="" color="black">'+str(array[2])+'</td>'
                strdata+='<td><font size="" color="black">'+str(array[3])+'</td>'
                strdata+='<td><font size="" color="black">'+str(array[4])+'</td>'
                strdata+='<td><font size="" color="black">'+str(array[5])+'</td>'
                strdata+='<td><font size="" color="black">'+str(array[7])+'</td>'
                strdata+='<td><img src="/static/files/'+array[6]+'" height="200" width="200"></img></td></tr>'
        context= {'data':strdata}
        return render(request, 'UserScreen.html', context)

def ViewReports(request):
    if request.method == 'GET':
        global username
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Suspicious Activity Address</th>'
        strdata+='<th><font size="" color="black">Type of Activity</th><th><font size="" color="black">Tip Description</th>'
        strdata+='<th><font size="" color="black">Hashcode</th>'
        strdata+='<th><font size="" color="black">Tip Date</th><th><font size="" color="black">Predicted Report Status</th><th><font size="" color="black">Tip Image</th></tr>'
        readDetails('tips')
        arr = details.split("\n")
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            '''
            if os.path.exists("CrimeApp/static/files/"+array[6]):
                os.remove("CrimeApp/static/files/"+array[6])
            content = api.get_pyobj(array[4])
            with open("CrimeApp/static/files/"+array[6], "wb") as file:
                file.write(content)
            file.close()
            '''
            strdata+='<tr><td><font size="" color="black">'+str(array[0])+'</td><td><font size="" color="black">'+array[1]+'</td><td><font size="" color="black">'+str(array[2])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[3])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[4])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[5])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[7])+'</td>'
            strdata+='<td><img src="/static/files/'+array[6]+'" height="200" width="200"></img></td></tr>'
        context= {'data':strdata}
        return render(request, 'AuthorityScreen.html', context)

def ViewUsers(request):
    if request.method == 'GET':
        global username
        strdata = '<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Password</th>'
        strdata+='<th><font size="" color="black">Contact No</th><th><font size="" color="black">Gender</th>'
        strdata+='<th><font size="" color="black">Email ID</th>'
        strdata+='<th><font size="" color="black">Address</th></tr>'
        readDetails('signup')
        arr = details.split("\n")
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            strdata+='<tr><td><font size="" color="black">'+str(array[1])+'</td><td><font size="" color="black">'+array[2]+'</td><td><font size="" color="black">'+str(array[3])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[4])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[5])+'</td>'
            strdata+='<td><font size="" color="black">'+str(array[6])+'</td></tr>'
        context= {'data':strdata}
        return render(request, 'AuthorityScreen.html', context)    

def RegisterAction(request):
    if request.method == 'POST':
        global details
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        username = encrypt(username.encode())#call AES encrypt function to encrypt given user details
        username = base64.b64encode(username).decode()#convert AES binary data to string
        output = "Username already exists"
        readDetails('signup')
        arr = details.split("\n")
        status = "none"
        for i in range(len(arr)-1):
            array = arr[i].split("#")
            if array[1] == username:
                status = username+" already exists"
                break
        if status == "none":
            details = ""
            data = "signup#"+username+"#"+password+"#"+contact+"#"+gender+"#"+email+"#"+address+"\n"
            saveDataBlockChain(data,"signup")
            context = {"data":"Signup process completed and record saved in Blockchain"}
            return render(request, 'Register.html', context)
        else:
            context = {"data":status}
            return render(request, 'Register.html', context)














        


