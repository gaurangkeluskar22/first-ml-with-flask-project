import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    if request.method=="POST":
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        runnynose=int(mydict['runnynose'])
        breathing=int(mydict['breathing'])
        bodypain=int(mydict['bodypain'])

    '''
    print(request.form.values())
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    print(final_features)
    prediction=model.predict(final_features)
    print(prediction)
    
    return render_template('index.html',predection_text='Patients probability of infection is {} %'.format(np.round(prediction,2)))


if __name__=="__main__":
    app.run(debug=True)