import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('rand.pkl', 'rb'))
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "VrmVgKxuio5eHkPSZANAs03hewxQf9w-OJafd4S7EBnz"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    gender = float(request.args.get('gender'))
    stream = float(request.args.get('stream'))
    internship = float(request.args.get('internship'))
    cgpa = float(request.args.get('cgpa'))
    backlogs = float(request.args.get('backlogs'))
    arr = np.array([gender,stream,internship,cgpa,backlogs])
    brr = np.asarray(arr, dtype=float)
    brr_list = brr.tolist()

    


    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields":[gender,stream,internship,cgpa,backlogs], "values":[brr_list]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/867e1de5-12e2-4dc4-9ee5-991a37fa1e3d/predictions?version=2021-05-01', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    output = model.predict([brr_list])
    
    print("Final Prediction :",output)
    

    if(output==1):
        out = 'Placed!!!'
    else:
        out = 'Not Placed!!! Work hard.......'
    return render_template('out.html', output=out)
    

if __name__ == "__main__":
    app.run(debug=True)


