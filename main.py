from flask import Flask, render_template , request
app = Flask(__name__)
import pickle

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()


@app.route('/', methods = ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Age = int(myDict['Age'])
        Fever = int(myDict['Fever'])
        Body_Pain = int(myDict['Body_Pain'])
        Runny_Nose = int(myDict['Runny_Nose'])
        Diff_Breath = int(myDict['Diff_Breath'])

        input_features = [Age,Fever,Body_Pain,Runny_Nose,Diff_Breath]

        infprob = clf.predict_proba([input_features])[0][1]
        return render_template('probability.html',inf = round(infprob)*100)
    return render_template('show.html')


if __name__ == '__main__':
    app.run(debug = True)
