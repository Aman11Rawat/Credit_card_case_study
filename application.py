import pickle
from flask import Flask ,render_template,request
import pandas as pd

model=pickle.load(open("Rf_model_ex.pkl","rb"))

app=Flask(__name__)

@app.route("/",methods=["POST","GET"])
def index():
    if request.method=="GET":
      return render_template("index.html")
    else:
        form_input=pd.DataFrame(request.form.to_dict(),index=[0])
        prediction=model.predict(form_input.astype(float))
        return str(prediction)
    
    
if __name__=="__main__":
     app.run(debug=True)