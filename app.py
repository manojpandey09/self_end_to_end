
    
#from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

#from flask import Flask,request,render_template,jsonify


#app=Flask(__name__)

#@app.route('/')
#def home_page():
    #return render_template("index.html")

#app.run()


#after writing the make a folder templates and inside folder make a file index.html  and code it 




#coiming from index.html file 
#coding again for the from.html file
#commenting above



    

    
from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":

        return render_template("form.html")
    
    else:
        data=CustomData(
            
            carat=float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color= request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

if __name__ == '__main__':
    #app.run()

#now make a file again in templates name form.html
# and do the coding there 

#make a file again results.html file inside templates 

# and run the file python app.py
# u  will see the the result but when u use / predict result is different  in locak host   





#after coming from main.yaml file
#WE are updating thee local host port to 8000 put comment on above local hostlinne no .  75 


    app.run(host="0.0.0.0",port=8000)

    #push to git hub repo



    