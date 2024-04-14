from tensorflow.keras.models import Sequential, model_from_json  
from tensorflow.keras.layers import Dense
import numpy
import tensorflow as tf
from flask import Flask,render_template,request
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

app = Flask(__name__ , static_folder='static')

@app.route("/")
def index():
   return render_template("pneumonia.html")

@app.route("/pneumonia")
def abcd():
   return render_template("pneumonia.html")
# Function to preprocess the image

@app.route("/detection", methods=['POST'])
def detection():
   img = request.files['image']
   img = Image.open(img.stream)
   img1 = img.convert('RGB')
   img1 = tf.keras.preprocessing.image.img_to_array(img1)
   img1 =  tf.image.resize(img1,(224,224))
   img1 = tf.reshape(img1,[1,224,224,3])
   json_file1 = open('C:/A/final projecct/model.json', 'r')
   
   loaded_model_json1 = json_file1.read()
   json_file1.close()
   loaded_model1 = model_from_json(loaded_model_json1)
# load weights into new model
   loaded_model1.load_weights("C:/A/final projecct/model.h5")
   prediction1 = loaded_model1.predict(img1 )
   predict_class1 = numpy.argmax(prediction1)
   if predict_class1 == 0:
      a="Normal"
   else:
      a="Pneumonia"
   
      
   json_file2 = open('C:/A/final projecct/cnnmodel.json', 'r')
   loaded_model_json2 = json_file2.read()
   json_file2.close()
   loaded_model2 = model_from_json(loaded_model_json2)
    #load weights into new model
   loaded_model2.load_weights("C:/A/final projecct/cnnmodel.h5")
   prediction2 = loaded_model2.predict(img1)
   predict_class2 = numpy.argmax(prediction2)
   if predict_class2 == 0:
      c="Normal"
   else:
      c="Pneumonia"


   return render_template("detection.html",ra=a,rc=c)

if __name__ == '__main__':
    app.run(debug=True)



