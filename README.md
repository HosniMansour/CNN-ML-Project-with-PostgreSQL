# CNN-ML-Project-with-postgresql

This is an update for my previous repository : https://github.com/HosniMansour/CNN-ML-Project

I used PostgreSQL to save the uploaded images with tags...

![alt text](https://github.com/HosniMansour/CNN-ML-Project-with-postgresql/blob/master/Screenshot/home.PNG?raw=true)

So to run this project you need have the previous repo files, instructions to run the previous project are here : http://blog.hosni.me/2018/01/a-cnn-machine-learning-project-using.html

Add the files of this repo and replace flaskr.py file, Config PostgreSQL in this line :

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost/mldb'

From command line go to python interpreter and use this to create the schema :

from flaskr import db
db.create_all()

Now in the command line run flaskr.py : python flaskr.py and you will be able to see the changes at 127.0.0.1:5000/admin
