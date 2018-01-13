import os
from cnn_bi_classifier import BiClassifier
from cnn_facial_classifier import FacialClassifier
from vgg16_classifier import VGG16Classifier

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import psycopg2

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_imageattach.context import store_context

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py
app.debug = True

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

bi_classifier = BiClassifier()
facial_classifier = FacialClassifier()
vgg16_classifier = VGG16Classifier()

# ========================== DataBase ==========================

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost/mldb'
db = SQLAlchemy(app)

# ================ CatVS Dogs =====================

class DogsCats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    image = db.Column(db.LargeBinary)
    predection = db.Column(db.String(80))
    proba = db.Column(db.Float)

    def __init__(self,filename,image,predection,proba):
        self.filename = filename
        self.image = image
        self.predection = predection
        self.proba = proba

    def __repr__(self):
        return '<predection %r>' % self.predection


# ================ FacialDB =====================

class FacialDB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    image = db.Column(db.LargeBinary)
    predection = db.Column(db.String(80))
    proba = db.Column(db.Float)

    def __init__(self,filename,image,predection,proba):
        self.filename = filename
        self.image = image
        self.predection = predection
        self.proba = proba

    def __repr__(self):
        return '<predection %r>' % self.predection


# ================ VGG16 =====================

class VGGDB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    image = db.Column(db.LargeBinary)
    predection = db.Column(db.ARRAY(db.String(100)))

    def __init__(self,filename,image,predection):
        self.filename = filename
        self.image = image
        self.predection = predection

# ========================== End DataBase ==========================

@app.route('/')
def classifiers():
    return render_template('classifiers.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def store_uploaded_image(action):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for(action,
                                filename=filename))


@app.route('/cats_vs_dogs', methods=['GET', 'POST'])
def cats_vs_dogs():
    if request.method == 'POST':
        return store_uploaded_image('cats_vs_dogs_result')
    return render_template('cats_vs_dogs.html')


@app.route('/facial', methods=['GET', 'POST'])
def facial():
    if request.method == 'POST':
        return store_uploaded_image('facial_result')
    return render_template('facial.html')


@app.route('/vgg16', methods=['GET', 'POST'])
def vgg16():
    if request.method == 'POST':
        return store_uploaded_image('vgg16_result')
    return render_template('vgg16.html')


@app.route('/cats_vs_dogs_result/<filename>')
def cats_vs_dogs_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    probability_of_dog, predicted_label = bi_classifier.predict(filepath)

    # ============ ADD TO DB :v :v ============
    file = open(filepath, 'rb')
    new = DogsCats(filename,file.read(),predicted_label,float(probability_of_dog))
    db.session.add(new)
    db.session.commit()
    # ============ ADD TO DB :v :v ============

    return render_template('cats_vs_dogs_result.html', filename=filename,
                           probability_of_dog=probability_of_dog, predicted_label=predicted_label)


@app.route('/facial_result/<filename>')
def facial_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Get this from new model --> it Work now hah :D
    predicted_char, predicted_label = facial_classifier.predict(filepath)

    # ============ ADD TO DB :v :v ============
    file = open(filepath, 'rb')
    new = FacialDB(filename,file.read(),predicted_label,float(predicted_char))
    db.session.add(new)
    db.session.commit()
    # ============ ADD TO DB :v :v ============

    return render_template('facial_result.html', filename=filename,
                           predicted_char=predicted_char, predicted_label=predicted_label)

@app.route('/vgg16_result/<filename>')
def vgg16_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    top3 = vgg16_classifier.predict(filepath)

    # ============ ADD TO DB :v :v ============
    file = open(filepath, 'rb')
    ap = {top3[0][1]+ " | " + str(top3[0][2]),top3[1][1]+ " | " + str(top3[1][2]),top3[2][1] + " | " +str(top3[2][2])}
    new = VGGDB(filename,file.read(),ap)
    db.session.add(new)
    db.session.commit()
    # ============ ADD TO DB :v :v ============

    return render_template('vgg16_result.html', filename=filename,
                           top3=top3)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

# ================================== Admin Panel ========================================


@app.route('/admin')
def admin():
    dogobj = DogsCats.query.order_by(-DogsCats.id.asc()).limit(5)
    facialobj = FacialDB.query.order_by(-FacialDB.id.asc()).limit(5)
    vggobj = VGGDB.query.order_by(-VGGDB.id.asc()).limit(5)
    return render_template('admin/home.html',dogobj=dogobj,facialobj=facialobj,vggobj=vggobj)


# ================ CatVS Dogs =====================

@app.route('/admin/cats_vs_dogs/<int:id>')
def doc_cat_img(id):
    img = DogsCats.query.get_or_404(id)
    return app.response_class(img.image, mimetype='application/octet-stream')


@app.route('/admin/cats_vs_dogs')
def admincatsvsdogs():
    obj = DogsCats.query.all()
    return render_template('admin/cats_vs_dogs.html',obj=obj)


@app.route('/admin/cats_vs_dogs/remove/<int:id>')
def removecatsvsdogs(id):
    DogsCats.query.filter(DogsCats.id == id).delete()
    db.session.commit()
    return redirect(url_for("admincatsvsdogs"))


# ================ Facial =====================


@app.route('/admin/facial/<int:id>')
def facial_img(id):
    img = FacialDB.query.get_or_404(id)
    return app.response_class(img.image, mimetype='application/octet-stream')


@app.route('/admin/facial')
def adminfacial():
    obj = FacialDB.query.all()
    return render_template('admin/facial.html',obj=obj)


@app.route('/admin/facial/remove/<int:id>')
def removefacial(id):
    FacialDB.query.filter(FacialDB.id == id).delete()
    db.session.commit()
    return redirect(url_for("adminfacial"))

# ================ VGG 16 =====================


@app.route('/admin/vgg16/<int:id>')
def vgg_img(id):
    img = VGGDB.query.get_or_404(id)
    return app.response_class(img.image, mimetype='application/octet-stream')


@app.route('/admin/vgg16')
def adminvgg():
    obj = VGGDB.query.all()
    return render_template('admin/vgg16.html',obj=obj)

@app.route('/admin/vgg16/remove/<int:id>')
def removevgg(id):
    VGGDB.query.filter(VGGDB.id == id).delete()
    db.session.commit()
    return redirect(url_for("adminvgg"))


# ================ VGG 16 =====================

@app.route('/admin/search', methods=['GET', 'POST'])
def searchnotag():
    obj = []
    obj2 = []
    obj3 = []
    val = "TAG"
    if request.method == 'POST':
        #obj = DogsCats.query.all()
        val = request.form.get('tag')
        obj = DogsCats.query.filter(DogsCats.predection.ilike("%"+val+"%")).all()
        obj2 = FacialDB.query.filter(FacialDB.predection.ilike("%"+val+"%")).all()
        #obj3 = VGGDB.query.filter(VGGDB.predection[0].ilike("%"+val+"%")).all()
    return render_template('admin/search.html',obj=obj,obj2=obj2,obj3=obj3,tag=val)

# ================================== End Admin Panel ========================================


def main():
    #bi_classifier.run_test()
    #facial_classifier.run_test()
    #vgg16_classifier.run_test()
    #app.run(debug=False)
    app.run(debug=True)


if __name__ == '__main__':
    main()
