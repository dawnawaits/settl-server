from flask import Flask, request, abort, jsonify
from model import ocr, predict


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['POST'])
def processRequest():
        if(request.method == 'POST'):
            if 'file' not in request.files:
                abort(406)
            file = request.files['file']
            if file and allowed_file(file.filename):
                input_text = ocr(file.filename)
                return jsonify(predict(input_text))

@app.route('/test',methods=['POST'])
def testRequest():
        if(request.method == 'POST'):
            if 'file' not in request.files:
                abort(406)
            file = request.files['file']
            if file and allowed_file(file.filename):
                return 'Hello World'


# if __name__ == '__main__':
#     app.run()