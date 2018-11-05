from flask import Flask
from Ass3_models import Model
from flask import request
import pickle
mdl = Model()

app = Flask(__name__)

pickle_out = open("Model_serialization.pickle", "ab")

@app.route("/train",methods=["GET"])
def train():
    model = request.args.get('Model')
    mdl.train(model)
    return '<form action="http://127.0.0.1:9090/evaluate_repo" method="get">' \
           '<h2>The Training is Done :) </h2>' \
           '<br>' \
           'Click here to get Evaluation Report ' \
           '<input type="submit" value="Evaluation Report">' \
           '</form>'


@app.route("/evaluate_repo")
def evaluate_repo():
    repo = mdl.evaluate_repo()
    repo = repo.split()
    pickle_out = open("Model_serialization.pickle", "ab")
    pickle.dump(repo, pickle_out)
    pickle_out.close()

    return \
        '<html>\
            <style>\
                th, td {\
                    padding: 15px;\
                } \
            </style>\
            <body>\
                \
                <h2>Evaluation Report</h2>\
                \
                <table>\
                  <tr>\
                    <th></th>\
                    <th>' + repo[0] + '</th>\
                    <th>' + repo[1] + '</th>\
                    <th>' + repo[2] + '</th> \
                    <th>' + repo[3] + '</th>\
                  </tr>\
                  <tr>\
                    <td>' + repo[4] + '</td>\
                    <td>' + repo[5] + '</td>\
                    <td>' + repo[6] + '</td>\
                    <td>' + repo[7] + '</td>\
                    <td>' + repo[8] + '</td>\
                  </tr>\
                  <tr>\
                    <td>' + repo[9] + '</td>\
                    <td>' + repo[10] + '</td>\
                    <td>' + repo[11] + '</td>\
                    <td>' + repo[12] + '</td>\
                    <td>' + repo[13] + '</td>\
                  </tr>\
                  <tr>\
                    <td>' + repo[14] + repo[15] + repo[16] +'</td>\
                    <td>' + repo[17] + '</td>\
                    <td>' + repo[18] + '</td>\
                    <td>' + repo[19] + '</td>\
                    <td>' + repo[20] + '</td>\
                  </tr>\
                </table>\
                \
                </br>\
                </br>\
                \
                <form action="http://127.0.0.1:9090/predict" method="get">\
                    <h2>Prediction</h2>\
                    \
                    Age : <input type="number" name="age" min="15" max="100">\
                    <br>\
                    \
                    Rating : <input type="number" name="rating" min="1" max="5">\
                    <br>\
                    \
                    Positive Feedback Count : <input type="number" name="feedback" min="0" max="100000">\
                    <br>\
                    \
                    Division Name : <input list="DN" name="DN">\
                    <datalist id="DN">\
                        <option value="General"></option>\
                        <option value="General Petite"></option>\
                        <option value="Initmates"></option>\
                    </datalist>\
                    <br>\
                    \
                    Department Name : <input list="DeptN" name="DeptN">\
                    <datalist id="DeptN">\
                        <option value="Bottoms"></option>\
                        <option value="Dresses"></option>\
                        <option value="Intimate"></option>\
                        <option value="Jackets"></option>\
                        <option value="Tops"></option>\
                        <option value="Trend"></option>\
                    </datalist>\
                    <br>\
                    \
                    Class Name : <input list="CN" name="CN">\
                    <datalist id="CN">\
                        <option value="Blouses"></option>\
                        <option value="Dresses"></option>\
                        <option value="Fine gauge"></option>\
                        <option value="Jackets"></option>\
                        <option value="Intimates"></option>\
                        <option value="Jeans"></option>\
                        <option value="Knits"></option>\
                        <option value="Layering"></option>\
                        <option value="Legwear"></option>\
                        <option value="Lounge"></option>\
                        <option value="Outerwear"></option>\
                        <option value="Pants"></option>\
                        <option value="Shorts"></option>\
                        <option value="Skirts"></option>\
                        <option value="Sleep"></option>\
                        <option value="Sweaters"></option>\
                        <option value="Swim"></option>\
                        <option value="Trend"></option>\
                    </datalist>\
                    \
                    <br>\
                    <br>\
                    <input type="submit" value="Submit">\
                </form>\
                \
            </body>\
        </html>'


@app.route("/predict",methods=["GET"])
def predict():
    age = request.args.get('age')
    rating = request.args.get('rating')
    feedback = request.args.get('feedback')
    DN = request.args.get('DN')
    DeptN = request.args.get('DeptN')
    CN = request.args.get('CN')
    #y_pred = mdl.predict([25,2,25,"General","Dresses","Dresses"])
    y_pred = mdl.predict([int(age),int(rating),int(feedback),DN,DeptN,CN])
    pickle_out = open("Model_serialization.pickle", "ab")
    pickle.dump("Recommended IND is : " + y_pred,pickle_out)
    pickle_out.close()

    return ' Recommended IND is : ' + y_pred #+'' \
           # '<br>' \
           # '<br>' \
           # '<br>'\
           # '<form action="http://127.0.0.1:9090/pickle" method="get">' \
           # 'Click here to get serialized data <input type="submit" value="Submit">'\
           # '</form>'


# @app.route("/evaluate")
# def evaluate():
#     score = mdl.evaluate()
#     return str(score)




# @app.route("/pickle",methods=["GET"])
# def pickleData():
#     pickle_in = open("Model_serialization.pickle", "rb+")
#     saved = pickle.load(pickle_in)
#     pickle_in.close()
#     return str(saved)

if __name__ == '__main__':
    try:
        app.run(port='9090',host='0.0.0.0')
    except Exception as e:
        print("Error")