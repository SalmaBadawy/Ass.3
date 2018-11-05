


####### Dataset description #######
#Clothing ID --> Unique ID of the product
#Age --> Age of the reviewer
#Title --> Title of the review
#Review Text --> review
#Rating --> Product rating by reviewer
#Recommended IND --> Whether the product is recommended or not by the reviewer
#Positive Feedback Count --> Number of positive feedback on the review
#Division Name --> Name of the division product is in
#Department Name --> Name of the department product is in
#Class Name --> Type of product



########### Data Preprocessing ############


import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

class Model :
    def __init__(self):
        self.labelDN = LabelEncoder()
        self.labelDeptN = LabelEncoder()
        self.labelCN = LabelEncoder()
        self.sc = StandardScaler()
        self.lr = LogisticRegression(random_state = 0)
        self.knn = KNeighborsClassifier(n_neighbors=10)
        self.svm = SVC(kernel='linear')
        self.nb = GaussianNB()
        self.pred = []


    def read(self , path):
        self.data = pd.read_csv(path)

    def preprocessing(self):
        self.data = self.data[["Age" , "Rating" ,"Positive Feedback Count" , "Division Name" ,"Department Name" , "Class Name" , "Recommended IND"]]

        self.data["Division Name"].fillna('General',inplace=True)
        self.data["Department Name"].fillna('Tops',inplace=True)
        self.data["Class Name"].fillna('Dresses',inplace=True)

        self.data["Division Name"] = self.labelDN.fit_transform(self.data["Division Name"].astype(str))
        self.data["Department Name"] = self.labelDeptN.fit_transform(self.data["Department Name"].astype(str))
        self.data["Class Name"] = self.labelCN.fit_transform(self.data["Class Name"].astype(str))

        self.data[["Age","Division Name","Department Name","Class Name"]]=self.sc.fit_transform(self.data[["Age","Division Name","Department Name","Class Name"]])

    def split(self):
        self.x = self.data.iloc[:,:-1].values
        self.y = self.data.iloc[:,-1].values

    def train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=0)

    def train(self , model):
        self.read("C:\\Users\\DELL\\OneDrive\\Desktop\\NTI\\Womens Clothing E-Commerce Reviews.csv")
        self.preprocessing()
        self.split()
        self.train_test()

        if model=="Logistic Regression" :
            self.lr.fit(self.x_train, self.y_train)
            self.y_pred = self.lr.predict(self.x_test)
            self.model_type = self.lr
        elif model=="K- nearest neighbor" :
            self.knn.fit(self.x_train, self.y_train)
            self.y_pred = self.knn.predict(self.x_test)
            self.model_type = self.knn
        elif model=="Naive Bayes" :
            self.nb.fit(self.x_train, self.y_train)
            self.y_pred = self.nb.predict(self.x_test)
            self.model_type = self.nb
        elif model=="SVM" :
            self.svm.fit(self.x_train, self.y_train)
            self.y_pred = self.svm.predict(self.x_test)
            self.model_type = self.svm
        else:
            print ("Choose correct model")

    def predict(self,pred):
        #pred =  pred.toarray().reshape(-1,1)
        # self.pred['age','rating','feedback','DN','DeptN','CN'] = self.pred[age,rating,feedback,DN,DeptN,CN]
        pred[3] = self.labelDN.transform([pred[3]])
        pred[4] = self.labelDeptN.transform([pred[4]])
        pred[5] = self.labelCN.transform([pred[5]])
        [[pred[0],pred[3],pred[4],pred[5]]] = self.sc.transform([[pred[0],pred[3],pred[4],pred[5]]])
        #pred =  pred.reshape(-1,1)
        result = self.model_type.predict([pred])
        return str(result)

    def evaluate(self):
        return self.model_type.score(self.x_test, self.y_test)

    def evaluate_repo(self):
        return  classification_report(self.y_test , self.y_pred)