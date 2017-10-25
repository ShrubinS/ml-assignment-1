from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


class Classification:
    def __init__(self,dataset,name):
        self.dataset=dataset
        self.name=name
        self.scoring = ['accuracy', 'f1_macro',  'f1_weighted', 'precision_macro', 'recall_macro']

    def dataPreProcessing(self):
        if (self.name == "skinnonskin"):
            self.dataset = self.dataset.sample(frac=0.5)
            self.dataset = self.dataset.head(12000)  # extracting top 12k rows
            X = self.dataset.drop("Class", axis=1)
            Y = self.dataset.Class
        if (self.name == 'sum'):
            self.dataset = self.dataset.head(1000)
            target = ['Target', 'Target Class', 'Target Class Encoded', 'Target_Class_Codes']
            X = self.dataset.drop(target, axis=1)
            Y = self.dataset.Target_Class_Codes
        return X, Y

    def logisticRegression(self):
        print "\n\nLogistic Regression"
        X, Y= self.dataPreProcessing()
        kfold=model_selection.KFold(n_splits=10, random_state=5)
        lr_model=LogisticRegression()
        scores=model_selection.cross_validate(lr_model,X,Y,cv=kfold,scoring=self.scoring)
        for score in self.scoring:
            print score, scores['test_'+score].mean(), '\n'


    def decisionTree(self):
        print "\n\nDecision Tree Classifier"
        X, Y = self.dataPreProcessing()
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        dtc_model = DecisionTreeClassifier()
        scores = model_selection.cross_validate(dtc_model, X, Y, cv=kfold, scoring=self.scoring)
        for score in self.scoring:
            print score, scores['test_' + score].mean(), '\n'

    # def supportVectorClassfier(self):
    #     print "\n\nSupport Vector Classifier"
    #     if (self.name == "skinnonskin"):
    #         self.dataset = self.dataset.sample(frac=0.5)
    #         self.dataset = self.dataset.head(12000)  # extracting top 12k rows
    #     X = self.dataset.drop("Class", axis=1)
    #     Y = self.dataset.Class
    #     kfold = model_selection.KFold(n_splits=10, random_state=5)
    #     svc_model = SVC(probability=True)
    #     for metric in self.scoring:
    #         score = model_selection.cross_val_score(svc_model, X, Y, cv=kfold, scoring=metric)
    #         print "metric: " + metric
    #         print "score :" + str(score)+"\n"

    def kNeighborClassifier(self):
        print "\n\nKNN Classifier"
        X, Y = self.dataPreProcessing()
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        knc_model = KNeighborsClassifier(n_neighbors=6)
        scores = model_selection.cross_validate(knc_model, X, Y, cv=kfold, scoring=self.scoring)
        for score in self.scoring:
            print score, scores['test_' + score].mean(), '\n'

    # def kMeansClassifier(self):
    #     print "\n\nKMeans Classifier"
    #     if (self.name == "skinnonskin"):
    #         self.dataset = self.dataset.sample(frac=0.5)
    #         self.dataset = self.dataset.head(12000)  # extracting top 12k rows
    #     X = self.dataset.drop("Class", axis=1)
    #     Y = self.dataset.Class
    #     kfold = model_selection.KFold(n_splits=10, random_state=5)
    #     kmc_model = KMeans(n_clusters=3)
    #     for metric in self.scoring:
    #         score = model_selection.cross_val_score(kmc_model, X, Y, cv=kfold, scoring=metric)
    #         print "metric: " + metric
    #         print "score :" + str(score)+"\n"

    def rFClassifier(self):
        print "\n\nRandom Forests Classifier"
        X, Y = self.dataPreProcessing()
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        rfc_model = RandomForestClassifier()
        scores = model_selection.cross_validate(rfc_model, X, Y, cv=kfold, scoring=self.scoring)
        for score in self.scoring:
            print score, scores['test_' + score].mean(), '\n'


