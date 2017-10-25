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
        self.scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'neg_log_loss']

    def logisticRegression(self):
        print "\n\nLogistic Regression"
        if(self.name=="skinnonskin"):
            self.dataset=self.dataset.sample(frac=0.5)
            self.dataset=self.dataset.head(12000)  #extracting top 12k rows
        X=self.dataset.drop("Class",axis=1)
        Y=self.dataset.Class
        kfold=model_selection.KFold(n_splits=10, random_state=5)
        lr_model=LogisticRegression()
        for metric in self.scoring:
            score=model_selection.cross_val_score(lr_model,X,Y,cv=kfold,scoring=metric)
            print "metric: " + metric
            print "score :"+ str(score)+"\n"

    def decisionTree(self):
        print "\n\nDecision Tree Classifier"
        if (self.name == "skinnonskin"):
            self.dataset = self.dataset.sample(frac=0.5)
            self.dataset = self.dataset.head(12000)  # extracting top 12k rows
        X = self.dataset.drop("Class", axis=1)
        Y = self.dataset.Class
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        dtc_model = DecisionTreeClassifier()
        for metric in self.scoring:
            score = model_selection.cross_val_score(dtc_model, X, Y, cv=kfold, scoring=metric)
            print "metric: " + metric
            print "score :" + str(score)+"\n"

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
        if (self.name == "skinnonskin"):
            self.dataset = self.dataset.sample(frac=0.5)
            self.dataset = self.dataset.head(12000)  # extracting top 12k rows
        X = self.dataset.drop("Class", axis=1)
        Y = self.dataset.Class
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        knc_model = KNeighborsClassifier(n_neighbors=6)
        for metric in self.scoring:
            score = model_selection.cross_val_score(knc_model, X, Y, cv=kfold, scoring=metric)
            print "metric: " + metric
            print "score :" + str(score)+"\n"

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
        if (self.name == "skinnonskin"):
            self.dataset = self.dataset.sample(frac=0.5)
            self.dataset = self.dataset.head(12000)  # extracting top 12k rows
        X = self.dataset.drop("Class", axis=1)
        Y = self.dataset.Class
        kfold = model_selection.KFold(n_splits=10, random_state=5)
        rfc_model = RandomForestClassifier()
        for metric in self.scoring:
            score = model_selection.cross_val_score(rfc_model, X, Y, cv=kfold, scoring=metric)
            print "metric: " + metric
            print "score :" + str(score)+"\n"


