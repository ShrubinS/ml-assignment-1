import config
import sys
import pandas as pd
import classification

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    for name, location in config.DATA_SETS.items():
        if (name == "skinnonskin"):
            df = pd.read_csv(location, sep=',|\t' ,engine='python' ,names=["G","R","B","Class"])
        else:
            df = pd.read_csv(location, sep=';', engine='python' )
            df['Target Class Encoded'] = df['Target Class'].astype('category')
            df['Target_Class_Codes'] = df['Target Class Encoded'].cat.codes
        print name
        classifier=classification.Classification(df,name)
        classifier.logisticRegression()
        classifier.decisionTree()
        classifier.kNeighborClassifier()
        classifier.rFClassifier()
        # print df.head()
    print "done"


if __name__ == "__main__":
    main()
