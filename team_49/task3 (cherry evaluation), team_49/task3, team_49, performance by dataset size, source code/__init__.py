import config
import sys
import pandas as pd
import classification

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    file = open(config.outputfile, 'w')
    orig_stdout = sys.stdout
    sys.stdout = file
    orig_stderr = sys.stderr
    sys.stderr = file

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
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

if __name__ == "__main__":
    main()
