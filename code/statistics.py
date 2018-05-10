import pandas as pd
import os
import matplotlib.pyplot as plt

class Statistics:
    """
    given working directory class loads all the data from different evaluation files,
    saved in format: variable    value for each line

    different functions visualize variables: their mean and standard deviation
    """
    def __init__(self, workingDir):
        """
        opens workingDir and loads data from each file in pandas dataFrame,
        taking columns from the first random file
        """
        self.dir = workingDir
        f = self.dir + os.listdir(self.dir)[0]
        columns = pd.read_csv(f, sep = '\t', header=None)[0]
        self.data = pd.DataFrame(columns=columns)
        for file in os.listdir(self.dir):
            f = self.dir + file
            df = pd.read_csv(f, sep='\t', header=None)
            df1 = df[1].to_frame()
            df1 = df1.transpose()
            df1.columns = df[0]
            self.data = self.data.append(df1, ignore_index=True)

    def get_vars(self):
        """
        returns a list of variables found in dataset
        """
        return self.data.columns.tolist()

    def plot(self, varx, vary, outFile):
        """
        plots varx and vary with mean and standard deviation of vary wrt varx
        varx, vary, outFile are strings
        """
        self.fig = plt.figure()
        outData = self.data[[varx, vary]].astype('float')
        y = outData.groupby(varx)[vary].mean()
        x = y.index.tolist()
        std = outData.groupby(varx)[vary].std()
        plt.plot(x, y.tolist(), 'k', color='#1B2ACC')
        #plt.plot(x=x, y=y.tolist(), fmt='k')
        plt.fill_between(x, (y - std/2.).tolist(), (y + std/2.).tolist(), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                         linewidth=1, antialiased=True)
        plt.show()
        self.fig.savefig(outFile)



if __name__=="__main__":
    st = Statistics("../eval_results/lmds/")
    st.plot('k', 'stress', './file.pdf')
    st.plot('n_landmarks', 'stress', './file.pdf')