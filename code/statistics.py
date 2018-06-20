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

    def plot(self,  varx, vary, outFile, title="", **kwargs):
        """
        plots varx and vary with mean and standard deviation of vary wrt varx
        varx, vary, outFile are strings
        """

        outData = self.data

        """
        filtering results satisfying query
        """
        #python2
        #for k, v in kwargs.iteritems():

        #python3
        for k, v in kwargs.items():
            outData = outData.loc[outData[k] == v]
        print(outData)

        self.fig = plt.figure()
        outData = outData[[varx, vary]].astype('float')
        y = outData.groupby(varx)[vary].mean()
        x = y.index.tolist()
        std = outData.groupby(varx)[vary].std()
        if (vary!="correlation"):
            plt.semilogy(x, y.tolist(), 'k', color='#1B2ACC')
        else:
            plt.plot(x, y.tolist(), 'k', color='#1B2ACC')
        #plt.plot(x, y.tolist(), 'k', color='#1B2ACC')
        #plt.plot(x=x, y=y.tolist(), fmt='k')
        plt.fill_between(x, (y - std/2.).tolist(), (y + std/2.).tolist(), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                         linewidth=1, antialiased=True)
        plt.ylabel(vary)
        plt.xlabel(varx)
        plt.title(title)
        plt.show()
        self.fig.savefig(outFile)


def plot_var(y_var="stress"):
    results_dir = "../plots/" + y_var + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if (y_var=='exec_time'):

        st_lmds = Statistics("../eval_results/lmds/")
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_20.pdf', 'lmds, embedding_size=20, policy random',
                     embedding_size=20)
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_30.pdf', 'lmds, embedding_size=30, policy random',
                     embedding_size=30)
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_40.pdf', 'lmds, embedding_size=40, policy random',
                     embedding_size=40)
    else:
        st_lmds = Statistics("../eval_results/lmds/")
        landmarks = [30, 50, 100, 150]
        for l in landmarks:
            st_lmds.plot('embedding_size', y_var, results_dir + 'lmds_n_landmarks_{0}.pdf'.format(l),
                         'lmds, n_landmarks={0}, policy random'.format(l),
                         n_landmarks=l)

    st_fastmap = Statistics("../eval_results/fastmap/")
    st_fastmap.plot('embedding_size', y_var, results_dir + 'fastmap.pdf', 'fastmap')

    st_resampling = Statistics("../eval_results/resampling/")
    st_resampling.plot('resampling_points', y_var, results_dir + 'resampling.pdf', 'resampling')

    st_lipschitz = Statistics("../eval_results/lipschitz/")
    st_lipschitz.plot('n_reference_objects', y_var, results_dir + 'lipschitz.pdf', 'lipschitz')

    st_dissimilarity = Statistics("../eval_results/dissimilarity/")
    st_dissimilarity.plot('n_prototypes', y_var, results_dir + 'dissimilarity.pdf', 'dissimilarity')


if __name__=="__main__":

    plot_var('stress')
    plot_var('correlation')
    plot_var('distortion')
    plot_var('exec_time')

    results_dir = "../plot_integers/"


    st_integer_fastmap = Statistics("../eval_results_integers/fastmap/")
    st_integer_fastmap.plot('k', "stress", results_dir + "stress/" + 'fastmap.pdf', 'fastmap')
    st_integer_fastmap.plot('k', "correlation", results_dir + "correlation/" + 'fastmap.pdf', 'fastmap')
    st_integer_fastmap.plot('k', "distortion", results_dir + "distortion/" + 'fastmap.pdf', 'fastmap')

    st_integer_lmds = Statistics("../eval_results_integers/lmds/")
    st_integer_lmds.plot('k', "stress", results_dir + "stress/" + 'lmds.pdf', 'lmds')
    st_integer_lmds.plot('k', "correlation", results_dir + "correlation/" + 'lmds.pdf', 'lmds')
    st_integer_lmds.plot('k', "distortion", results_dir + "distortion/" + 'lmds.pdf', 'lmds')

    st_integer_dissimilarity = Statistics("../eval_results_integers/dissimilarity/")
    st_integer_dissimilarity.plot('k', "stress", results_dir + "stress/" + 'dissimilarity.pdf', 'dissimilarity')
    st_integer_dissimilarity.plot('k', "correlation", results_dir + "correlation/" + 'dissimilarity.pdf',
                                  'dissimilarity')
    st_integer_dissimilarity.plot('k', "distortion", results_dir + "distortion/" + 'dissimilarity.pdf', 'dissimilarity')

    # st.plot('embedding_size', 'distortion', './file.pdf')
    # st.plot('embedding_size', 'correlation', './file.pdf')
    # st.plot('embedding_size', 'exec_time', './file.pdf')
