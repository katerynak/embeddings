import pandas as pd
import os
import matplotlib.pyplot as plt


class Statistics:
    """
    given working directory class loads all the data from different evaluation files,
    saved in format: variable    value for each line

    different functions visualize variables: their mean and standard deviation
    """
    def __init__(self, workingDir, **kwargs):
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

        for k, v in kwargs.items():
            self.data = self.data.loc[self.data[k] == v]

    def get_vars(self):
        """
        returns a list of variables found in dataset
        """
        return self.data.columns.tolist()

    def get_var_data(self, varx, vary, **kwargs):
        """
        returns vector of mean values and vector of variances
        :return:
        """
        outData = self.data

        """
        filtering results satisfying query
        """
        # python2
        # for k, v in kwargs.iteritems():

        # python3
        for k, v in kwargs.items():
            outData = outData.loc[outData[k] == v]
        #print(outData)

        outData = outData[[varx, vary]].astype('float')
        y = outData.groupby(varx)[vary].mean()
        x = y.index.tolist()
        std = outData.groupby(varx)[vary].std()

        return x, y, std

    def get_max_var(self, var, **kwargs):
        return self.data.var.max()

    def plot(self,  varx, vary, outFile, title="", **kwargs):
        """
        plots varx and vary with mean and standard deviation of vary wrt varx
        varx, vary, outFile are strings
        """

        outData = self.data

        self.fig = plt.figure
        """
        filtering results satisfying query
        """
        #python2
        #for k, v in kwargs.iteritems():

        #python3
        for k, v in kwargs.items():
            outData = outData.loc[outData[k] == v]
        print(outData)

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
    results_dir = "../plots_track_102311/" + y_var + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig = plt.figure()

    if (y_var=='exec_time'):

        st_lmds = Statistics("../results_track_102311/lmds/")
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_20.pdf', 'lmds, embedding_size=20, policy random',
                     embedding_size=20)
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_30.pdf', 'lmds, embedding_size=30, policy random',
                     embedding_size=30)
        st_lmds.plot('n_landmarks', y_var, results_dir + 'lmds_emb_size_40.pdf', 'lmds, embedding_size=40, policy random',
                     embedding_size=40)
    else:
        st_lmds = Statistics("../results_track_102311/lmds/")
        landmarks = [30, 50, 100, 150]
        for l in landmarks:
            st_lmds.plot('embedding_size', y_var, results_dir + 'lmds_n_landmarks_{0}.pdf'.format(l),
                         'lmds, n_landmarks={0}, policy random'.format(l),
                         n_landmarks=l)

    st_fastmap = Statistics("../results_track_102311/fastmap/")
    st_fastmap.plot('embedding_size', y_var, results_dir + 'fastmap.pdf', 'fastmap')

    st_resampling = Statistics("../results_track_102311/resampling/")
    st_resampling.plot('resampling_points', y_var, results_dir + 'resampling.pdf', 'resampling')

    # st_lipschitz = Statistics("../results_track_102311/lipschitz/")
    # st_lipschitz.plot('n_reference_objects', y_var, results_dir + 'lipschitz.pdf', 'lipschitz')

    st_dissimilarity = Statistics("../results_track_102311/dissimilarity/")
    st_dissimilarity.plot('n_prototypes', y_var, results_dir + 'dissimilarity.pdf', 'dissimilarity')


def plot_all(y_var="stress", resutls_dir=None,  outFile=None, log=False):

    # fig, ax = plt.subplots()
    #
    # st_lmds = Statistics("{}lmds/".format(resutls_dir),  n_landmarks=150)
    # #st_lmds = Statistics("{}lmds_60/".format(resutls_dir))
    # st_fastmap = Statistics("{}fastmap/".format(resutls_dir))
    # st_lipschitz = Statistics("{}lipschitz/".format(resutls_dir))
    # st_dissimilarity = Statistics("{}dissimilarity/".format(resutls_dir))
    # st_resampling = Statistics("{}resampling/".format(resutls_dir))
    #
    # stats = [st_fastmap, st_dissimilarity, st_lmds, st_lipschitz, st_resampling]
    # labels = ['fastmap', 'dissimilarity', 'lmds', 'lipschitz', 'resampling']
    # #stats = [st_fastmap]
    # #varxs = ['embedding_size']
    # #labels = ['fastmap']
    # varxs = ['embedding_size', 'n_prototypes', 'embedding_size', 'n_reference_objects', 'resampling_points']
    # #varxs = ['k', 'k', 'k', 'k']
    # #dashes = ['solid', 'dashed', ':', '-.', ]
    # colors = ['r', 'g', 'c', 'm', 'b']
    #
    # # stats = [st_fastmap, st_dissimilarity, st_lmds]
    # # labels = ['fastmap', 'dissimilarity', 'lmds']
    # # varxs = ['embedding_size', 'n_prototypes', 'embedding_size']
    # # dashes = ['solid', 'dashed', ':']
    #
    # for stat, varx, color, label in zip(stats, varxs, colors, labels):
    #     x, y, std = stat.get_var_data(varx, y_var)
    #     if log:
    #         ax.semilogy(x, y.tolist(), 'k', color=color, label=label)
    #     else:
    #         ax.plot(x, y.tolist(), 'k', color=color, label=label)
    #     plt.fill_between(x, (y - std / 2.).tolist(), (y + std / 2.).tolist(), alpha=0.2, edgecolor=color,
    #                      facecolor=color,
    #                      linewidth=1, antialiased=True)
    #
    # ax.legend(loc='lower right')
    # plt.ylabel(y_var)
    # plt.xlabel("embedding size")
    # plt.title('')
    # plt.show()
    #
    # if outFile:
    #     fig.savefig(outFile)

    #-----------------------------------------------------------------------------------------------------------

    fig, ax = plt.subplots()

    st_lmds = Statistics("{}lmds_60/".format(resutls_dir))
    st_fastmap = Statistics("{}fastmap/".format(resutls_dir))
    st_lipschitz = Statistics("{}lipschitz/".format(resutls_dir))
    st_dissimilarity = Statistics("{}dissimilarity/".format(resutls_dir))

    stats = [st_fastmap, st_dissimilarity, st_lmds, st_lipschitz]
    labels = ['fastmap', 'dissimilarity', 'lmds', 'lipschitz']
    varxs = ['k', 'k', 'k', 'k']
    dashes = [':', 'dashed', 'solid', '-.', ]
    colors = ['r', 'g', 'c', 'm']

    for stat, varx, color, label, dash in zip(stats, varxs, colors, labels, dashes):
        x, y, std = stat.get_var_data(varx, y_var)
        if log:
            ax.semilogy(x, y.tolist(), 'k', color=color, label=label, linestyle=dash, linewidth=3, alpha=0.3)
        else:
            ax.plot(x, y.tolist(), 'k', color=color, label=label, linestyle=dash, linewidth=3, alpha=0.3)
        plt.fill_between(x, (y - std / 2.).tolist(), (y + std / 2.).tolist(), alpha=0.2, edgecolor=color,
                         facecolor=color,
                         linewidth=1, antialiased=True)

    ax.legend(loc='lower right')
    plt.ylabel(y_var)
    plt.xlabel("embedding size")
    plt.title('')
    plt.show()

    if outFile:
        fig.savefig(outFile)


if __name__=="__main__":

    #outdirs= ['../plots_track_102311/', '../plots_track_100307/']
    # outdirs= [ '../plots_track_100307/']
    # #results_dirs= ['../results_track_102311/', '../plots_track_100307/']
    # results_dirs = [ '../results_track_100307/']
    outdirs = ['../plot_integers/']
    results_dirs = ['../eval_results_integers/']
    variables = ['correlation', 'distortion', 'stress', 'exec_time']
    for results_dir, outdir in zip(results_dirs, outdirs):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # for var in variables:
        #     plot_all(var, results_dir, outdir+var+'.pdf')
        plot_all('correlation', results_dir, outdir + 'correlation' + '.pdf')
        plot_all('distortion', results_dir, outdir + 'distortion' + '.pdf', log=True)
        plot_all('stress', results_dir, outdir + 'stress' + '.pdf', log=True)
        plot_all('exec_time', results_dir, outdir + 'exec_time' + '.pdf')
