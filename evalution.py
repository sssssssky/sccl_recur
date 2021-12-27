import numpy as np
from scipy.optimize import linear_sum_assignment


class evalution:
    def __init__(self, n_calss: int):
        self.n_class = n_calss

    def get_cluster_label(self, y_true: np.ndarray, y_pre: np.ndarray) -> None:
        """according to xiongyali to get the confusion matrix

        Args:
            y_ture: the ture label of samples
            y_label: the output of cluster
        
        """
        conf = np.dot(self.n_class, y_true) + y_pre
        unique, counts = np.unique(conf, return_counts=True)
        confusion_matrix = np.zeros(self.n_class**2)
        for i in range(len(unique)):
            confusion_matrix[unique[i]] = counts[i]
        confusion_matrix = confusion_matrix.reshape(self.n_class,-1)
        row_ind,col_ind=linear_sum_assignment(np.dot(-1, confusion_matrix))
        self.confusion_matirx = confusion_matrix[:, col_ind]

    def acc(self) -> float:
        """compute and return accuracy
         
        Returns:
            accuracy
        """
        tp = self.confusion_matirx.diagonal().sum()
        all_n = self.confusion_matirx.sum()
        return tp / all_n

    def presion(self,):

        return None
    def f_score(self,):

        return None

    def purity(self, y_true: np.ndarray, y_pre: np.ndarray) -> float:
        """compute the purity of result
        
        Args:
            y_ture: the ture label of samples
            y_label: the output of cluster

        Returns:
            the purity of result

        """
        y_pre_with_label = np.zeros(y_pre.shape)
        labels = np.unique(y_true)
        bins = np.concatenate([labels,[np.max(labels) + 1]], axis = 0)
        for cluster in np.unique(y_pre):
            hist, _ = np.histogram(y_true[y_pre == cluster], bins = bins)
            cluster_class = np.argmax(hist)
            y_pre_with_label[y_pre == cluster ] =cluster_class

        return   sum(y_true == y_pre_with_label) / len(y_pre)