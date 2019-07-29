import os
import scipy
from scipy.stats import pearsonr
import numpy as np
from sklearn import metrics

preds=[]
labels=[]
j=0
n_folds=1+ np.arange(72)
for i in n_folds:
        q = n_folds[j]
        inp = np.loadtxt(os.path.join("{}_h500_f8_filter5_lre-3_stde-1_rege-2/".format(q),'testing_predictions.txt'))
        tar = np.loadtxt(os.path.join("{}_h500_f8_filter5_lre-3_stde-1_rege-2/".format(q),'testing_labels.txt'))
	preds.append(inp)
	labels.append(tar)
	j += 1

testpredictions = np.reshape(np.array(preds),-1)
testlabels = np.reshape(np.array(labels),-1)

np.savetxt('patient_predictions.txt', testpredictions)
np.savetxt('patient_labels.txt', testlabels)

fpr, tpr, thresholds = metrics.roc_curve(testlabels, testpredictions, pos_label=1)

print ('AUC:', metrics.auc(fpr, tpr))
print ('threshold:', thresholds)

np.savetxt('patient_fpr.txt', fpr, fmt='%.6f')
np.savetxt('patient_tpr.txt', tpr, fmt='%.6f')
