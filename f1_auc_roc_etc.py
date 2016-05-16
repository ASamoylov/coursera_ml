# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:17:59 2016

@author: Delirium
"""

import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")

class1 = data[ data["true"] == 1 ]
class0 = data[ data["true"] == 0 ]

tp = class1[ class1["pred"] == 1].shape[0]
fn = class1[ class1["pred"] == 0].shape[0]

tn = class0[ class0["pred"] == 0].shape[0]
fp = class0[ class0["pred"] == 1].shape[0]

print "%i %i %i %i" % (tp, fp, fn, tn)

acc =  metrics.accuracy_score(data["true"], data["pred"])
pr = metrics.precision_score(data["true"], data["pred"])
rec = metrics.recall_score(data["true"], data["pred"])
f1 = metrics.f1_score(data["true"], data["pred"])

print "%.3f %.3f %.3f %.3f" % (acc, pr, rec, f1)

data2 = pd.read_csv("scores.csv")

for i in range(1,data2.shape[1]):
    ref = data2.iloc[:, 0]
    cl =  data2.iloc[:,i]
    print "%s: %f" % (data2.columns[i], metrics.roc_auc_score(ref, cl))
    (m_pr, m_rec, m_tr) = metrics.precision_recall_curve(ref, cl)
    df = pd.DataFrame({"pr":m_pr, "rec": m_rec})
    plt.plot(m_rec, m_pr)
    plt.xlim([0.7, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    print "max = %.3f" % df["pr"][ df["rec"]>= 0.7 ].max()
    plt.legend()
    plt.show()
    print "************"
    