import pdb
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt
import pickle 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str, help='split')
args = parser.parse_args()
#project_dir = "/Users/shachideshpande/Downloads/PRS_Data/"
project_dir = "/share/kuleshov/ssd86/UKBB_Downloads/calibration_scripts/prs-uncertainty/"
NUM_BUCKETS=10
normalize=True
SUBSET = 1000


def ece(y_true, y_pred, n_bins=10):
    x_axis = np.arange(0, 1.1, (1.0)/n_bins)
    y_axis = np.zeros(x_axis.shape)
    N = len(y_true)
    score = 0
    for i, x in enumerate(x_axis):
        if(i==0):
            continue
        bin_outputs = y_true[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        bin_preds = y_pred[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        
            
        if(len(bin_outputs)>0):
            y_axis[i]=bin_outputs.mean()
            avg_pred = bin_preds.mean()
        else:
            y_axis[i]=0
            avg_pred = 0
        
        score+=abs(y_axis[i]-avg_pred)*((1.0*len(bin_outputs))/N)
  
    return score
def ece_curve(y_true, y_pred, n_bins=10):
    x_axis = np.arange(0, 1.1, (1.0)/n_bins)

    y_axis = np.zeros(x_axis.shape)
    x_axis_curve = []
    y_axis_curve = []
    bin_count_curve = []
    N = len(y_true)
    score = 0
    for i, x in enumerate(x_axis):
        if(i==0):
            continue
        bin_outputs = y_true[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        bin_preds = y_pred[np.logical_and(y_pred<=x, y_pred>x_axis[i-1])]
        
        bin_count_curve.append(len(bin_outputs)/N)  
        if(len(bin_outputs)>0):
            y_axis[i]=bin_outputs.mean()
            avg_pred = bin_preds.mean()
            x_axis_curve.append(avg_pred)
            y_axis_curve.append(y_axis[i])
        else:
            y_axis[i]=0
            avg_pred = 0
  
    return np.array(y_axis_curve), np.array(x_axis_curve), np.array(bin_count_curve)

def plot_calibration_curves(X_plot, y_plot, classifier, calibrated_classifier, savefile):

    y_true = y_plot
    y_pred = classifier.predict_proba(X_plot)[:,1]
    prob_true, prob_pred, bin_count_curve = ece_curve(y_true, y_pred, n_bins=10)
    plt.plot(prob_pred, prob_true, label="plain", color="orange")
    y_max = np.histogram(y_pred, range=(0,1),bins=10)[0].max()
    plt.hist(y_pred, range=(0,1), weights=1/(y_max) * np.ones(len(y_pred)) ,bins=10, alpha=0.4, color="orange")

    y_pred = calibrated_classifier.predict_proba(X_plot)[:,1]
    prob_true, prob_pred, bin_count_curve = ece_curve(y_true, y_pred, n_bins=10)
    plt.plot(prob_pred, prob_true, label="calib", color="blue")
    y_max = np.histogram(y_pred, range=(0,1),bins=10)[0].max()
    plt.hist(y_pred, range=(0,1), weights=1/(y_max) * np.ones(len(y_pred)) ,bins=10, alpha=0.4, color="blue")
    
    
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--' , color="black", label='ideal')
    plt.legend()
    plt.axis('square')

    if(savefile is not None):
      plt.savefig(savefile)
    else:
      plt.show()
    plt.clf()

def assess_classifier(X_cls, y_cls, classifier):
    tn, fp, fn, tp = confusion_matrix(y_cls, classifier.predict(X_cls), labels=[0,1]).ravel()
    positive_eps=0
    if(tp+fp==0):
        print(("tp+fp=0!"))
        positive_eps=1
    ppv=tp/(fp+tp+positive_eps)
    npv=tn/(tn+fn)
    tpr=tp/(tp+fn)
    f1_eps=0
    if(ppv+tpr==0):
        print(("ppv+tpr=0!"))
        f1_eps=1
    f1_score=2*ppv*tpr/(ppv+tpr+f1_eps)
    auc=roc_auc_score(y_cls, classifier.predict_proba(X_cls)[:, 1])
    return (ppv, npv, tpr, f1_score, auc)

final_results = {"rvalue_before":{"train":[], "test":[], "val":[], "gwas":[]},
"rvalue_after": {"train":[], "test":[], "val":[], "gwas":[]},
"cal_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"cal_score_after": {"train":[], "test":[], "val":[], "gwas":[]},
"auc_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"auc_score_after": {"train":[], "test":[], "val":[], "gwas":[]},
"f1_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"f1_score_after": {"train":[], "test":[], "val":[], "gwas":[]},
"ppv_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"ppv_score_after": {"train":[], "test":[], "val":[], "gwas":[]},
"npv_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"npv_score_after": {"train":[], "test":[], "val":[], "gwas":[]}
}
save_file = project_dir+"/cal_run_logs/binary."+args.fname+".pkl"
fname = args.fname
reps = 5

for iteration in range(reps):
    # extract appropriate dataset split indices
    split = iteration+1

    train_indices=pd.read_csv(project_dir+"real-data/train_indices_%d.csv"%split)['x']-1
    test_indices=pd.read_csv(project_dir+"real-data/test_indices_%d.csv"%split)['x']-1
    val_indices=pd.read_csv(project_dir+"real-data/val_indices_%d.csv"%split)['x']-1
    
    # extract required covariates for the dataset split 
    covariates=pd.read_csv(project_dir+"gwas_covariates.covar", header=None, sep="\t").to_numpy()
    # below, train dataset is for the recalibrator model (thus, it is actually an independent calibration dataset that we derive as a subset of heldout test dataset when doing GWAS)
    standardize = StandardScaler().fit(covariates[train_indices])
    covar_train = standardize.transform(covariates[test_indices[:1000]])[:,2:]
    # dataset to test our methods
    covar_test = standardize.transform(covariates[test_indices[1000:2000]])[:,2:]
    # we use the below heldout validation dataset subset to train a logistic regression model
    covar_val = standardize.transform(covariates[val_indices[:1000]])[:,2:]
    # GWAS dataset subset
    covar_gwas = standardize.transform(covariates[train_indices[:1000]])[:,2:]

    # gev samples
    posterior_samples = pd.read_csv(project_dir+"/real-data/test_posterior_samples/"+fname+"_"+str(split)+"_1.csv").to_numpy() 
    posterior_samples_val = pd.read_csv(project_dir+"/real-data/val_posterior_samples/"+fname+"_"+str(split)+"_1.csv").to_numpy() 
    posterior_samples_gwas = pd.read_csv(project_dir+"/real-data/train_posterior_samples/"+fname+"_"+str(split)+"_1.csv").to_numpy()

    # outcomes
    y = pd.read_csv(project_dir+"/real_pheno_binary/"+fname+".csv",header=None, sep='\t').to_numpy()[:, 2]
    y_train = y[test_indices[:1000]]
    y_test=y[test_indices[1000:2000]]
    y_val = y[val_indices[:1000]]
    y_gwas = y[train_indices[:1000]]

    
    if(normalize):
        posterior_samples = posterior_samples - posterior_samples.mean(axis=0)
        posterior_samples_val = posterior_samples_val - posterior_samples_val.mean(axis=0)
        posterior_samples_gwas = posterior_samples_gwas - posterior_samples_gwas.mean(axis=0)
        
        # y_train = y_train - y_train.mean()
        # y_test = y_test - y_test.mean()
        # y_val = y_val - y_val.mean()
        # y_gwas = y_gwas - y_gwas.mean()

    # set inputs for LR to be gev means
    X = posterior_samples.astype(float).mean(axis=1)
    X_gev_train = X[:SUBSET]
    X_gev_test = X[SUBSET:]
    X_gev_val = posterior_samples_val.astype(float).mean(axis=1)
    X_gev_gwas = posterior_samples_gwas.astype(float).mean(axis=1)
    
    
    # remove NA values
    
    X_gev_train = X_gev_train[np.logical_not(np.isnan(y_train))]
    X_gev_test = X_gev_test[np.logical_not(np.isnan(y_test))]
    X_gev_val = X_gev_val[np.logical_not(np.isnan(y_val))]
    X_gev_gwas = X_gev_gwas[np.logical_not(np.isnan(y_gwas))]

    covar_train = covar_train[np.logical_not(np.isnan(y_train))]
    covar_test = covar_test[np.logical_not(np.isnan(y_test))]
    covar_val = covar_val[np.logical_not(np.isnan(y_val))]
    covar_gwas = covar_gwas[np.logical_not(np.isnan(y_gwas))]
    

    y_train = y_train[np.logical_not(np.isnan(y_train))]
    y_test = y_test[np.logical_not(np.isnan(y_test))]
    y_val = y_val[np.logical_not(np.isnan(y_val))]
    y_gwas = y_gwas[np.logical_not(np.isnan(y_gwas))]

        

    X_train = np.concatenate((X_gev_train.reshape(-1, 1), covar_train), axis=1)
    X_val = np.concatenate((X_gev_val.reshape(-1, 1), covar_val), axis=1)
    X_test = np.concatenate((X_gev_test.reshape(-1, 1), covar_test), axis=1)
    X_gwas = np.concatenate((X_gev_gwas.reshape(-1, 1), covar_gwas), axis=1)

    print("num positives in y_train, y_val, y_test and y_gwas", sum(y_train), sum(y_val), sum(y_test), sum(y_gwas))
    print("length of y_train, y_val, y_test and y_gwas", len(y_train), len(y_val), len(y_test), len(y_gwas))
    
    # clf = LogisticRegression(random_state=0, C=1, class_weight="balanced").fit(X_val, y_val)
    clf = LogisticRegression(random_state=0, C=1, class_weight={0:1, 1:5.5}).fit(X_val, y_val)
    # pdb.set_trace()
    X_calib_train = clf.predict_proba(X_train)[:,1].reshape(-1,1)
    X_calib_test = clf.predict_proba(X_test)[:,1].reshape(-1,1)
    X_calib_val = clf.predict_proba(X_val)[:,1].reshape(-1,1)
    X_calib_gwas = clf.predict_proba(X_gwas)[:,1].reshape(-1,1)

    # calibrated_clf = CalibratedClassifierCV(clf, cv="prefit", method='isotonic').fit(X_train, y_train)
    calibrated_clf = LogisticRegression(random_state=0, C=0.1, class_weight={0:1, 1:5.5}).fit(clf.predict_proba(X_train)[:,1].reshape(-1,1), y_train)
    '''plot_calibration_curves(X_train, y_train, clf, calibrated_clf, project_dir+"/real_plot_results/binary.train."+fname+str(split)+".png")
    plot_calibration_curves(X_test, y_test, clf, calibrated_clf, project_dir+"/real_plot_results/binary.test."+fname+str(split)+".png")
    plot_calibration_curves(X_val, y_val, clf, calibrated_clf, project_dir+"/real_plot_results/binary.val."+fname+str(split)+".png")
    plot_calibration_curves(X_gwas, y_gwas, clf, calibrated_clf, project_dir+"/real_plot_results/binary.gwas."+fname+str(split)+".png")
    '''
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_train, y_train, clf)
    final_results["ppv_score_before"]["train"].append(ppv)
    final_results["npv_score_before"]["train"].append(npv)
    final_results["f1_score_before"]["train"].append(f1_score)
    final_results["auc_score_before"]["train"].append(auc)
    
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_val, y_val, clf)
    final_results["ppv_score_before"]["val"].append(ppv)
    final_results["npv_score_before"]["val"].append(npv)
    final_results["f1_score_before"]["val"].append(f1_score)
    final_results["auc_score_before"]["val"].append(auc)
    
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_test, y_test, clf)
    final_results["ppv_score_before"]["test"].append(ppv)
    final_results["npv_score_before"]["test"].append(npv)
    final_results["f1_score_before"]["test"].append(f1_score)
    final_results["auc_score_before"]["test"].append(auc)
    
    #ppv, npv, tpr, f1_score, auc = assess_classifier(X_gwas, y_gwas, clf)
    final_results["ppv_score_before"]["gwas"].append(ppv)
    final_results["npv_score_before"]["gwas"].append(npv)
    final_results["f1_score_before"]["gwas"].append(f1_score)
    final_results["auc_score_before"]["gwas"].append(auc)
    
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_calib_train, y_train, calibrated_clf)
    final_results["ppv_score_after"]["train"].append(ppv)
    final_results["npv_score_after"]["train"].append(npv)
    final_results["f1_score_after"]["train"].append(f1_score)
    final_results["auc_score_after"]["train"].append(auc)
    
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_calib_val, y_val, calibrated_clf)
    final_results["ppv_score_after"]["val"].append(ppv)
    final_results["npv_score_after"]["val"].append(npv)
    final_results["f1_score_after"]["val"].append(f1_score)
    final_results["auc_score_after"]["val"].append(auc)
    
    ppv, npv, tpr, f1_score, auc = assess_classifier(X_calib_test, y_test, calibrated_clf)
    final_results["ppv_score_after"]["test"].append(ppv)
    final_results["npv_score_after"]["test"].append(npv)
    final_results["f1_score_after"]["test"].append(f1_score)
    final_results["auc_score_after"]["test"].append(auc)
    
    #ppv, npv, tpr, f1_score, auc = assess_classifier(X_calib_gwas, y_gwas, clf)
    final_results["ppv_score_after"]["gwas"].append(ppv)
    final_results["npv_score_after"]["gwas"].append(npv)
    final_results["f1_score_after"]["gwas"].append(f1_score)
    final_results["auc_score_after"]["gwas"].append(auc)
    
    cal_before_train, cal_after_train = ece(y_train, clf.predict_proba(X_train)[:,1], n_bins=10), ece(y_train, calibrated_clf.predict_proba(X_calib_train)[:,1], n_bins=10)
    cal_before_test, cal_after_test = ece(y_test, clf.predict_proba(X_test)[:,1], n_bins=10), ece(y_test, calibrated_clf.predict_proba(X_calib_test)[:,1], n_bins=10)
    cal_before_val, cal_after_val = ece(y_val, clf.predict_proba(X_val)[:,1], n_bins=10), ece(y_val, calibrated_clf.predict_proba(X_calib_val)[:,1], n_bins=10)
    cal_before_gwas, cal_after_gwas = ece(y_gwas, clf.predict_proba(X_gwas)[:,1], n_bins=10), ece(y_gwas, calibrated_clf.predict_proba(X_calib_gwas)[:,1], n_bins=10)
    
    
    

    rvalue_train = (pearsonr(clf.predict_proba(X_train)[:,1], y_train)[0])**2
    rvalue_test = (pearsonr(clf.predict_proba(X_test)[:,1], y_test)[0])**2
    rvalue_val = (pearsonr(clf.predict_proba(X_val)[:,1], y_val)[0])**2
    rvalue_gwas = (pearsonr(clf.predict_proba(X_gwas)[:,1], y_gwas)[0])**2

    final_results["rvalue_before"]["train"].append(rvalue_train)
    final_results["rvalue_before"]["val"].append(rvalue_val)
    final_results["rvalue_before"]["test"].append(rvalue_test)
    final_results["rvalue_before"]["gwas"].append(rvalue_gwas)

    rvalue_train = (pearsonr(calibrated_clf.predict_proba(X_calib_train)[:,1], y_train)[0])**2
    rvalue_test = (pearsonr(calibrated_clf.predict_proba(X_calib_test)[:,1], y_test)[0])**2
    rvalue_val = (pearsonr(calibrated_clf.predict_proba(X_calib_val)[:,1], y_val)[0])**2
    rvalue_gwas = (pearsonr(calibrated_clf.predict_proba(X_calib_gwas)[:,1], y_gwas)[0])**2


    final_results["rvalue_after"]["train"].append(rvalue_train)
    final_results["rvalue_after"]["val"].append(rvalue_val)
    final_results["rvalue_after"]["test"].append(rvalue_test)
    final_results["rvalue_after"]["gwas"].append(rvalue_gwas)

    
    final_results["cal_score_before"]["train"].append(cal_before_train)
    final_results["cal_score_before"]["val"].append(cal_before_val)
    final_results["cal_score_before"]["test"].append(cal_before_test)
    final_results["cal_score_before"]["gwas"].append(cal_before_gwas)

    final_results["cal_score_after"]["train"].append(cal_after_train)
    final_results["cal_score_after"]["val"].append(cal_after_val)
    final_results["cal_score_after"]["test"].append(cal_after_test)
    final_results["cal_score_after"]["gwas"].append(cal_after_gwas)



pickle.dump(final_results, open(save_file, "wb"))

print("==================================")
print("Final metrics")
print("==================================")
print("Dataset", "Train", "Test", "Val", "GWAS", sep="\t")
print("Rvalue before [Mean]", np.array(final_results["rvalue_before"]["train"]).mean(), np.array(final_results["rvalue_before"]["test"]).mean(), np.array(final_results["rvalue_before"]["val"]).mean(), np.array(final_results["rvalue_before"]["gwas"]).mean(), sep="\t")
print("Rvalue before [Std error]", np.array(final_results["rvalue_before"]["train"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["test"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["val"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["gwas"]).std()/np.sqrt(5), sep="\t")


print("Rvalue after [Mean]", np.array(final_results["rvalue_after"]["train"]).mean(), np.array(final_results["rvalue_after"]["test"]).mean(), np.array(final_results["rvalue_after"]["val"]).mean(), np.array(final_results["rvalue_after"]["gwas"]).mean(), sep="\t")
print("Rvalue after [Std error]", np.array(final_results["rvalue_after"]["train"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["test"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["val"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["gwas"]).std()/np.sqrt(5), sep="\t")

print("Cal score before [Mean]",np.array(final_results["cal_score_before"]["train"]).mean(), np.array(final_results["cal_score_before"]["test"]).mean(), np.array(final_results["cal_score_before"]["val"]).mean(), np.array(final_results["cal_score_before"]["gwas"]).mean(),  sep='\t')
print("Cal score before [Std error]",np.array(final_results["cal_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("Cal score after [Mean]", np.array(final_results["cal_score_after"]["train"]).mean(), np.array(final_results["cal_score_after"]["test"]).mean(), np.array(final_results["cal_score_after"]["val"]).mean(), np.array(final_results["cal_score_after"]["gwas"]).mean(), sep='\t')
print("Cal score after [Std error]", np.array(final_results["cal_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("PPV score before [Mean]",np.array(final_results["ppv_score_before"]["train"]).mean(), np.array(final_results["ppv_score_before"]["test"]).mean(), np.array(final_results["ppv_score_before"]["val"]).mean(), np.array(final_results["ppv_score_before"]["gwas"]).mean(),  sep='\t')
print("PPV score before [Std error]",np.array(final_results["ppv_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["ppv_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["ppv_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["ppv_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("PPV score after [Mean]", np.array(final_results["ppv_score_after"]["train"]).mean(), np.array(final_results["ppv_score_after"]["test"]).mean(), np.array(final_results["ppv_score_after"]["val"]).mean(), np.array(final_results["ppv_score_after"]["gwas"]).mean(), sep='\t')
print("PPV score after [Std error]", np.array(final_results["ppv_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["ppv_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["ppv_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["ppv_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("NPV score before [Mean]",np.array(final_results["npv_score_before"]["train"]).mean(), np.array(final_results["npv_score_before"]["test"]).mean(), np.array(final_results["npv_score_before"]["val"]).mean(), np.array(final_results["npv_score_before"]["gwas"]).mean(),  sep='\t')
print("NPV score before [Std error]",np.array(final_results["npv_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["npv_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["npv_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["npv_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("NPV score after [Mean]", np.array(final_results["npv_score_after"]["train"]).mean(), np.array(final_results["npv_score_after"]["test"]).mean(), np.array(final_results["npv_score_after"]["val"]).mean(), np.array(final_results["npv_score_after"]["gwas"]).mean(), sep='\t')
print("NPV score after [Std error]", np.array(final_results["npv_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["npv_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["npv_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["npv_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("F1 score before [Mean]",np.array(final_results["f1_score_before"]["train"]).mean(), np.array(final_results["f1_score_before"]["test"]).mean(), np.array(final_results["f1_score_before"]["val"]).mean(), np.array(final_results["f1_score_before"]["gwas"]).mean(),  sep='\t')
print("F1 score before [Std error]",np.array(final_results["f1_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["f1_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["f1_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["f1_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("F1 score after [Mean]", np.array(final_results["f1_score_after"]["train"]).mean(), np.array(final_results["f1_score_after"]["test"]).mean(), np.array(final_results["f1_score_after"]["val"]).mean(), np.array(final_results["f1_score_after"]["gwas"]).mean(), sep='\t')
print("F1 score after [Std error]", np.array(final_results["f1_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["f1_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["f1_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["f1_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("AUC score before [Mean]",np.array(final_results["auc_score_before"]["train"]).mean(), np.array(final_results["auc_score_before"]["test"]).mean(), np.array(final_results["auc_score_before"]["val"]).mean(), np.array(final_results["auc_score_before"]["gwas"]).mean(),  sep='\t')
print("AUC score before [Std error]",np.array(final_results["auc_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["auc_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["auc_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["auc_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("AUC score after [Mean]", np.array(final_results["auc_score_after"]["train"]).mean(), np.array(final_results["auc_score_after"]["test"]).mean(), np.array(final_results["auc_score_after"]["val"]).mean(), np.array(final_results["auc_score_after"]["gwas"]).mean(), sep='\t')
print("AUC score after [Std error]", np.array(final_results["auc_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["auc_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["auc_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["auc_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')
