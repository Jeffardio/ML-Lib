from application_parameters import *
import numpy
import matplotlib.pyplot as plt
import scipy

def compute_accuracy(predicted_class: numpy.ndarray, L: numpy.ndarray):
    return (predicted_class == L).sum() / L.shape[0]

def print_statistics(accuracy: float, min_dcf: float, t_star: float):
    print(f"--- Accuracy of the model: {accuracy} %")
    print(f"---- Min DCF of the model:  {min_dcf}")
    print(f"--- Best treshold: {t_star}")


def plot_bayes_error_plot(score,labels,model, calibrated=""):
    score = score.ravel()
    n_points = 100
    eff_prior_log_odds = numpy.linspace(-4, 4, n_points)
    dcf = numpy.zeros(n_points)
    mindcf = numpy.zeros(n_points)
    for (idx, p) in enumerate(eff_prior_log_odds):
        pi = 1 / (1 + numpy.exp(-p))
        dcf[idx] = compute_NDCF(score,labels, pi, 1, 1)
        mindcf[idx] = compute_minimum_NDCF(score, labels, pi, 1,1)[0]
    plt.plot(eff_prior_log_odds, dcf, label="actDCF", color="r")
    plt.plot(eff_prior_log_odds, mindcf ,':', label="minDCF", color="r")

    plt.title(model)
    plt.ylim([0,1])
    plt.xlim([-4, 4])
    plt.ylabel("DCF")
    plt.xlabel("$log \\frac{ \\tilde{\pi}}{1-\\tilde{\pi}}$")
    plt.legend()
    plt.savefig(f"figure/{model}_{calibrated}")
    plt.close()

def build_conf_mat_uniform(prediction, L):
    conf_mat = numpy.zeros((n_classes,n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
                conf_mat[i][j] = (1*numpy.bitwise_and(prediction == i, L == j)).sum()
    

    return conf_mat

def build_conf_mat(llr: numpy.ndarray,L: numpy.ndarray,pi:float, C_fn:float,C_fp:float):
    t = -numpy.log(pi*C_fn/((1-pi)*C_fp))
    predictions = 1*(llr > t)
    return build_conf_mat_uniform(predictions,L)


def compute_DCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
    conf_mat = build_conf_mat(llr, L, pi, C_fn, C_fp)
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    
    return pi * C_fn * FNR + (1-pi) * C_fp * FPR

def compute_NDCF(llr: numpy.ndarray, L: numpy.ndarray, pi: float, C_fn: float, C_fp: float):
    return compute_DCF(llr, L, pi, C_fn, C_fp) / min([pi*C_fn, (1-pi)*C_fp])

def compute_ROC_points(llr, L):
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    ROC_points_TPR = numpy.zeros(L.shape[0] +2 )
    ROC_points_FPR = numpy.zeros(L.shape[0] +2 )
    for (idx,t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        TPR = numpy.bitwise_and(pred == 1, L == 1 ).sum() / N_label1
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / N_label0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR

def compute_DET_points(llr, L):
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    DET_points_FNR = numpy.zeros(L.shape[0] +2 )
    DET_points_FPR = numpy.zeros(L.shape[0] +2 )
    for (idx,t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        FNR = 1 - (numpy.bitwise_and(pred == 1, L == 1 ).sum() / N_label1)
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / N_label0
        DET_points_FNR[idx] = FNR
        DET_points_FPR[idx] = FPR
    return DET_points_FNR, DET_points_FPR

def plot_ROC(llr, L):
    ROC_points_TPR, ROC_points_FPR = compute_ROC_points(llr, L)
    plt.plot(ROC_points_FPR, ROC_points_TPR, color='r')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.show()

def plot_ROC2(llr1,llr2, L,label1,label2):
    ROC_points_TPR1, ROC_points_FPR1 = compute_ROC_points(llr1, L)
    ROC_points_TPR2, ROC_points_FPR2 = compute_ROC_points(llr2, L)
    plt.plot(ROC_points_FPR1, ROC_points_TPR1, color='r', label=label1)
    plt.plot(ROC_points_FPR2, ROC_points_TPR2, color='b', label=label2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.savefig(f"figure/ROC_{label1}_{label2}")

def plot_ROC3(llr1,llr2,llr3, L,label1,label2,label3, eval):
    ROC_points_TPR1, ROC_points_FPR1 = compute_ROC_points(llr1, L)
    ROC_points_TPR2, ROC_points_FPR2 = compute_ROC_points(llr2, L)
    ROC_points_TPR3, ROC_points_FPR3 = compute_ROC_points(llr3, L)
    plt.plot(ROC_points_FPR1, ROC_points_TPR1, color='r', label=label1 + eval)
    plt.plot(ROC_points_FPR2, ROC_points_TPR2, color='b', label=label2 + eval)
    plt.plot(ROC_points_FPR3, ROC_points_TPR3, color='g', label=label3 + eval)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.savefig(f"figure/ROC_{label1}_{label2}_{label3}")

def plot_DET3(llr1,llr2,llr3, L,label1,label2,label3, eval):
    DET_points_TPR1, DET_points_FPR1 = compute_DET_points(llr1, L)
    DET_points_TPR2, DET_points_FPR2 = compute_DET_points(llr2, L)
    DET_points_TPR3, DET_points_FPR3 = compute_DET_points(llr3, L)
    plt.plot(DET_points_FPR1, DET_points_TPR1, color='r', label=label1 + eval)
    plt.plot(DET_points_FPR2, DET_points_TPR2, color='b', label=label2 + eval)
    plt.plot(DET_points_FPR3, DET_points_TPR3, color='g', label=label3 + eval)
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    
    plt.xscale('log')
    plt.yscale('log')
   
    plt.legend()
    plt.grid()
    plt.savefig(f"figure/DET_{label1}_{label2}_{label3}")

def compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn):
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    return (pi * C_fn * FNR + (1-pi) * C_fp * FPR)  / min([pi*C_fn, (1-pi)*C_fp])

def compute_minimum_NDCF(scores, L, pi, C_fp, C_fn):

    scores = scores.ravel()
    
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(scores),numpy.array([numpy.inf])])
    DCF = numpy.zeros(tresholds.shape[0])
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (scores > t)
        conf_mat = build_conf_mat_uniform(pred, L)
        DCF[idx] = compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn)
    argmin = DCF.argmin()
    return DCF[argmin], tresholds[argmin]

def plot_PC_heatmap(D: numpy.ndarray):
    heatmap = numpy.zeros((D.shape[0],D.shape[0]))
    for f1 in range(D.shape[0]):
        for f2 in range(D.shape[0]):
            if f2 <= f1:
                heatmap[f1][f2] = abs(scipy.stats.pearsonr(D[f1,:], D[f2,:])[0])
                heatmap[f2][f1] = heatmap[f1][f2]
    plt.figure()
    plt.xticks(numpy.arange(0,12),numpy.arange(1,13))  
    plt.yticks(numpy.arange(0,12),numpy.arange(1,13))          
        
    plt.imshow(heatmap, cmap='Blues')
    plt.savefig("figure/heatmap_male.png")


def plot_hist(D: numpy.ndarray, L: numpy.ndarray):
    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(features_name[i])
        for j in range(n_classes):
            plt.hist(D[:, L == j][i, :], bins = 100, density = True, alpha = 0.4, label=classes[j], linewidth = 1.0, edgecolor='black' )
        plt.legend()
        plt.tight_layout()
        plt.savefig('figure/hist_'+ features_name[i]+'ok.png')     
    plt.show()

def print_statistics(accuracy: float, act_dcf: float, min_dcf: float, t_star: float):
    
    print(f"{accuracy} {act_dcf} {min_dcf}")
    print(f"--- Accuracy of the model: {accuracy} %")
    print(f"--- Act DCF of the model: {act_dcf}")
    print(f"---- Min DCF of the model:  {min_dcf}")
    print(f"--- Best treshold: {t_star}")


classes = {
    0: "Male",
    1: "Female"
}

latex_simbol = {
    "lamba" : "$\ lamba$",
    "C" : "C"
}


def plot_minDCF(min_dcf_05, min_dcf_01, min_dcf_09,eval_dcf_05,eval_dcf_01,eval_dcf_09, pre_proc_tech, ranges, model):
    plt.figure()
    plt.title(pre_proc_tech)
    plt.xlabel('$C$')
    plt.ylabel("minDCF")
    plt.xscale('log')

    
    plt.plot(ranges, min_dcf_05, ':',label = "minDCF $(\\tilde{\pi} = 0.5)$ [Val]", color = 'r')
    plt.plot(ranges, min_dcf_01, ':',label = "minDCF $(\\tilde{\pi} = 0.1)$ [Val]", color = 'b')
    plt.plot(ranges, min_dcf_09, ':',label = "minDCF $(\\tilde{\pi} = 0.9)$ [Val]", color = 'g')
    

    plt.plot(ranges, eval_dcf_05, label = "minDCF $(\\tilde{\pi} = 0.5)$ [Eval]", color = 'r')
    plt.plot(ranges, eval_dcf_01, label = "minDCF $(\\tilde{\pi} = 0.1)$ [Eval]", color = 'b')
    plt.plot(ranges, eval_dcf_09, label = "minDCF $(\\tilde{\pi} = 0.9)$ [Eval]", color = 'g')
    """
    
    plt.plot(ranges, min_dcf_05, ':',label = "$\log\gamma = -1$ -- [Val]", color = 'r')
    plt.plot(ranges, min_dcf_01, ':',label = "$\log\gamma = -2$ -- [Val]", color = 'b')
    plt.plot(ranges, min_dcf_09, ':',label = "$\log\gamma = -3$ -- [Val]", color = 'g')
    

    plt.plot(ranges, eval_dcf_05, label = "$\log\gamma = -1$ -- [Eval]", color = 'r')
    plt.plot(ranges, eval_dcf_01, label = "$\log\gamma = -2$ -- [Eval]", color = 'b')
    plt.plot(ranges, eval_dcf_09, label = "$\log\gamma = -3$ -- [Eval]", color = 'g')
    
    """
    
    
    #plt.plot(ranges, min_dcf_01, label = "$log\gamma = -1$")
    #plt.plot(ranges, min_dcf_05, label = "$log\gamma = -2$")
    #plt.plot(ranges, min_dcf_09, label = "$log\gamma = -3$")
    
    
    plt.xlim(ranges[0], ranges[-1])
    plt.legend()
    plt.savefig("test_plot/" + model + "/new_" + pre_proc_tech + ".png")

def plot_minDCF_GMM(gauss_dcf, zscore_dcf, bounds, mod):
    plt.figure()
    plt.title(mod)
    plt.xlabel("GMM components")
    plt.ylabel("minDCF")
    x_axis = numpy.arange(len(bounds))
    bounds = numpy.array(bounds)
    plt.bar(x_axis + 0.00 , gauss_dcf, width = 0.25,linewidth = 1.0, edgecolor='black', color="Red", label = "Gaussianization")
    plt.bar(x_axis + 0.25, zscore_dcf, width = 0.25,linewidth = 1.0, edgecolor='black', color="Orange" ,label = "Z-Score")
    plt.xticks([r + 0.125 for r in range(len(bounds))],
        bounds)
    plt.legend()
    plt.savefig("plot/GMM/new_" + mod + ".png")

def K_fold_split(D, L, K, seed = 27):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    chunks = numpy.array_split(idx, K)
    folds = []
    folds_label = []
    for i in range(K):
        folds.append(D[:,chunks[i]])
        folds_label.append(L[chunks[i]])
    return folds, folds_label

def next_kfold_iter(folds, folds_label):
    folds = [folds[1], *folds[2:], folds[0]]
    folds_label = [folds_label[1], *folds_label[2:], folds_label[0]]
    return folds, folds_label

def train_and_eval(folds, folds_label):
    DTR = numpy.concatenate(folds[1:], axis = 1)
    LTR = numpy.concatenate(folds_label[1:])
    DTE = folds[0]
    LTE = folds_label[0]
    return DTR, LTR, DTE, LTE    

def single_split(D,L):
    numpy.random.seed(10)
    idx = numpy.random.permutation(D.shape[0])

    train = idx[:int(D.shape[0] * 0.5)]
    eval = idx[int(D.shape[0] * 0.5):]
    return D[train],L[train],D[eval],L[eval]