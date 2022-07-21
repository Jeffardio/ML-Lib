from ctypes import c_char
from application_parameters import *
from probability import *
from statistics import plot_bayes_error_plot, compute_accuracy, compute_minimum_NDCF
import numpy




def MultivariateGaussian(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Multivariate Gaussian Model--------")
    mu_cn, C_cn = compute_MG_parameters(DTR, LTR)
    SJoint = compute_MG_score_matrix(DTE, mu_cn, C_cn)
    LLR = compute_LLR(SJoint)
    return LLR, "MG"


def NaiveBaiyes(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Naive Baiyes Model--------")
    mu_cn, C_cn = compute_NB_parameters(DTR, LTR)
    SJoint = compute_MG_score_matrix(DTE, mu_cn, C_cn)
    LLR = compute_LLR(SJoint)
    return LLR, "NB"

def TiedNaiveBayes(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, LTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Tied Naive Baiyes Model--------")
    mu_cn, C_cn = compute_TNB_parameters(DTR, LTR)
    SJoint = compute_MG_score_matrix(DTE,LTE, mu_cn, C_cn)
    LLR = compute_LLR(SJoint)
    return LLR, "TNB"

def TiedMultivariateGaussian(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Tied Baiyes Model--------")
    mu_cn, C_cn = compute_TMG_parameters(DTR, LTR)
    SJoint = compute_MG_score_matrix(DTE, mu_cn, C_cn)
    LLR = compute_LLR(SJoint)
    return LLR, "TMG"

def LogisticRegression(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, _lambda, verbose = False):
    if verbose:
        print("--------Logistic Regression Model--------")
    obj = LR_obj_wrap(DTR,LTR,_lambda)
    (x,f,d) = scipy.optimize.fmin_l_bfgs_b(obj,numpy.zeros((DTR.shape[0]+1)), approx_grad = True, factr=100.0)
    w_opt =  vcol(x[0:-1])
    b_opt = x[-1]
    S = compute_LR_score_matrix(DTE, w_opt, b_opt)
    return S, f"LR_lambda_{_lambda:.3f}"

def QuadraticLogisticRegression(DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray, _lambda: float, verbose = False):
    if verbose:
        print("--------Quadratic Logistic Regression Model--------")
        print(f"Lambda =  {_lambda:.3f}")
    DTR, DTE = polynomial_trasformation(DTR, DTE)
    return LogisticRegression(DTR, LTR, DTE, _lambda), f"QLR_lambda_{_lambda:.3f}"





def SupportVectorMachine(DTR, LTR, DTE, K, C, verbose = False):
    if verbose:
        print("--------Support Vector Machine Model--------")
        print(f"K = {K}, C = {C:.3f}")
    w_opt = compute_SVM_parameters(DTR,LTR,K, C)
    EXTDTE = numpy.vstack([DTE, numpy.ones(DTE.shape[1]) * K])
    S = compute_LR_score_matrix(EXTDTE, w_opt, 0)
    return S, f"SVM_K_{K}_C_{C:.3f}"

def PolynomialSupportVectorMachine(DTR, LTR, DTE, K, C, d, c, verbose = False):
    if verbose:
        print("--------Polynomial Support Vector Machine Model--------")
        print(f"K = {K}, C = {C:.3f}, d = {d}, c = {c}")
    Z, _, _ = z_vector(LTR)
    KDTR = (DTR.T.dot(DTR) + c) ** d + K**2
    H_hat = vcol(Z) * vrow(Z) * KDTR
    alpha_opt = compute_PolSVM_parameters(H_hat, DTR, LTR, C)
    S = compute_PolSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, c, d)
    return S, f"PSVM_K_{K}_C_{C:.3f}_d_{d}_c_{c}"

def RadialSupportVectorMachine(DTR, LTR, DTE, K, C, gamma, verbose = False):
    if verbose:
        print("--------RBF Support Vector Machine Model--------")
        print(f"K = {K}, C = {C:.3f}, gamma = {gamma}")
    print("gamma = ",gamma)
    Z,_ ,_ = z_vector(LTR)
    KDTR = numpy.zeros((DTR.shape[1],DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            KDTR[i][j] = numpy.exp(-gamma * numpy.linalg.norm(DTR[:,i] - DTR[:,j]) **2) + (K)**2
    H_hat = vcol(Z) * vrow(Z) * KDTR
    alpha_opt = compute_RBFSVM_parameters(H_hat, DTR, LTR, C)
    S = compute_RBFSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, gamma)
    return S, f"RSVM_K_{K}_C_{C:.3f}_gamma_{gamma}"


def GaussianMixtureModel(DTR, LTR, DTE, iterations = 5, alpha = 0.1, psi = 0.01, mod="Normal", verbose = False):
    if verbose:
        print(f"Components:{2**iterations}, Mod:{mod}")
    GMM = compute_GMM_parameters(DTR, LTR, iterations, alpha, psi, mod)
    SJoint = compute_GMM_score_matrix(DTE, GMM)
    LLR = compute_LLR(SJoint)
    return LLR, f"GMM_components_{2**iterations}_mod_{mod}_alpha_{alpha}_psi_{psi}"


classifier_model = {
    "MG" : MultivariateGaussian,
    "TMG" : TiedMultivariateGaussian,
    "NB" : NaiveBaiyes,
    "TNB" : TiedNaiveBayes,
    "LR" : LogisticRegression,
    "QLR": QuadraticLogisticRegression,
    "SVM": SupportVectorMachine,
    "RSVM": RadialSupportVectorMachine,
    "PLR": PolynomialSupportVectorMachine,
    "PSVM": PolynomialSupportVectorMachine,
    "GMM": GaussianMixtureModel,
}


