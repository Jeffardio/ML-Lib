from application_parameters import *
from probability import *
from validation import single_split
import numpy




def MultivariateGaussian(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Multivariate Gaussian Model--------")
    mu_cn, C_cn = compute_MG_parameters(DTR, LTR)
    SPost = compute_MG_SPost_matrix(DTE, mu_cn, C_cn)
    S = compute_score(SPost)
    return S, "MG"


def NaiveBaiyes(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Naive Baiyes Model--------")
    mu_cn, C_cn = compute_NB_parameters(DTR, LTR)
    SPost = compute_MG_SPost_matrix(DTE, mu_cn, C_cn)
    S = compute_score(SPost)
    return S, "NB"

def TiedNaiveBayes(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Tied Naive Baiyes Model--------")
    
    mu_cn, C_cn = compute_TNB_parameters(DTR, LTR)
    SPost = compute_MG_SPost_matrix(DTE, mu_cn, C_cn)
    S = compute_score(SPost)
    return S, "TNB"

def TiedMultivariateGaussian(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, verbose = False):
    if verbose:
        print("--------Tied Baiyes Model--------")
    mu_cn, C_cn = compute_TMG_parameters(DTR, LTR)
    
    SPost = compute_MG_SPost_matrix(DTE, mu_cn, C_cn)
    S = compute_score(SPost)
    return S, "TMG"

def LogisticRegression(DTR: numpy.ndarray, LTR:numpy.ndarray, DTE: numpy.ndarray, _lambda, prior = [0.5,0.5], verbose = False):
    if verbose:
        print("--------Logistic Regression Model--------")
        print(f"Lambda =  {_lambda}")
    
    obj = LR_obj_wrap(DTR,LTR,_lambda, prior)
    (x,f,d) = scipy.optimize.fmin_l_bfgs_b(obj,numpy.zeros((DTR.shape[0]+1)), approx_grad = True, factr=1.0)
    w_opt =  vcol(x[0:-1])
    b_opt = x[-1]
    S = compute_LR_score_matrix(DTE, w_opt, b_opt)
    return S, f"LR_{prior_probability[1]}_lambda_{_lambda}"

#actually not a classifier
def Calibration(not_calibrated_score: numpy.ndarray, labels : numpy.ndarray, prior = [0.5,0.5]):
    DTR, LTR, DTE, LTE = single_split(not_calibrated_score, labels)
    DTR = vrow(DTR)
    DTE = vrow(DTE)
    calibrated_score = LogisticRegression(DTR, LTR, DTE, 0, prior)[0]
    calibrated_score = calibrated_score - numpy.log(prior[1] / prior[0])
    return calibrated_score, LTE
    #to assess results
    #return DTE, calibrated_score, LTE 

def QuadraticLogisticRegression(DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray, _lambda: float, prior = [0.5, 0.5]  ,verbose = False):
    if verbose:
        print("--------Quadratic Logistic Regression Model--------")
        print(f"Lambda =  {_lambda}")
    DTR, DTE = polynomial_trasformation(DTR, DTE)
    return LogisticRegression(DTR, LTR, DTE, _lambda, prior)[0], f"QLR_{prior_probability[1]}_lambda_{_lambda}"



def SupportVectorMachine(DTR, LTR, DTE, K, C, prior = [0.5,0.5], verbose = True):
    if verbose:
        print("--------Support Vector Machine Model--------")
        print(f"K = {K}, C = {C}")
    w_opt = compute_SVM_parameters(DTR,LTR,K, C, prior)
    EXTDTE = numpy.vstack([DTE, numpy.ones(DTE.shape[1]) * K])
    S = compute_LR_score_matrix(EXTDTE, w_opt, 0)
    return S, f"SVM_{prior_probability[1]}_K_{K}_C_{C}"

def PolynomialSupportVectorMachine(DTR, LTR, DTE, K, C, d, c, prior = [0.5,0.5], verbose = True):
    if verbose:
        print("--------Polynomial Support Vector Machine Model--------")
        print(f"K = {K}, C = {C}, d = {d}, c = {c}")
    Z, _, _ = z_vector(LTR)
    KDTR = (DTR.T.dot(DTR) + c) ** d + K**2
    H_hat = vcol(Z) * vrow(Z) * KDTR
    alpha_opt = compute_PolSVM_parameters(H_hat, DTR, LTR, C, prior)
    S = compute_PolSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, c, d)
    return S, f"PSVM_{prior_probability[1]}_K_{K}_C_{C}_d_{d}_c_{c}"

def RadialSupportVectorMachine(DTR, LTR, DTE, K, C, gamma, prior = [0.5,0.5], verbose = True):
    if verbose:
        print("--------RBF Support Vector Machine Model--------")
        print(f"K = {K}, C = {C}, gamma = {gamma}")
    
    Z,_ ,_ = z_vector(LTR)
    KDTR = numpy.zeros((DTR.shape[1],DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            KDTR[i][j] = numpy.exp(-gamma * numpy.linalg.norm(DTR[:,i] - DTR[:,j]) **2) + (K)**2
    H_hat = vcol(Z) * vrow(Z) * KDTR
    alpha_opt = compute_RBFSVM_parameters(H_hat, DTR, LTR, C, prior)
    S = compute_RBFSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, gamma)
    return S, f"RSVM_{prior_probability[1]}__K_{K}_C_{C}_gamma_{gamma}"


def GaussianMixtureModel(DTR, LTR, DTE, iterations = 5, alpha = 0.1, psi = 0.01, mod="Normal", verbose = False):
    if verbose:
        print(f"Components:{2**iterations}, Mod:{mod}")
    GMM = compute_GMM_parameters(DTR, LTR, iterations, alpha, psi, mod)
    SPost = compute_GMM_SPost_matrix(DTE, GMM)
    S = compute_score(SPost)
    return S, f"GMM_components_{2**iterations}_mod_{mod}_alpha_{alpha}_psi_{psi}"

def Fusion(first_DTR,second_DTR,first_DTE,second_DTE,LTR):
    #
    DTR = numpy.vstack((first_DTR,second_DTR))
    DTE = numpy.vstack((first_DTE,second_DTE))
    
    score,_ = LogisticRegression(DTR,LTR,DTE,0, [0.5,0.5])
    return score, "Fusion"

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


