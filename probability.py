import scipy.linalg
import scipy.optimize
import scipy.stats
import numpy
from utils import *
from application_parameters import *



def logpdf_GAU_ND(X: numpy.ndarray, mu: numpy.ndarray, C: numpy.ndarray) -> numpy.ndarray:
    M = X.shape[0]
    _, det_C = numpy.linalg.slogdet(C)
    inv_C = numpy.linalg.inv(C)
    density_array = -0.5 * M * numpy.log(2 * numpy.pi) - 0.5 * det_C
    density_array = density_array - 0.5 * ((X - mu) * numpy.dot(inv_C, (X - mu))).sum(0)
    return density_array


def compute_ll(X: numpy.ndarray, mu: numpy.ndarray, C: numpy.ndarray) -> float:
    logpdf_array = logpdf_GAU_ND(X, mu, C)
    return logpdf_array.sum()


def compute_LLR(SJoint: numpy.ndarray):
    return SJoint[1] - SJoint[0]


def compute_class_posterior(SJoint: numpy.ndarray):
    for i in range(n_classes):
        SJoint[i] += eff_prior[i]
    return SJoint - vrow(scipy.special.logsumexp(SJoint, axis=0))


def compute_predicted_class(SPost: numpy.ndarray):
    return SPost.argmax(axis=0)


"""
GAUSSIAN MODEL
"""

def compute_MG_score_matrix(D: numpy.ndarray, mu_cn: numpy.ndarray, C_cn: numpy.ndarray):
    SJoint = numpy.zeros((n_classes, D.shape[1]))
    n_F = D.shape[0]
    for i in range(n_classes):
        SJoint[i:i+1, :] = logpdf_GAU_ND(D, mu_cn[:, i:i+1], C_cn[:, i*n_F: i*n_F + n_F])
    return SJoint

def compute_MG_parameters(D: numpy.ndarray, L: numpy.ndarray):
    n_F = D.shape[0]
    mu_cn = numpy.zeros((n_F, n_classes))
    C_cn = numpy.zeros((n_F, n_F*n_classes))
    for i in range(n_classes):
        mu_cn[:, i:i+1] = compute_mean(D[:, L == i])
        C_cn[:, i*n_F:i*n_F + n_F] = compute_covariance(D[:, L == i] - mu_cn[:, i:i+1])
    return mu_cn, C_cn

def compute_TMG_parameters(D: numpy.ndarray, L:numpy.ndarray):
    mu_cn, C_cn = compute_MG_parameters(D, L)
    n_F = D.shape[0]
    C = numpy.zeros((n_F,n_F))
    for i in range(n_classes):
        C += (L == i).sum() * C_cn[:,i*n_F:i*n_F + n_F]
    C *= 1/(D.shape[1])
    for i in range(n_classes):
        C_cn[:,i*n_F:i*n_F + n_F] = C
    return mu_cn, C_cn

def compute_TNB_parameters(D: numpy.ndarray, L: numpy.ndarray):
    mu_cn, C_cn = compute_MG_parameters(D, L)
    n_F = D.shape[0]
    for i in range(n_classes):
        C_cn[:, i*n_F:i*n_F + n_F] *= numpy.identity(n_F)
    C = numpy.zeros((n_F, n_F))
    for i in range(n_classes):
        C += (L == i).sum() * C_cn[:, i*n_F:i*n_F + n_F]
    C *= 1/(D.shape[1])
    for i in range(n_classes):
        C_cn[:, i*n_F:i*n_F + n_F] = C
    return mu_cn, C_cn


def compute_NB_parameters(D: numpy.ndarray, L: numpy.ndarray):
    mu_cn, C_cn = compute_MG_parameters(D, L)
    n_F = D.shape[0]
    for i in range(n_classes):
        C_cn[:,i*n_F:i*n_F + n_F] *= numpy.identity(n_F)
    return mu_cn, C_cn


"""
LOGISTIC REGRESSION
"""


def compute_LR_score_matrix(D: numpy.ndarray, W: numpy.ndarray, B: numpy.ndarray):
    return numpy.dot(W.T, D) + B


def LR_obj_wrap(DTR: numpy.ndarray, LTR: numpy.ndarray, _lambda: float):
    _, Z_f, Z_t = z_vector(LTR)

    def logreg_obj(V):
        w = vcol(V[0:-1])
        b = V[-1]
        n_T = DTR[:, LTR == 1].shape[1]
        n_F = DTR[:, LTR == 0].shape[1]
        S_true = compute_LR_score_matrix(DTR[:, LTR == 1], w, b)
        S_false = compute_LR_score_matrix(DTR[:, LTR == 0], w, b)
        w_true_costs = (prior_probability[1]/n_T) * \
            numpy.logaddexp(0, -Z_t * S_true).sum()
        w_false_costs = (prior_probability[0]/n_F) * \
            numpy.logaddexp(0, -Z_f * S_false).sum()
        return _lambda/2 * pow(numpy.linalg.norm(w), 2) + w_true_costs + w_false_costs
    return logreg_obj


def polynomial_trasformation(DTR: numpy.ndarray, DTE: numpy.ndarray):
    n_T = DTR.shape[1]
    n_E = DTE.shape[1]
    n_F = n_F**2 + n_F
    quad_DTR = numpy.zeros((n_F, n_T))
    quad_DTE = numpy.zeros((n_F, n_E))
    for i in range(n_T):
        x = DTR[:, i:i+1]
        quad_DTR[:, i:i+1] = stack(x)
    for i in range(n_E):
        x = DTE[:, i:i+1]
        quad_DTE[:, i:i+1] = stack(x)
    return quad_DTR, quad_DTE


def stack(array):
    n_F = array.shape[0]
    xxT = array @ array.T
    column = numpy.zeros((n_F ** 2 + n_F, 1))
    for i in range(n_F):
        column[i*n_F:i*n_F + n_F, :] = xxT[:, i:i+1]
    column[n_F ** 2: n_F ** 2 + n_F, :] = array
    return column


"""
SUPPORT VECTOR MACHINE
"""


def compute_weight_C(C, LTR):
    bounds = numpy.zeros((LTR.shape[0]))
    pi_t_emp = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * prior_probability[1] / pi_t_emp
    bounds[LTR == 0] = C * prior_probability[0] / (1 - pi_t_emp)
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))


def SVM_dual_obj_wrap(H_hat):
    def SVM_dual_obj(alpha: numpy.ndarray):
        alpha = vcol(alpha)
        gradient = vrow(H_hat.dot(alpha) - numpy.ones((alpha.shape[0], 1)))
        obj_L = 0.5 * alpha.T.dot(H_hat).dot(alpha) - alpha.T @ numpy.ones(alpha.shape[0])
        return obj_L, gradient
    return SVM_dual_obj


def compute_SVM_parameters(DTR, LTR, K, C):
    D_hat = numpy.vstack([DTR, numpy.ones(DTR.shape[1]) * K])
    G_hat = D_hat.T.dot(D_hat)
    Z, _, _ = z_vector(LTR)
    H_hat = vcol(Z) * vrow(Z) * G_hat
    dual_obj = SVM_dual_obj_wrap(H_hat)
    (alpha, f, _d) = scipy.optimize.fmin_l_bfgs_b(
        dual_obj, numpy.zeros(DTR.shape[1]), 
        bounds=compute_weight_C(C, LTR), 
        factr=100.0)
    w_opt = numpy.dot(D_hat, vcol(alpha) * vcol(Z))
    return w_opt


def compute_PolSVM_parameters(H_hat: numpy.ndarray, DTR: numpy.ndarray, LTR, C: numpy.ndarray):
    dual_obj = SVM_dual_obj_wrap(H_hat)
    (alpha, f, _d) = scipy.optimize.fmin_l_bfgs_b(
        dual_obj, 
        numpy.zeros(DTR.shape[1]),  
        bounds = compute_weight_C(C, LTR), 
        factr=100.0)
    return alpha


def compute_PolSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, c, d):
    return (vcol(alpha_opt) * vcol(Z) * ((DTR.T.dot(DTE) + c) ** d + K**2)).sum(0)


def compute_RBFSVM_parameters(H_hat: numpy.ndarray, DTR: numpy.ndarray, LTR, C: numpy.ndarray):
    dual_obj = SVM_dual_obj_wrap(H_hat)
    (alpha, f, _d) = scipy.optimize.fmin_l_bfgs_b(dual_obj, numpy.zeros(
        DTR.shape[1]),  bounds=compute_weight_C(C, LTR), factr=100.0)
    return alpha


def compute_RBFSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, gamma):
    exp_dist = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            exp_dist[i][j] += numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i:i+1] - DTE[:, j:j+1]) ** 2) + (K)**2
    return (vcol(alpha_opt) * vcol(Z) * (exp_dist)).sum(0)


"""
GAUSSIAN MIXTURE MODEL
"""


def logpdf_GMM(X: numpy.ndarray, gmm: list):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for idx, g in enumerate(gmm):
        w, mu, C = g[0], g[1], g[2]
        S[idx, :] = logpdf_GAU_ND(X, mu, C) + numpy.log(w)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def GMM_diagonal(GMM, _Z_vec, _N):
    for g in range((len(GMM))):
        sigma = GMM[g][2] * numpy.eye(GMM[g][2].shape[0])
        GMM[g] = (GMM[g][0], GMM[g][1], sigma)
    return GMM


def GMM_tied(GMM, Z_vec, N):

    tied_Sigma = numpy.zeros((GMM[0][2].shape[0], GMM[0][2].shape[0]))
    for g in range((len(GMM))):
        tied_Sigma += GMM[g][2] * Z_vec[g]
    tied_Sigma = (1/N) * tied_Sigma
    for g in range((len(GMM))):
        GMM[g] = (GMM[g][0], GMM[g][1], tied_Sigma)
    return GMM


def GMM_tied_diagonal(GMM, Z_vec, N):
    tied_GMM = GMM_tied(GMM, Z_vec, N)
    tied_and_diagonal = GMM_diagonal(tied_GMM, Z_vec, N)
    return tied_and_diagonal


def do_nothing(GMM, Z_vec, N):
    return GMM


GMM_fun = {
    "Diagonal": GMM_diagonal,
    "Tied": GMM_tied,
    "Normal": do_nothing,
    "Tied-Diagonal": GMM_tied_diagonal,
}


def logpdf_GAU_ND_Opt(X, mu, C):
    P = numpy.linalg.inv(C)
    const = - 0.5 * X.shape[0] * numpy.log(2 * numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const - 0.5 * numpy.dot((x - mu).T, numpy.dot(P, (x - mu)))
        Y.append(res)
    return numpy.array(Y).ravel()


def GMM_EM(X: numpy.ndarray, gmm: list, psi: float, mod: str):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_Opt(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        P = numpy.exp(SJ-SM)
        gmmNew = []
        Z_vec = numpy.zeros((G))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            Z_vec[g] = Z
            F = (vrow(gamma)*X).sum(1)
            S = numpy.dot(X, (vrow(gamma)*X).T)
            w = Z/N
            mu = vcol(F/Z)
            Sigma = S/Z - numpy.dot(mu, mu.T)
            gmmNew.append((w, mu, Sigma))
        gmmNew = GMM_fun[mod](gmmNew, Z_vec, N)
        for i in range(G):
            transformed_sigma = gmmNew[i][2]
            U, s, _ = numpy.linalg.svd(transformed_sigma)
            s[s < psi] = psi
            gmmNew[i] = (gmmNew[i][0], gmmNew[i][1], numpy.dot(U, vcol(s)*U.T))
        gmm = gmmNew
    return gmm


def GMM_LBG(iterations: int, X: numpy.ndarray, start_gmm: list, alpha: float, psi: float, mod: str):
    start_gmm = GMM_fun[mod](start_gmm, [X.shape[1]], X.shape[1])
    for i in range(len(start_gmm)):
        transformed_sigma = start_gmm[i][2]
        U, s, _ = numpy.linalg.svd(transformed_sigma)
        s[s < psi] = psi
        start_gmm[i] = (start_gmm[i][0], start_gmm[i]
                        [1], numpy.dot(U, vcol(s)*U.T))
    start_gmm = GMM_EM(X, start_gmm, psi, mod)
    for i in range(iterations):
        gmmNew = list()
        for g in (start_gmm):
            Sigma_g = g[2]
            U, s, _ = numpy.linalg.svd(Sigma_g)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            new_w = g[0]/2
            mu = g[1]
            gmmNew.append((new_w, mu + d, Sigma_g))
            gmmNew.append((new_w, mu - d, Sigma_g))
        start_gmm = GMM_EM(X, gmmNew, psi, mod)
    return start_gmm


def compute_GMM_parameters(DTR, LTR, iterations, alpha, psi, mod):
    GMM = list()
    for _ in range(n_classes):
        GMM.append(list())
    for i in range(n_classes):
        mu = compute_mean(DTR[:, LTR == i])
        C = compute_covariance(DTR[:, LTR == i] - mu)
        GMM[i] = [(1.0, mu, C)]
        GMM[i] = GMM_LBG(iterations, DTR[:, LTR == i],GMM[i], alpha, psi, mod)
    return GMM


def compute_GMM_score_matrix(DTE, GMM):
    S = numpy.zeros((n_classes, DTE.shape[1]))
    for idx, gmm in enumerate(GMM):
        S[idx] = logpdf_GMM(DTE, gmm)
    return S


def z_vector(L):
    Z = numpy.zeros((L.shape[0]))
    Z = 2*L - 1
    return Z, Z[Z == -1], Z[Z == 1]
