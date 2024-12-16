# BP

import numpy as np
import stim
import pymatching
from color_code_stim import *
from scipy.sparse import csc_matrix

# =============================================================================
# D=5
# T=5
# p=0.001
# 
# colorcode = ColorCode(d=D,
#                 rounds=T,
#                 cnot_schedule='LLB',  # Default CNOT schedule optimized in our paper.
#                 p_circuit=p)
# =============================================================================

# =============================================================================
# dems = {}
# for color in ['r', 'g', 'b']:
#     dem1, dem2 = colorcode.decompose_detector_error_model(color)
#     dems[color] = dem1, dem2  # stim.DetectorErrorModel
# =============================================================================

circuit = stim.Circuit.generated("surface_code:rotated_memory_x", 
                                 distance=3, 
                                 rounds=3, 
                                 after_clifford_depolarization=0.005)

model = circuit.detector_error_model(decompose_errors=True)
#matching = pymatching.Matching.from_detector_error_model(model)

sampler = circuit.compile_detector_sampler()
#syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
syndrome = np.array([[False, False, False, False, False, False, False, False, False,False, False, False, False, False, False,  True, False, False,
       False, False, False, False, False, False]])

T = 30

def BP(DEM, syndrome, T):
    t = 0
    x = np.array([0 for i in range(DEM.num_errors)])
    M = []
    N = []
    E = {}
    L = {}#L(x|y)
    L1 = {}#L(nm)
    Z = {}#Znm
    Z1 = {}#Zn
    
    for i in range(DEM.num_errors):
        N.append(i)
        Z1[i] = 0
        l0 = DEM[i].targets_copy()
        e = DEM[i].args_copy()[0]
        L[i] = np.log((e/(1-e)))#np.log((1-e)/e)
        for node in l0:
            if node == stim.target_separator():
                continue
            m = node.val
            if node == stim.target_logical_observable_id(m):
                break
            else:
                E[(i,m)] = e                
                Z[(i,m)] = np.log((e/(1-e)))#np.log(((1-e)/e))
                if m in M:
                    continue
                else:
                    M.append(m)

    H = np.array([[0 for i in range(DEM.num_errors)] for j in range(len(syndrome))])
    
    for i in range(len(syndrome)):
        for j in range(DEM.num_errors):
            if (j,i) in E.keys():
                H[i][j] = 1
    
    while True:# np.any(np.dot(H, x) - syndrome) is False:
        t += 1
        for m in M:
            for n in N:
                if (n,m) in E.keys():
                    if syndrome[m] == 0:
                        L1[(n,m)] = Lmn(M, N, m, n, Z, E)
                    else:
                        L1[(n,m)] = -Lmn(M, N, m, n, Z, E)
                
        for n in N:
            for m in M:
                if (n,m) in E.keys():
                    Z[(n,m)] = Znm(n, m, L, L1)
                
        for n in N:
            Z1[n] = Zn(n, L, L1)
        
        for i in range(len(N)):
            if Z1[i] >= 0:
                x[i] = 0
            elif Z1[i] < 0:
                x[i] = 1
        
        
        if np.all(np.dot(H, x)%2 == syndrome%2) or t == T:
            #print(t)
            break
    
    dem = stim.DetectorErrorModel()
    #print(L1)
    
    for i in range(DEM.num_errors):
        detectors = DEM[i].targets_copy()
        deminstruction = stim.DemInstruction(DEM[i].type, DEM[i].args_copy(), detectors)
        dem.append(deminstruction)
        
    for i in range(DEM.num_detectors):
        detectors = DEM[DEM.num_errors + i].targets_copy()
        #print(detectors)
        deminstruction = stim.DemInstruction(DEM[DEM.num_errors + i].type, DEM[DEM.num_errors + i].args_copy(), detectors)
        dem.append(deminstruction)
    
    for i in range(len(DEM)):
        detectors = DEM[i].targets_copy()
        if i < DEM.num_errors:
            v = Z1[i]
            v1 = 1/(1+np.e**v)
            deminstruction = stim.DemInstruction(DEM[i].type, [v1], detectors)
        else:
            deminstruction = stim.DemInstruction(DEM[i].type, DEM[i].args_copy(), detectors)
        dem.append(deminstruction)
    
    return x, dem, Z1, H, Z, L, L1
        
def Lmn(M, N, m, n, Z, E):     
    lmn = 1
    s = 1
    for edge in Z.keys():
        if edge[0] != n and edge[1] == m:
            lmn *= np.tanh(abs(Z[edge])/2)
            #Lmn += F(abs(Z[edge]))
            s *= np.sign(Z[edge])
    #Lmn = s*F(Lmn)#s*2*np.arctanh(Lmn/(1e17))
    Lmn = s*2*np.arctanh(lmn)
    
    return Lmn

def Znm(n, m, L, L1):
    Zmn = L[n]
    for edge in L1.keys():
        if edge[0] == n and edge[1] != m:
            Zmn += L1[edge]
    return Zmn

def Zn(n, L, L1):
    Zn = L[n]
    for edge in L1.keys():
        if edge[0] == n:
            Zn += L1[edge]
    return Zn

def F(x):
    return np.log((np.e**x+1)/(np.e**x-1))

def G(x):
    return 1/(1+np.e**x)

for i in range(len(syndrome)):
    x, dem, Z1, H, Z, L, L1 = BP(model, syndrome[i], T)
    if np.any(x):
# =============================================================================
#         print(model)
#         print(dem)
#         print(Z1)
# =============================================================================
        print(i)
        
        matching1 = pymatching.Matching.from_detector_error_model(dem)
        
class BipartiteGraphBP:
    def __init__(self, num_vars, num_checks, edges, prior_llrs, syndrome, H):
        """
        初始化针对二部图的 LLR 信念传播算法
        :param num_vars: 变量节点的数量
        :param num_checks: 检查节点的数量
        :param edges: 二部图中的边 [(var, check), ...]
        :param prior_llrs: 变量节点的先验 LLR 值
        """
        self.num_vars = num_vars
        self.num_checks = num_checks
        self.edges = edges
        self.prior_llrs = np.array(prior_llrs)
        self.var_to_check_msgs = {edge: 0.0 for edge in edges}  # 初始化变量到检查节点的消息
        self.check_to_var_msgs = {edge[::-1]: 0.0 for edge in edges}  # 初始化检查到变量节点的消息
        self.syndrome = syndrome
        self.H = H

    @staticmethod
    def tanh_sum_rule(llrs):
        """
        使用 tanh-sum rule 更新消息
        :param llrs: 邻居消息的 LLR 值列表
        :return: 更新后的消息
        """
        if not llrs:
            return np.inf
        product = np.prod(np.tanh(np.array(llrs) / 2))
        return 2 * np.arctanh(product)

    def run(self, max_iters=30, tol=1e-20):
        """
        运行信念传播算法
        :param max_iters: 最大迭代次数
        :param tol: 收敛阈值
        """
        for _ in range(max_iters):
            new_var_to_check_msgs = {}
            new_check_to_var_msgs = {}

            # 变量节点更新消息到检查节点
            for (v, c) in self.var_to_check_msgs:
                # 收集变量节点 v 的所有邻居检查节点的消息，排除检查节点 c
                neighbors = [check for (var, check) in self.var_to_check_msgs if var == v and check != c]
                incoming_msgs = [self.check_to_var_msgs[(check, v)] for check in neighbors]
                new_var_to_check_msgs[(v, c)] = self.prior_llrs[v] + sum(incoming_msgs)
            
            
            #print(new_var_to_check_msgs)
            # 检查节点更新消息到变量节点
            for (c, v) in self.check_to_var_msgs:
                # 收集检查节点 c 的所有邻居变量节点的消息，排除变量节点 v
                neighbors = [var for (var, check) in self.var_to_check_msgs if check == c and var != v]
                incoming_msgs = [self.var_to_check_msgs[(var, c)] for var in neighbors]
                if self.syndrome[c] == 0:
                    new_check_to_var_msgs[(c, v)] = self.tanh_sum_rule(incoming_msgs)
                else:
                    new_check_to_var_msgs[(c, v)] = -self.tanh_sum_rule(incoming_msgs)
            #print(new_check_to_var_msgs)

            # 检查收敛条件
            max_change = max(
                abs(new_var_to_check_msgs[edge] - self.var_to_check_msgs[edge]) for edge in self.var_to_check_msgs
            )
            self.var_to_check_msgs = new_var_to_check_msgs
            self.check_to_var_msgs = new_check_to_var_msgs

            marginals = self.marginal_llrs()
            x = []
            for n in marginals:
                if n < 0:
                    x.append(1)
                else:
                    x.append(0)
            x = np.array(x)
            #print(x)

            #print(_)
            #if max_change < tol and :
            if np.all(np.dot(self.H, x)%2 == self.syndrome%2):
# =============================================================================
#                 print(new_var_to_check_msgs)
#                 print(new_check_to_var_msgs)
# =============================================================================
                print(_)
                break

    def marginal_llrs(self):
        """
        计算所有变量节点的边缘 LLR
        :return: 每个变量节点的边缘 LLR 列表
        """
        marginals = []
        for v in range(self.num_vars):
            neighbors = [check for (var, check) in self.var_to_check_msgs if var == v]
            incoming_msgs = [self.check_to_var_msgs[(check, v)] for check in neighbors]
# =============================================================================
#             print(v)
#             print(incoming_msgs)
# =============================================================================
            marginals.append(self.prior_llrs[v] + sum(incoming_msgs))
        return marginals

    def marginal_probabilities(self):
        """
        根据边缘 LLR 计算变量节点的边缘概率
        :return: 每个变量节点的边缘概率 (P(0), P(1)) 列表
        """
        marginals = self.marginal_llrs()
        probabilities = []
        for llr in marginals:
            if llr == np.inf:
                p0 = 1.0
            else:
                p0 = np.exp(llr) / (1 + np.exp(llr))
            probabilities.append((p0, 1 - p0))
        return probabilities



    
if __name__ == '__main__':
    
    #s = np.array([0,0])
    #s = syndrome[0]
    
    
    t = 0
    s = np.array([1,1,0,1,0,1])
    x = np.array([0 for i in range(15)])
    M = [0,1,2,3,4,5]
    N = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    E = {(0,0):6.2,(0,2):6.2,
         (1,0):11,(1,1):11,(1,2):11,(1,4):11,
         (2,0):11,(2,1):11,(2,2):11,(2,5):11,
         (3,0):11,(3,2):11,(3,4):11,(3,5):11,
         (4,0):11,(4,3):11,
         (5,0):11,(5,1):11,(5,3):11,(5,4):11,
         (6,0):6.2,(6,1):6.2,(6,3):6.2,(6,5):6.2,
         (7,0):11,(7,3):11,(7,4):11,(7,5):11,
         (8,1):11,(8,4):11,
         (9,1):11,(9,2):11,(9,3):11,(9,4):11,
         (10,1):11,(10,5):11,
         (11,1):6.2,(11,2):6.2,(11,3):6.2,(11,5):6.2,
         (12,2):11,(12,3):11,
         (13,2):11,(13,3):11,(13,4):11,(13,5):11,
         (14,4):11,(14,5):11}
    L = {0:6.2,1:11,2:11,3:11,4:11,5:11,6:6.2,7:11,8:11,9:11,10:11,11:6.2,12:11,13:11,14:11}#L(x|y)
    L1 = {}#L(nm)
    Z = {(0,0):6.2,(0,2):6.2,
         (1,0):11,(1,1):11,(1,2):11,(1,3):11,
         (2,0):11,(2,1):11,(2,2):11,(2,5):11,
         (3,0):11,(3,2):11,(3,4):11,(3,5):11,
         (4,0):11,(4,3):11,
         (5,0):11,(5,1):11,(5,3):11,(5,4):11,
         (6,0):6.2,(6,1):6.2,(6,3):6.2,(6,5):6.2,
         (7,0):11,(7,3):11,(7,4):11,(7,5):11,
         (8,1):11,(8,4):11,
         (9,1):11,(9,2):11,(9,3):11,(9,4):11,
         (10,1):11,(10,5):11,
         (11,1):6.2,(11,2):6.2,(11,3):6.2,(11,5):6.2,
         (12,2):11,(12,3):11,
         (13,2):11,(13,3):11,(13,4):11,(13,5):11,
         (14,4):11,(14,5):11}#Znm
    Z1 = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0}#Zn
# =============================================================================
#     s = np.array([0,1])
#     x = np.array([0 for i in range(3)])
#     M = [0,1]
#     N = [0,1,2]
#     E = {(0,0):np.log(9),(1,0):np.log(9),
#          (1,1):np.log(9),(2,1):np.log(9)}
#     L = {0:np.log(9),1:np.log(9),2:np.log(9)}#L(x|y)
#     L1 = {}#L(nm)
#     Z = {(0,0):np.log(9),(1,0):np.log(9),
#          (1,1):np.log(9),(2,1):np.log(9)}#Znm
#     Z1 = {0:0,1:0,2:0}#Zn
# =============================================================================


    H = np.array([[0 for i in range(len(x))] for j in range(len(s))])
    
    for i in range(len(s)):
        for j in range(len(x)):
            if (j,i) in E.keys():
                H[i][j] = 1
    
    while True:
        t += 1
        for m in M:
            for n in N:
                if (n,m) in E.keys():
                    if s[m] == 0:
                        L1[(n,m)] = Lmn(M, N, m, n, Z, E)
                    else:
                        L1[(n,m)] = -Lmn(M, N, m, n, Z, E)
                
        for n in N:
            for m in M:
                if (n,m) in E.keys():
                    Z[(n,m)] = Znm(n, m, L, L1)
                
        for n in N:
            Z1[n] = Zn(n, L, L1)
        
        for i in range(len(N)):
            if Z1[i] >= 0:
                x[i] = 0
            elif Z1[i] < 0:
                x[i] = 1
        
        
        if np.all(np.dot(H, x)%2 == s%2) or t == T:
# =============================================================================
#             print(t)
#             print(x)
# =============================================================================
            break
        
    # 示例使用
    # =============================================================================
    # num_vars = 15
    # num_checks = 6
    # edges = [(0,0),(0,2),
    #      (1,0),(1,1),(1,2),(1,4),
    #      (2,0),(2,1),(2,2),(2,5),
    #      (3,0),(3,2),(3,4),(3,5),
    #      (4,0),(4,3),
    #      (5,0),(5,1),(5,3),(5,4),
    #      (6,0),(6,1),(6,3),(6,5),
    #      (7,0),(7,3),(7,4),(7,5),
    #      (8,1),(8,4),
    #      (9,1),(9,2),(9,3),(9,4),
    #      (10,1),(10,5),
    #      (11,1),(11,2),(11,3),(11,5),
    #      (12,2),(12,3),
    #      (13,2),(13,3),(13,4),(13,5),
    #      (14,4),(14,5)]
    # prior_llrs = [6.2,11,11,11,11,11,6.2,11,11,11,11,6.2,11,11,11]  # 变量节点的先验 LLR
    # syndrome = np.array([1,1,0,1,0,1])
    # =============================================================================
    num_vars = 2
    num_checks = 3
    edges = [(0,0),(0,1),
         (1,1),(1,2)
         ]
    prior_llrs = [np.log(9), np.log(9)]  # 变量节点的先验 LLR
    syndrome = np.array([0,1,1])

    H = np.array([[0 for i in range(num_vars)] for j in range(num_checks)])

    for i in range(num_checks):
        for j in range(num_vars):
            if (j,i) in edges:
                H[i][j] = 1


    bp = BipartiteGraphBP(num_vars, num_checks, edges, prior_llrs, syndrome, H)
    bp.run()

    # 输出变量节点的边缘概率
    probabilities = bp.marginal_probabilities()
    for i, (p0, p1) in enumerate(probabilities):
        print(f"变量节点 {i} 的边缘概率: P(0)={p0:.20f}, P(1)={p1:.20f}")
    print(probabilities)

    # =============================================================================
    # for (p0,p1) in probabilities:
    #     outcome = np.log(p0/p1)
    #     print(outcome)
    # =============================================================================
