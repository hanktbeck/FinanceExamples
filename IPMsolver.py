#----------------------------------------------------------------------
# Convex Optimization via IPM
''' author: Hank Beck '''
# date: 4/01/2024
# email: hanktbeck@gmail.com
#----------------------------------------------------------------------
# DESCRIPTION
#----------------------------------------------------------------------
'''
Implementation of the interior point method for solving convex optimization problems.
First focusing on the solving of quadratic programming problem (mean Variance),
and working towards solving second order cone programming and semi-definite programming problems.
'''
#======================================================================
# Revision Log
#
# Rev   Date          Author      Description    
#======================================================================
'''

# (1)   4/01/2024    HTB         (+) Initial Set up   
'''
#----------------------------------------------------------------------
# MODULES
#----------------------------------------------------------------------
#from numpy import linalg as LA
import numpy as np
from numpy.linalg import norm
from numpy.linalg import solve
# Used to initialize our line search
from scipy.optimize import linprog
import scipy
# ----------------------------------------------------------------------
#   Problem Class
# ----------------------------------------------------------------------
class QuadProg:
    # Base class for Quadratic Programming. Not efficiently set up yet as we have only focused on Mean Variance.
    def __init__(self,x0, Q, c ):
        self.x0 = x0
        self.x = x0
        self.Q = Q
        self.c = c
        # self.A = A
        # self.b = b
        # self.y = y
        # self.s = s
        # self.mu = 1
        # self.kappa = 1
    
    #----------------------------------------------------------------------
    # Calculus
    #----------------------------------------------------------------------
        
    def objfunc(self):
        return 0.5 * ( (self.x.T) @ (self.Q) @ (self.x) )+ ((self.c.T) @ (self.x))
    
    def Phase1_basic(self):
        # Center in the feasible region for constraints Ax = b and Dx >= d
        # min_s
        # s.t. Dx 
        return NotImplemented
        
    def Phase1_SumOfInfeas(self):
        return NotImplemented

    
    #----------------------------------------------------------------------
    # update values
    #----------------------------------------------------------------------
    
    def update_x(self, x):
        self.x = x
    
class MeanVarIPM(QuadProg):

    def __init__(self,x0: np.ndarray, Q: np.ndarray, c: np.ndarray,
                       A: np.ndarray, b: np.ndarray,
                       D: np.ndarray, d: np.ndarray,
                       y0: np.ndarray, z0: np.ndarray, s0: np.ndarray, psi = 1, gamma = 1,
                                                                                NoShorting  = True,
                                                                                FullyInvest = True,
                                                                                NoLeverage  = True):
        super().__init__(x0, Q , c)
    
        # Objective components
        self.gamma = gamma
        self.V = self.Q
        self.mu = self.c
        
        # Equality constraints
        self.A = A
        self.b = b
        
        # Inequality constraints
        self.D = D
        self.d = d
        self.z0 = z0
        self.s0 = s0
        #print('D:',D)
        #print('d:',d)
        
        if FullyInvest:
            # 1^T x >= 1
            # Normally 1^T x = 1, but we can relax this to 1^T x >= 1 and pair it with
            self.D = np.concatenate((self.D, np.ones((1,self.x.shape[0]))), axis = 0)
            self.d = np.concatenate((self.d, np.array([[1]])))
            self.z0 = np.concatenate((self.z0, np.array([[1]])))
            self.s0 = np.concatenate((self.s0, np.array([[1]])))
        
        #print('D:',self.D)
        #print('d:',self.d)
        
        if NoLeverage:
            # -1^T x >= -1
            self.D = np.concatenate((self.D, -np.ones((1,self.x.shape[0]))), axis = 0)
            self.d = np.concatenate((self.d, np.array([[-1]])))
            self.z0 = np.concatenate((self.z0, np.array([[1]])))
            self.s0 = np.concatenate((self.s0, np.array([[1]])))

        #print('D:',self.D)
        #print('d:',self.d)

        if NoShorting:
            # Ix >= 0
            
            if np.any(self.x0 < 0):
                raise ValueError('Short Positions Prohibited. \n \
                                  All elements of x0 should be non-negative or No Shorting Set to false.\n \
                                  If a feasible start is not known use Phase1 method to find a feasible start.')
            
            self.D = np.concatenate((self.D, np.eye(self.x.shape[0])), axis = 0)
            self.d = np.concatenate((self.d, np.zeros(self.x.shape)))
            self.z0 = np.concatenate((self.z0, np.ones(self.x.shape)))
            self.s0 = np.concatenate((self.s0, np.ones(self.x.shape)))
        
        # Lagrange multipliers
        self.y0 = y0
        self.y = y0
        #self.z0 = z0
        self.z = self.z0
        self.s = self.s0
        #self.s0 = s0
        
        # pertubation
        self.psi = psi
        
        # path histor
        self.x_path   = None
        self.obj_path = None
        self.err_path = None

        #check that the dimensions are correct
        if np.any(self.mu.shape != self.x.shape):
            raise ValueError('mu and x have different dimensions')
        if np.any(self.y.shape != self.b.shape):
            raise ValueError('y and b have different dimensions')
        if np.any(self.s.shape != self.d.shape):
            raise ValueError('s and d have different dimensions')
        if np.any(self.z.shape != self.d.shape):
            print('z:',self.z.shape)
            print('d:',self.d.shape)
            raise ValueError('z and d have different dimensions')

    def objfunc(self):
            return 0.5 * self.gamma * ( (self.x.T) @ (self.V) @ (self.x) )+ ((self.mu.T) @ (self.x))

    def step_split(self, step):
        n = self.x.shape[0]
        m = self.b.shape[0]
        p = self.d.shape[0]
                    
        step_x = step[:n]
        step_y = step[n:(n+m)]
        step_z = step[(n+m):(n+m+p)]
        step_s = step[(n+m+p):]
        return step_x, step_y, step_z, step_s
    # ----------------------------------------------------------------------
    # Calculus
    # ----------------------------------------------------------------------    
    def gradIPM(self, CustomStep = False,
                        custom_step = None):
        # Using notation from report
        
        x = self.x
        y = self.y
        z = self.z
        s = self.s

        if CustomStep:
            cust_x, cust_y, cust_z, cust_s = self.step_split(custom_step)
            x = cust_x
            y = cust_y
            z = cust_z
            s = cust_s
        
        grad_x = ((self.gamma*self.V @ x) - (self.mu) + (self.A.T @ y) - (self.D.T @ z))
        grad_y = (self.A @ x - self.b)
        grad_z = (s+self.d-(self.D@x))
        grad_s = ((np.diag(s.flatten())@np.diag(z.flatten()))@np.ones(shape = self.d.shape)) \
                 -(self.psi*np.ones(self.d.shape))
        
        return grad_x, grad_y, grad_z, grad_s, np.concatenate((grad_x,grad_y,grad_z,grad_s))
    
    def hessIPM(self):
        n = self.x.shape[0]
        m = self.b.shape[0]
        p = self.d.shape[0] 
        #print('D.T:',self.D.T.shape)
        row1 = np.concatenate((self.gamma*self.V, self.A.T,        (-1*self.D.T),   np.zeros((n,p))), axis = 1)
        row2 = np.concatenate((self.A,            np.zeros((m,m)), np.zeros((m,p)), np.zeros((m,p))), axis = 1)
        row3 = np.concatenate((-1*self.D,         np.zeros((p,m)), np.zeros((p,p)), np.eye(p)),   axis = 1)
        row4 = np.concatenate((np.zeros((p,n)),   np.zeros((p,m)), np.diag(self.s.flatten()), np.diag(self.z.flatten())), axis = 1)
        return np.concatenate((row1,row2,row3,row4), axis = 0)
    
    def PertubationUpdate(self, dual, slack):
        '''
        input
        -----
        dual : array_like
            Lagrange multiplier for the inequality constraints.
        slack: array_like
            slack variable for the inequality constraints.
        '''
        if dual.shape != slack.shape:
            print('dual and slack have different dimensions')
            return None

        pertubation = (dual.T @ slack )/ slack.shape[0]

        return 0.1 * pertubation
    
    #----------------------------------------------------------------------
    # update values
    #----------------------------------------------------------------------
    def _update_psi(self):
        #update = self.PertubationUpdate(self.x, self.s)
        self.psi = self.PertubationUpdate(self.z, self.s)

    #----------------------------------------------------------------------
    # Solver
    #----------------------------------------------------------------------
    def IPM_QuadProg(self, tol=1e-6, max_iter=10000,
                                                  LineSearch = True):
        #----------------------------------------------------------------------
        # Algorithm 2.3: Interior Point Method for a quadratic Programming
        # -from "Optimization methods in Finance" by Gerard Cornuejols, Javier Pena and Reha Tutuncu
        #----------------------------------------------------------------------
        # 1. choose x0 in the interior of the feasible set, and s0 > 0
        # For k = 0,1,2,... do
        #---- Solve the Newton system FOR (x,y,z) for (x^k,y^k,z^k) and mu:= 0.1 mu(x^k,s^k)
        #---- Choose step size alpha_k \in (0,1] by line search
        #---- Set (x^(k+1),y^(k+1),z^(k+1)) := (x^k,y^k,z^k) + alpha_k (dx,dy,dz)
        # End For

        """
        tol : float, optional
            Gradient tolerance for terminating the solver.
        max_iter : int, optional
            Maximum number of iteration for terminating the solver.

        output
        ------
        x : array_like
            Final solution
        obj_his : array_like
            Objective function value convergence history
        err_his : array_like
            Norm of gradient convergence history
        exit_flag : int
            0, norm of gradient below `tol`
            1, exceed maximum number of iteration
            2, others
        """
        
        if np.any(self.s0 < 0):
            #print('s0 is not feasible')
            raise ValueError('s0 must be non-negative')
        if np.any(self.z0 < 0):
            raise ValueError('z0 must be non-negative')
        '''
        n = self.x.shape[0]
        m = self.b.shape[0]
        p = self.d.shape[0] 
        '''
        # initial step
        self._update_psi()
        self.x = np.copy(self.x0)
        g = self.gradIPM()[4]
        H = self.hessIPM()
        #alpha = 0.9
        #
        obj = self.objfunc()
        err = norm(g)
        #
        obj_his = np.zeros(max_iter + 1)
        err_his = np.zeros(max_iter + 1)
        x_his = np.zeros((max_iter + 1, self.x.shape[0]))
        #
        obj_his[0] = obj
        err_his[0] = err
        x_his[0]   = self.x.flatten()

        #try:
        #    get_ipython()
        #    Ipy = True
        #except:
        #    Ipy = False

        # start iteration
        iter_count = 0
        while err >= tol:
            
            #if Ipy:
            #    %clear
            #    print('iter:',iter_count)
            
            # Newton's step
            step = solve(H, -g)
            #step = solve(H, g)
            step_x, step_y, step_z, step_s = self.step_split(step)
            '''
            step_x = step[:n]
            step_y = step[n:(n+m)]
            step_z = step[(n+m):(n+m+p)]
            step_s = step[(n+m+p):]
            '''
            # line search
            alpha, line_search_flag = self.CPTlineSearch(x_k = np.concatenate((self.x,
                                                                               self.y,
                                                                               self.z,
                                                                               self.s)),
                                                                               del_vec= step, shrinkRate = 0.5)
            
            if line_search_flag == 1:
                print('Line search failed')
                return 2
            
            # update the path
            self.x = self.x + alpha*step_x
            self.y = self.y + alpha*step_y
            self.s = self.s + alpha*step_s
            self.z = self.z + alpha*step_z
            
            # update pertubations for new location
            self._update_psi()

            # update function, gradient and Hessian
            g = self.gradIPM()[4]
            H = self.hessIPM()
            obj = self.objfunc()
            err = norm(g)
            #
            iter_count += 1
            obj_his[iter_count] = obj
            err_his[iter_count] = err
            x_his[iter_count] = self.x.flatten()
            
            # check if exceed maximum number of iteration
            if iter_count >= max_iter:
                #print('IPM reach maximum number of iteration.')
                self.obj_path = obj_his[:iter_count+1]
                self.err_path = err_his[:iter_count+1]
                self.x_path = x_his[:iter_count+1]
                return  1
        #
        #print('IPM complete')
        self.obj_path = obj_his[:iter_count+1]
        self.err_path = err_his[:iter_count+1]
        self.x_path = x_his[:iter_count+1]
        return 0

    def CPTlineSearch(self, x_k, del_vec, shrinkRate = 0.5,tol = 1e-15, stop_tol =1e-9, max_iter = 1e6):
        #Line Search Function
        #Modified Armijo line search
        # -from "Optimization methods in Finance" by Gerard Cornuejols, Javier Pena and Reha Tutuncu
        """
        input
        -----
        self : Quadratic Program Object
            Quadratic Program being optimized
        x_k : array_like
        	Base point (Previous step).
        del_vec : array_like
	    	Step along the central path
	    shrinkRate : float, optional
	    	step size shrink ratio

	    output
	    ------
	    alpha_curr : float or None
	    	When sucess return the step size, otherwise return None.
        exit_flag : int
            0, norm of gradient below `tol`
            1, exceed maximum number of iteration
	    """
        '''
        # dimensions
        n = self.x.shape[0]
        m = self.b.shape[0]
        p = self.d.shape[0] 
        '''
        # Function which returns the next destination along the path for a given step size alpha
        stepped_vec = lambda alpha: x_k + alpha*del_vec

	    # determine initial step size
        ## alpha_init := max{alpha: x_k + alpha*del_vec >= 0}
        #print(x_k)
        #print(del_vec)
        alpha_max = linprog(c=-1,A_ub=(-1*del_vec), b_ub= x_k.flatten(), bounds=(0,1), method='highs')


        #cons = [{'type': 'ineq',
        #         'fun': stepped_vec}]
        #alpha_max = minimize_scalar(fun = lambda alpha: -alpha, bounds=(0,1),
        #                                                        constraints = cons)
        #
        if not alpha_max.success:
            raise ValueError('Line search: no feasible step size.')
        #

        # Check if step is not orthogonal to the gradient
        
        #direction_test = self.gradIPM()[4].T@(del_vec)
        #if direction_test < 0:
        #    print('Line search: not a descent direction.')
        #    return None, 1
        
        #

        # Initialize step size comparison criteria
        alpha_curr = 0.99*alpha_max.x
        res_norm_prev = norm(self.gradIPM()[4])
        #cust_x, cust_y, cust_z, cust_s = self.step_split(stepped_vec(alpha_curr))
        #res_norm_step = norm(self.gradIPM(CustomStep=True,
        #                                        custom_x = cust_x,
        #                                        custom_y = cust_y,
        #                                        custom_z = cust_z,
        #                                        custom_s = cust_s)[4])
        #
        res_norm_step = norm(self.gradIPM(CustomStep=True,custom_step = stepped_vec(alpha_curr))[4])
        stop_cond = ((1 - (alpha_curr*0.01))*res_norm_prev)
        #
        i = 0 # iteration counter
        # Line search
        while res_norm_step >= (stop_cond+stop_tol):
            alpha_curr *= shrinkRate
            i += 1
            
            #if res_norm_step <= stop_cond + stop_tol:
            #    break

            if stop_cond != ((1 - (alpha_curr*0.01))*res_norm_prev):
                if alpha_curr < tol:
                    print(f'Line search: step size below tolerance ({tol}).')
                    return None, 1
                
            if alpha_curr < tol*100:
                print('res_norm_step:',res_norm_step)
                print('stop_cond:',stop_cond)
                print('alpha_curr:',alpha_curr)
                print('i:',i)

            
            
            if i > max_iter:
                print(f'Line search: max num ({max_iter}) iterations reached.')
                return None, 1
            
            res_norm_step = norm(self.gradIPM(CustomStep=True,custom_step = stepped_vec(alpha_curr))[4])
            stop_cond = ((1 - (alpha_curr*0.01))*res_norm_prev)
            #cust_x, cust_y, cust_z, cust_s = self.step_split(stepped_vec(alpha_curr))
            #res_norm_step = norm(self.gradIPM(CustomStep=True,
            #                                        custom_x = cust_x,
            #                                        custom_y = cust_y,
            #                                        custom_z = cust_z,
            #                                        custom_s = cust_s)[4])
        #
        return alpha_curr, 0
    
    def scipy_linesearch(self):
        scipy.optimize.line_search()
        return NotImplemented
    def get_feasible_start(self):
        # generates a feasible start
        return NotImplemented
# ----------------------------------------------------------------------
# ====================================================================== 
# Utility
#----------------------------------------------------------------------
def largestWeightIdxs(QP, n = 3):
    indxr = np.copy(QP.x.flatten())
    largestWeights = np.argsort(indxr)[-n:]
    return largestWeights

# Graphics --------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot3DPath(QP):
    if QP.x.shape[0] < 3:
        print("Use 2D or 1D")
        return
    #indxr = np.copy(QP.x.flatten())
    largestWeights = largestWeightIdxs(QP)
    x1 = QP.x_path[:,largestWeights[0]]
    x2 = QP.x_path[:,largestWeights[1]]
    x3 = QP.x_path[:,largestWeights[2]]
    # Create a 3D plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot(x1, x2, x3, label = 'Path of Portfolio weights', color = 'b', marker = 'o')
    #Highlight the path start in red
    ax.scatter(x1[0], x2[0],x3[0], color='red', label='Feasible Start Point')

    # Highlight the last point (index -1) in green
    ax.scatter(x1[-1], x2[-1],x3[-1], color='green', label='Optimized Point')
    # Set range for each axis
    ax.set_xlim([np.min(x1)-0.01, np.max(x1)+0.01])  # Set range for X axis
    ax.set_ylim([np.min(x2)-0.01, np.max(x2)+0.01])  # Set range for Y axis
    ax.set_zlim([np.min(x3)-0.01, np.max(x3)+0.01])  # Set range for Z axis
    
    # Set labels and title
    ax.set_xlabel('Asset {}'.format(largestWeights[0]))
    ax.set_ylabel('Asset {}'.format(largestWeights[1]))
    ax.set_zlabel('Asset {}'.format(largestWeights[2]))
    ax.set_title('Path of 3 largest Portfolio weights')
    # Add a subtitle
    #ax.text(0, -4,0, 'Asset {}: {}'.format(largestWeights[0],indxr[largestWeights[0]]), fontsize=12, ha='center')
    ax.set_box_aspect(aspect = None, zoom = 0.9)
    ax.legend()
    # Show the plot
    plt.show()
    return
# Fin