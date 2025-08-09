"""
Python port of the psifun (digamma function) from Fortran funmod2.f90
Faithful translation maintaining identical numerical behavior
"""

import numpy as np
import sys

def psifun(xx):
    """
    EVALUATION OF THE DIGAMMA FUNCTION
    
    PSIFUN(XX) IS ASSIGNED THE VALUE 0 WHEN THE DIGAMMA FUNCTION CANNOT
    BE COMPUTED.
    
    THE MAIN COMPUTATION INVOLVES EVALUATION OF RATIONAL CHEBYSHEV
    APPROXIMATIONS PUBLISHED IN MATH. COMP. 27, 123-127(1973) BY
    CODY, STRECOK AND THACHER.
    
    Originally written at Argonne National Laboratory for the FUNPACK
    package of special function subroutines. Modified by A.H. Morris (NSWC).
    """
    
    # Constants
    dx0 = 1.461632144968362341262659542325721325
    piov4 = 0.785398163397448
    
    # Coefficients for rational approximation of PSIFUN(X) / (X - X0), 0.5 <= X <= 3.0
    p1 = np.array([
        0.895385022981970e-02, 0.477762828042627e+01,
        0.142441585084029e+03, 0.118645200713425e+04,
        0.363351846806499e+04, 0.413810161269013e+04,
        0.130560269827897e+04
    ])
    
    q1 = np.array([
        0.448452573429826e+02, 0.520752771467162e+03,
        0.221000799247830e+04, 0.364127349079381e+04,
        0.190831076596300e+04, 0.691091682714533e-05
    ])
    
    # Coefficients for rational approximation of PSIFUN(X) - LN(X) + 1 / (2*X), X > 3.0
    p2 = np.array([
        -0.212940445131011e+01, -0.701677227766759e+01,
        -0.448616543918019e+01, -0.648157123766197e+00
    ])
    
    q2 = np.array([
        0.322703493791143e+02, 0.892920700481861e+02,
        0.546117738103215e+02, 0.777788548522962e+01
    ])
    
    # Machine dependent constants
    xmax1 = min(sys.maxsize, 1.0 / sys.float_info.epsilon)
    xsmall = 1e-9
    
    x = xx
    aug = 0.0
    
    if x >= 0.5:
        goto_200 = True
    else:
        goto_200 = False
    
    if not goto_200:
        # X < 0.5, USE REFLECTION FORMULA
        # PSIFUN(1-X) = PSI(X) + PI * COTAN(PI*X)
        
        if abs(x) > xsmall:
            goto_100 = True
        else:
            goto_100 = False
            
        if not goto_100:
            if x == 0.0:
                return 0.0  # Error return
            
            # 0 < ABS(X) <= XSMALL. USE 1/X AS A SUBSTITUTE FOR PI*COTAN(PI*X)
            aug = -1.0 / x
            goto_150 = True
        else:
            goto_150 = False
            
        if not goto_150:
            # REDUCTION OF ARGUMENT FOR COTAN
            w = -x
            sgn = piov4
            
            if w <= 0.0:
                w = -w
                sgn = -sgn
                
            # Make an error exit if X <= -XMAX1
            if w >= xmax1:
                return 0.0  # Error return
                
            nq = int(w)
            w = w - nq
            nq = int(w * 4.0)
            w = 4.0 * (w - nq * 0.25)
            
            # W is now related to the fractional part of 4.0 * X
            # Adjust argument to correspond to values in first quadrant and determine sign
            n = nq // 2
            if (n + n) != nq:
                w = 1.0 - w
            z = piov4 * w
            m = n // 2
            if (m + m) != n:
                sgn = -sgn
                
            # Determine final value for -PI*COTAN(PI*X)
            n = (nq + 1) // 2
            m = n // 2
            m = m + m
            
            if m != n:
                # Use SIN/COS as a substitute for TAN
                aug = sgn * ((np.sin(z) / np.cos(z)) * 4.0)
            else:
                # Check for singularity
                if z == 0.0:
                    return 0.0  # Error return
                # Use COS/SIN as a substitute for COTAN
                aug = sgn * ((np.cos(z) / np.sin(z)) * 4.0)
        
        x = 1.0 - x
    
    # Label 200
    if x > 3.0:
        goto_300 = True
    else:
        goto_300 = False
        
    if not goto_300:
        # 0.5 <= X <= 3.0
        den = x
        upper = p1[0] * x
        
        for i in range(5):
            den = (den + q1[i]) * x
            upper = (upper + p1[i+1]) * x
            
        den = (upper + p1[6]) / (den + q1[5])
        xmx0 = x - dx0
        fn_val = den * xmx0 + aug
        return fn_val
    
    # Label 300 - IF X >= XMAX1, PSIFUN = LN(X)
    if x >= xmax1:
        fn_val = aug + np.log(x)
        return fn_val
        
    # 3.0 < X < XMAX1
    w = 1.0 / (x * x)
    den = w
    upper = p2[0] * w
    
    for i in range(3):
        den = (den + q2[i]) * w
        upper = (upper + p2[i+1]) * w
        
    aug = upper / (den + q2[3]) - 0.5 / x + aug
    fn_val = aug + np.log(x)
    return fn_val