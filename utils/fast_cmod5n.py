import warnings
import numpy as np

# Ignore overflow errors for wind calculations over land
warnings.simplefilter("ignore", RuntimeWarning)


def cmod5n_forward(v, phi, theta):
    """!     ---------
    !     cmod5n_forward(v, phi, theta)
    !         inputs:
    !              v     in [m/s] wind velocity (always >= 0)
    !              phi   in [deg] angle between azimuth and wind direction
    !                    (= D - AZM)
    !              theta in [deg] incidence angle
    !         output:
    !              CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
    !
    !        All inputs must be Numpy arrays of equal sizes
    """
    # Constants
    thetm = 40.0
    thethr = 25.0
    z_pow = 1.6

    # Model coefficients
    C = np.array([
        0,
        -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
        6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 0.0066, 0.3222,
        0.0120, 22.7000, 2.0813, 3.0000, 8.3659, -3.3428, 1.3236, 6.2437,
        2.3893, 0.3249, 4.1590, 1.6930,
    ])
    
    y0 = C[19]
    PN = C[20]
    a = y0 - (y0 - 1) / PN
    b = 1.0 / (PN * (y0 - 1.0) ** (3 - 1))

    # Convert angles to radians and calculate cosines (vectorized)
    fi = np.radians(phi)
    csfi = np.cos(fi)
    cs2_fi = 2.00 * csfi ** 2 - 1.00

    # Precompute scaled incidence angle terms
    x = (theta - thetm) / thethr
    xx = x ** 2

    # B0 calculation
    a0 = C[1] + C[2] * x + C[3] * xx + C[4] * x * xx
    a1 = C[5] + C[6] * x
    a2 = C[7] + C[8] * x
    GAM = C[9] + C[10] * x + C[11] * xx
    s0 = C[12] + C[13] * x
    
    # Calculate sigmoid term with vectorized comparison
    s = a2 * v
    s_vec = np.maximum(s, s0)
    a3 = 1.0 / (1.0 + np.exp(-s_vec))
    
    # Apply correction where s < s0
    mask = s < s0
    if np.any(mask):
        a3[mask] = a3[mask] * (s[mask] / s0[mask]) ** (s0[mask] * (1.0 - a3[mask]))
    
    # Final B0 calculation
    B0 = (a3 ** GAM) * 10.0 ** (a0 + a1 * v)

    # B1 calculation (vectorized)
    B1 = C[15] * v * (0.5 + x - np.tanh(4.0 * (x + C[16] + C[17] * v)))
    B1 = C[14] * (1.0 + x) - B1
    B1 = B1 / (np.exp(0.34 * (v - C[18])) + 1.0)

    # B2 calculation (vectorized)
    V0 = C[21] + C[22] * x + C[23] * xx
    D1 = C[24] + C[25] * x + C[26] * xx
    D2 = C[27] + C[28] * x

    V2 = v / V0 + 1.0
    mask = V2 < y0
    if np.any(mask):
        V2[mask] = a + b * (V2[mask] - 1.0) ** PN
    
    B2 = (-D1 + D2 * V2) * np.exp(-V2)

    # Combine terms
    return B0 * (1.0 + B1 * csfi + B2 * cs2_fi) ** z_pow


def cmod5n_inverse_MonteCarlo(sigma0, phi, theta, v_range=None, dir_range=None):
    """
    Monte Carlo inversion of CMOD5N to get wind speed and direction.
    
    Args:
        sigma0: Normalized Radar Cross Section (linear)
        phi: azimuth angle (degrees)
        theta: incidence angle (degrees)
        v_range: Optional wind speed range to search within (default: 5-70 m/s)
        dir_range: Optional direction range to search within (default: -180 to 180 degrees)
        
    Returns:
        (ws, wdir): Wind speed and direction
    """
    # Define search grids with sensible defaults
    if v_range is None:
        v = np.linspace(5, 70, 14)  # Fewer samples, still covers range
    else:
        v = np.linspace(v_range[0], v_range[1], 14)
        
    if dir_range is None:
        vdir = np.arange(-180, 180, 30)  # Larger step size for first pass
    else:
        vdir = np.linspace(dir_range[0], dir_range[1], 12)
    
    # Initialize with large value
    min_delta = np.inf
    best_ws = None
    best_dir = None
    
    # First pass - coarse grid search
    for speed in v:
        for direction in vdir:
            sigma0_calc = cmod5n_forward(speed, direction-phi, theta)
            delta = np.mean((sigma0 - sigma0_calc) ** 2)
            
            if delta < min_delta:
                min_delta = delta
                best_ws = speed
                best_dir = direction
    
    # Second pass - refine search around best match
    if best_ws is not None:
        # Refined search grid
        v_refined = np.linspace(max(1, best_ws-10), best_ws+10, 10)
        vdir_refined = np.linspace(best_dir-30, best_dir+30, 12)
        
        for speed in v_refined:
            for direction in vdir_refined:
                sigma0_calc = cmod5n_forward(speed, direction-phi, theta)
                delta = np.mean((sigma0 - sigma0_calc) ** 2)
                
                if delta < min_delta:
                    min_delta = delta
                    best_ws = speed
                    best_dir = direction
    
    return best_ws, best_dir


def cmod5n_inverse(sigma0_obs, phi, incidence, iterations=10, initial_guess=10.0):
    """
    Iterative inversion of CMOD5N to retrieve wind speed.
    
    Args:
        sigma0_obs: Observed Normalized Radar Cross Section (linear)
        phi: Angle between azimuth and wind direction (degrees)
        incidence: Incidence angle (degrees)
        iterations: Number of iterations to perform
        initial_guess: Initial wind speed guess (m/s)
        
    Returns:
        Wind speed (m/s)
    """
    # Initialize with the guess
    V = np.full_like(sigma0_obs, initial_guess, dtype=float)
    step = initial_guess / 2.0
    
    # Use binary search approach for faster convergence
    for _ in range(iterations):
        sigma0_calc = cmod5n_forward(V, phi, incidence)
        mask = sigma0_calc > sigma0_obs
        
        # Apply corrections
        V += step
        V[mask] -= 2 * step
        
        # Reduce step size for next iteration
        step /= 2.0
    
    return V


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # Wind direction (degrees)
    wdir = 0
    
    # Incidence angle (degrees)
    theta_i = 40
    
    # Wind speeds (m/s)
    U = np.linspace(2, 20, 100)  # More points for smoother curve
    
    # Calculate normalized radar cross section
    sigma_0 = cmod5n_forward(U, wdir + np.zeros_like(U), theta_i + np.zeros_like(U))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(U, 10 * np.log10(sigma_0))
    plt.grid(True)
    plt.xlabel("Wind speed $U_{10}$ [m/s]")
    plt.ylabel("Normalized radar cross section $\\sigma_0$ [dB]")
    plt.title("CMOD5N Forward Model (θ = 40°, φ = 0°)")
    plt.tight_layout()