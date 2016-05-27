
# schechter parameters obtained from Baldry+2011
def schechter_function(M,M_star=10**(10.66),phi1=3.96e-3,alpha1=-0.35,phi2=0.79e-3,alpha2=-1.47):
    return (np.exp(-M/M_star))*((phi1*(M/M_star)**alpha1) + (phi2*(M/M_star)**alpha2))*(1/M_star)