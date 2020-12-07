import numpy as np
def Spktwo2three(S_e,F_e):
    detF=np.linalg.det(F_e)
    # given a 2d Spk stress, first push it to 2d sigma 
    sigma2d=1/(detF) * np.dot(F_e,np.dot(S_e,F_e.transpose()))
    # add zeros in the third dimension of sigma
    sigma3d=np.zeros([3,3])
    sigma3d[0:2,0:2]=sigma2d   # This is sigma E
    # Make 3d F here
    F_3d=np.eye(3)
    F_3d[0:2,0:2]=F_e
    F_3d_inv=np.linalg.inv(F_3d)
    detF3d=np.linalg.det(F_3d)
    # now, push the sigma3d back to Second pk
    S_3d=detF3d*np.dot(F_3d_inv,np.dot(sigma3d,F_3d_inv.transpose()))

    # print("*********************")
    # print("S_2d",S_e)
    # print("F_e_2d",F_e)
    # print("sigma3d",sigma3d)
    # print("S_3d",S_3d)


    return S_3d