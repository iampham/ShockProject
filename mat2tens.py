import numpy as np
def mat2tens(cij_mat):
    """Convert from Voigt to full tensor notation 
       Convert from the 6*6 elastic constants matrix to 
       the 3*3*3*3 tensor representation. Recoded from 
       the Fortran implementation in DRex. Use the optional 
       argument "compl" for the elastic compliance (not 
       stiffness) tensor to deal with the multiplication 
       of elements needed to keep the Voigt and full 
       notation consistant.
    """
    cij_tens = np.zeros((3,3,3,3))
    m2t = np.array([[0,5,4],[5,1,3],[4,3,2]])
    # if compl:
    #     cij_mat = cij_mat / np.array([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    #                                   [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    #                                   [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    #                                   [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
    #                                   [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
    #                                   [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    cij_tens[i,j,k,l] = cij_mat[m2t[i,j],m2t[k,l]]
    return cij_tens