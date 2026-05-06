import numpy as np

def compute_homography(M_points, m_points):
    """
    Tính ma trận Homography H (3x3) dùng DLT (Direct Linear Transform).
    M_points: Mảng các điểm 3D (Z=0), shape (N, 2)
    m_points: Mảng các điểm 2D trên ảnh, shape (N, 2)
    """
    assert len(M_points) == len(m_points), "Số lượng điểm 3D và 2D phải bằng nhau"
    N = len(M_points)
    
    A = []
    for i in range(N):
        X, Y = M_points[i][0], M_points[i][1]
        u, v = m_points[i][0], m_points[i][1]
        
        row1 = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
        row2 = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]
        A.append(row1)
        A.append(row2)
        
    A = np.array(A)
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Nghiệm h là hàng cuối của V^T (hoặc cột cuối của V)
    h = Vt[-1]
    
    H = h.reshape((3, 3))
    H = H / H[2, 2] # Chuẩn hóa
    
    return H
