import numpy as np

def create_v_ij(H, i, j):
    # i, j là chỉ số cột của H (0-indexed)
    hi = H[:, i]
    hj = H[:, j]
    v = np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2]
    ])
    return v

def compute_intrinsics(B_vec):
    B11, B12, B22, B13, B23, B33 = B_vec
    
    # Ma trận B theo lý thuyết phải là ma trận xác định dương, nên B11 phải > 0.
    # Nếu giải ra b bị ngược dấu, ta đảo dấu toàn bộ.
    if B11 < 0:
        B11, B12, B22, B13, B23, B33 = -B11, -B12, -B22, -B13, -B23, -B33
        
    # Rút trích phân tích đại số từ B
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(np.abs(lamda / B11))
    beta = np.sqrt(np.abs(lamda * B11 / (B11*B22 - B12**2)))
    gamma = -B12 * alpha**2 * beta / lamda
    u0 = gamma * v0 / beta - B13 * alpha**2 / lamda
    
    A = np.array([
        [alpha, gamma, u0],
        [    0,  beta, v0],
        [    0,     0,  1]
    ])
    return A

def zhang_init(H_list):
    # 1. Tìm ma trận B
    V = []
    for H in H_list:
        v12 = create_v_ij(H, 0, 1)
        v11 = create_v_ij(H, 0, 0)
        v22 = create_v_ij(H, 1, 1)
        
        V.append(v12)
        V.append(v11 - v22)
        
    V = np.array(V)
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1] # Vector 6 ẩn của B
    
    # 2. Tìm ma trận nội A
    A = compute_intrinsics(b)
    
    # 3. Tìm tham số ngoại R, t cho mỗi ảnh
    R_list = []
    t_list = []
    A_inv = np.linalg.inv(A)
    
    for H in H_list:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        
        lamda_R = 1.0 / np.linalg.norm(A_inv @ h1)
        r1 = lamda_R * (A_inv @ h1)
        r2 = lamda_R * (A_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lamda_R * (A_inv @ h3)
        
        # Dùng SVD ép R thành ma trận trực giao chuẩn
        R_approx = np.column_stack([r1, r2, r3])
        U_r, S_r, Vt_r = np.linalg.svd(R_approx)
        R_ortho = U_r @ Vt_r
        
        R_list.append(R_ortho)
        t_list.append(t)
        
    return A, R_list, t_list
