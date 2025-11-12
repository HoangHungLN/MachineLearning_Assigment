import numpy as np

def forward_algorithm(O, pi, A, B):
    """
    Tính toán xác suất của một chuỗi quan sát O bằng Giải thuật Forward.

    Arguments:
    O: numpy.array (T,)
        Chuỗi quan sát (dưới dạng chỉ số, vd: [0, 1, 2])
    pi: numpy.array (N,)
        Vector xác suất trạng thái ban đầu.
    A: numpy.array (N, N)
        Ma trận xác suất chuyển trạng thái. A[i, j]: là xác suất chuyển từ trạng thái i sang trạng thái j.
    B: numpy.array (N, M)
        Ma trận xác suất sinh ra các quan sát. B[i, k]: là xác suất sinh ra quan sát k từ trạng thái i.

    Returns:
    float:
        Tổng xác suất của chuỗi quan sát O, P(O|lambda).

    """
    T = len(O)  # Độ dài chuỗi quan sát
    N = A.shape[0]  # Số lượng trạng thái của mô hình
    
    # Tạo bảng alpha
    alpha = np.zeros((T, N))

    # 1. Khởi tạo alpha[0, i]
        alpha[0, :] = pi * B[:, O[0]]

    # 2. Đệ quy (t=1 đến T-1)
    for t in range(1, T):
            alpha[t, :] = (alpha[t-1, :] @ A) * B[:, O[t]]
        
    # 3. P(O|lambda) = Tổng các alpha[T-1, i]
    total_prob = np.sum(alpha[T-1, :])

    return total_prob
