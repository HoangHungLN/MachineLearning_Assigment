def viterbi_algorithm(words, states, start_p, trans_p, emit_p):
    """
    words   : list các từ trong câu
    states  : list các nhãn POS
    start_p : dict, xác suất để 1 states bắt đầu trước
    trans_p : dict lồng, xác suất chuyển trạng thái (truyển từ trạng thái x_prev-> x)
    emit_p  : dict lồng, P(word | tag),


    return: (best_tags, best_prob)
        best_tags: list nhãn POS tốt nhất
        best_prob: xác suất của chuỗi nhãn đó
    """
    T = len(words)
    # V[t][tag] = xác suất tốt nhất cho đến vị trí t khi ở tag
    #Khởi tạo V là các dict rỗng
    V = [{} for _ in range(T)]
    # backpointer[t][tag] = tag tốt nhất ở vị trí t-1 (dùng để truy vết ngược)
    #Khởi tạo backpointer là các dict rỗng
    backpointer = [{} for _ in range(T)]

    # --- Bước 1: khởi tạo (t = 0) ---
    first_word = words[0]
    for tag in states:
        # nếu word không có trong emit_p[tag], ta coi xác suất là 0
        emit_prob = emit_p.get(tag, {}).get(first_word, 0.0)
        V[0][tag] = start_p.get(tag, 0.0) * emit_prob
        backpointer[0][tag] = None

    # --- Bước 2: lặp để tìm bảng Viterbi (t = 1->T-1) ---
    for t in range(1, T):
        word = words[t]
        for curr_tag in states:
            best_prob = 0.0
            best_prev_tag = None

            #lấy xác xuất phát xạ P(words|tag)
            emit_prob = emit_p.get(curr_tag, {}).get(word, 0.0)

            #tìm xác suất chuyển trạng thái tốt nhất
            for prev_tag in states:
                prev_prob = V[t-1].get(prev_tag, 0.0)
                trans_prob = trans_p.get(prev_tag, {}).get(curr_tag, 0.0)

                prob = prev_prob * trans_prob * emit_prob

                if prob > best_prob:
                    best_prob = prob
                    best_prev_tag = prev_tag

            #Lưu xác suất tốt nhất từ t đến tag và lưu truy vết của nó
            V[t][curr_tag] = best_prob
            backpointer[t][curr_tag] = best_prev_tag

    # --- Bước 3: kết thúc: chọn tag cuối tốt nhất ---
    best_last_tag = None
    best_last_prob = 0.0
    for tag in states:
        
        if V[T-1].get(tag, 0.0) > best_last_prob:
            best_last_prob = V[T-1][tag]
            best_last_tag = tag

    # --- Bước 4: truy vết ngược để lấy toàn bộ chuỗi nhãn ---
    best_tags = [best_last_tag]
    for t in range(T-1, 0, -1):
        prev_tag = backpointer[t][best_tags[-1]]
        best_tags.append(prev_tag)
    #đảo ngược chuỗi
    best_tags.reverse()

    return best_tags, best_last_prob
