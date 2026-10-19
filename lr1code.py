import json
from pathlib import Path
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ВВОД ИЗ ФАЙЛА
def load_input():
    path = Path("input.json")
    if not path.exists():
        raise FileNotFoundError("Файл input.json не найден. Создайте его в текущем каталоге.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_matrix(name, M, precision=6):
    print(f"{name}:")
    if M is None:
        print("None")
        return
    arr = np.array(M)
    print(np.array2string(arr, precision=precision, suppress_small=True))

# 1.1 LU
def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("LU-разложение требует квадратную матрицу")
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("Матрица вырождена или близка к вырожденной")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swaps += 1
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U, swaps


def lu_solve(P, L, U, b):
    b = np.array(b, dtype=float)
    n = L.shape[0]
    Pb = P @ b
    y = np.zeros(n)
    for i in range(n):
        s = Pb[i] - L[i, :i] @ y[:i]
        y[i] = s / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i] - U[i, i + 1:] @ x[i + 1:]
        x[i] = s / U[i, i]
    return x


def inverse_from_lu(P, L, U):
    n = L.shape[0]
    inv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        inv[:, j] = lu_solve(P, L, U, e)
    return inv


# 1.2 ПРОГОНКА
def solve_tridiagonal(a, b, c, d):
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("Неверные размеры диагоналей для метода прогонки")
    alpha = np.zeros(n)
    beta = np.zeros(n)
    if abs(b[0]) < 1e-15:
        raise ValueError("Нулевой ведущий элемент прогонки")
    alpha[0] = -c[0] / b[0] if n > 1 else 0.0
    beta[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i - 1] * alpha[i - 1]
        if abs(denom) < 1e-15:
            raise ValueError("Деление на ноль в методе прогонки")
        alpha[i] = -c[i] / denom if i < n - 1 else 0.0
        beta[i] = (d[i] - a[i - 1] * beta[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return alpha, beta, x


# 1.3 ИТЕРАЦИИ
def simple_iteration(A, b, eps, max_iter=10000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Нулевой элемент на диагонали")
    x = np.zeros(n)
    for it in range(1, max_iter + 1):
        x_new = (b - A @ x + diag * x) / diag
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, it
        x = x_new
    return x, max_iter


def seidel(A, b, eps, max_iter=10000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError("Нулевой элемент на диагонали")
            s1 = A[i, :i] @ x[:i] if i > 0 else 0.0
            s2 = A[i, i + 1:] @ x_old[i + 1:] if i < n - 1 else 0.0
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if np.max(np.abs(x - x_old)) < eps:
            return x, it
    return x, max_iter


def norm_inf(M):
    M = np.array(M, dtype=float)
    if M.size == 0:
        return 0.0
    return float(np.max(np.sum(np.abs(M), axis=1)))


def simple_iter_matrix(A): #для якоби
    A = np.array(A, dtype=float)
    n = A.shape[0]
    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Нулевой элемент на диагонали")
    D_inv = np.diag(1.0 / diag)
    return np.eye(n) - D_inv @ A


def seidel_iter_matrix(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    D_plus_L = np.tril(A)
    U = np.triu(A, 1)
    B = np.zeros((n, n))
    for j in range(n):
        rhs = -U[:, j]
        y = np.zeros(n)
        for i in range(n):
            if abs(D_plus_L[i, i]) < 1e-15:
                raise ValueError("Нулевой элемент на диагонали")
            s = rhs[i] - D_plus_L[i, :i] @ y[:i]
            y[i] = s / D_plus_L[i, i]
        B[:, j] = y
    return B


# 1.4 ЯКОБИ
def jacobi_eigen(A, eps, max_iter=1000):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("Метод вращений Якоби требует симметрии")
    V = np.eye(n)
    rotations = 0
    for _ in range(max_iter):
        max_val = 0.0
        p = q = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j
        if max_val < eps:
            break
        a = A[p, p]
        b = A[p, q]
        d = A[q, q]
        if abs(b) < 1e-300:
            A[p, q] = A[q, p] = 0.0
            continue
        tau = (d - a) / (2.0 * b)
        if tau >= 0:
            t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c
        J = np.eye(n)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = -s
        J[q, p] = s
        A = J.T @ A @ J
        V = V @ J
        rotations += 1
    vals = np.diag(A).copy()
    idx = np.argsort(vals)
    vals = vals[idx]
    V = V[:, idx]
    return vals, V, rotations

# 1.5 QR
def qr_householder(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for k in range(min(m - 1, n)):
        x = R[k:, k].copy()
        normx = np.sqrt(np.sum(x * x))
        if normx < 1e-15:
            continue
        if x[0] >= 0:
            alpha = -normx
        else:
            alpha = normx
        v = x.copy()
        v[0] -= alpha
        v_norm2 = np.dot(v, v)
        if v_norm2 < 1e-30:
            continue
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:]) / v_norm2
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v) / v_norm2
    return Q, R


def eig2x2(B):
    a, b = B[0, 0], B[0, 1]
    c, d = B[1, 0], B[1, 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc >= 0:
        sq = np.sqrt(disc)
        return np.array([(tr + sq) / 2.0, (tr - sq) / 2.0], dtype=complex)
    else:
        sq = np.sqrt(-disc)
        return np.array([complex(tr / 2.0, sq / 2.0), complex(tr / 2.0, -sq / 2.0)])

def qr_eigenvalues(A, eps=1e-10, max_iter=5000):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=complex)
    eigvals = np.zeros(n, dtype=complex)
    m = n
    it_total = 0
    while m > 0:
        if m == 1:
            eigvals[m - 1] = A[0, 0]
            m -= 1
            continue
        if m == 2:
            eigvals[0:2] = eig2x2(A[0:2, 0:2])
            m = 0
            continue
        if abs(A[m - 1, m - 2]) < eps * (abs(A[m - 2, m - 2]) + abs(A[m - 1, m - 1]) + 1e-30):
            eigvals[m - 1] = A[m - 1, m - 1]
            m -= 1
            continue
        if m > 2 and abs(A[m - 2, m - 3]) < eps * (abs(A[m - 3, m - 3]) + abs(A[m - 2, m - 2]) + 1e-30):
            eigvals[m - 2:m] = eig2x2(A[m - 2:m, m - 2:m])
            m -= 2
            continue
        B = A[m - 2:m, m - 2:m]
        a, b = B[0, 0], B[0, 1]
        c, d = B[1, 0], B[1, 1]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4.0 * det
        if disc >= 0:
            sq = np.sqrt(disc)
            lam1 = (tr + sq) / 2.0
            lam2 = (tr - sq) / 2.0
            mu = lam1 if abs(lam1 - d) < abs(lam2 - d) else lam2
        else:
            mu = d
        Q, R = qr_householder(A[:m, :m] - mu * np.eye(m))
        A[:m, :m] = R @ Q + mu * np.eye(m)
        it_total += 1
        if it_total > max_iter:
            for i in range(m):
                eigvals[i] = A[i, i]
            break
    idx = np.lexsort((eigvals.imag, eigvals.real))
    return eigvals[idx]


def main():
    data = load_input()
    print("Лабораторная №1")
    # 1.1
    if "lu" in data:
        A = np.array(data["lu"]["A"], dtype=float)
        b = np.array(data["lu"]["b"], dtype=float)
        P, L, U, swaps = lu_decomposition(A)
        print("\n1.1 LU-разложение")
        print_matrix("Исходная матрица A", A)
        print_matrix("Вектор b", b)
        print_matrix("Матрица перестановок P", P)
        print_matrix("Матрица L", L)
        print_matrix("Матрица U", U)
        print_matrix("Результат умножения L*U", L @ U)
        print_matrix("P*A", P @ A)
        print("Проверка L*U = P*A:", np.allclose(L @ U, P @ A))
        x = lu_solve(P, L, U, b)
        print_matrix("Решение системы Ax = b", x)
        invA = inverse_from_lu(P, L, U)
        print_matrix("Обратная матрица A^-1", invA)
        detA = ((-1) ** swaps) * np.prod(np.diag(U))
        print(f"Определитель det(A) = {detA:.10f}")
        print_matrix("Проверка A*A^-1", A @ invA)
        print("A·A^(-1) ≈ E:", np.allclose(A @ invA, np.eye(A.shape[0])))
    else:
        print("\nСекция lu отсутствует в input.json")
    # 1.2
    if "tridiagonal" in data:
        tr = data["tridiagonal"]
        a = np.array(tr["a"], dtype=float)
        b = np.array(tr["b"], dtype=float)
        c = np.array(tr["c"], dtype=float)
        d = np.array(tr["d"], dtype=float)
        alpha, beta, x = solve_tridiagonal(a, b, c, d)
        print("\n1.2 Метод прогонки")
        print_matrix("a (нижняя диагональ)", a)
        print_matrix("b (главная диагональ)", b)
        print_matrix("c (верхняя диагональ)", c)
        print_matrix("d (правая часть)", d)
        print_matrix("Прогоночные коэффициенты alpha", alpha)
        print_matrix("Прогоночные коэффициенты beta", beta)
        print_matrix("Итоговое решение системы x", x)
    else:
        print("\nСекция tridiagonal отсутствует в input.json")
    # 1.3
    if "iterative" in data:
        itd = data["iterative"]
        A = np.array(itd["A"], dtype=float)
        b = np.array(itd["b"], dtype=float)
        eps = float(itd.get("epsilon", 1e-6))
        print("\n1.3 Метод простых итераций и метод Зейделя")
        print(f"Заданная точность ε = {eps}")
        exact = None
        try:
            P, L, U, _ = lu_decomposition(A)
            exact = lu_solve(P, L, U, b)
            print_matrix("Эталонное решение (через LU)", exact)
        except Exception as e:
            print("Не удалось найти эталонное решение через LU:", e)
        if "exact" in itd:
            print_matrix("Эталонное решение из файла", np.array(itd["exact"], dtype=float))
        try:
            B_si = simple_iter_matrix(A)
            n_si = norm_inf(B_si)
            print(f"Норма матрицы перехода ПИ ||B||беск = {n_si:.6f}")
            if n_si < 1:
                print("  ||B||беск < 1 => достаточное условие сходимости выполнено")
            else:
                print("  ||B||беск >= 1 => достаточное условие не гарантирует сходимость")
        except Exception as e:
            print("Не удалось вычислить норму для простых итераций:", e)
        try:
            B_se = seidel_iter_matrix(A)
            n_se = norm_inf(B_se)
            print(f"Норма матрицы перехода Зейделя ||B||беск = {n_se:.6f}")
            if n_se < 1:
                print("  ||B||беск < 1 => достаточное условие сходимости выполнено")
            else:
                print("  ||B||беск >= 1 => достаточное условие не гарантирует сходимость")
        except Exception as e:
            print("Не удалось вычислить норму для Зейделя:", e)
        x_si, it_si = simple_iteration(A, b, eps)
        x_se, it_se = seidel(A, b, eps)
        print_matrix("Решение методом простых итераций", x_si)
        print(f"Количество итераций (ПИ): {it_si}")
        print_matrix("Решение методом Зейделя", x_se)
        print(f"Количество итераций (Зейдель): {it_se}")
        if exact is not None:
            print(f"Погрешность ПИ: {np.max(np.abs(x_si - exact)):.3e}")
            print(f"Погрешность Зейделя: {np.max(np.abs(x_se - exact)):.3e}")
        print("Сравнение методов по скорости сходимости:")
        if it_si < it_se:
            print(f"  ПИ быстрее: {it_si} < {it_se}")
        elif it_se < it_si:
            print(f"  Зейдель быстрее: {it_se} < {it_si}")
        else:
            print("  Одинаковое число итераций")
    else:
        print("\nСекции iterative нет в input.json")
    # 1.4
    if "jacobi" in data:
        jd = data["jacobi"]
        A = np.array(jd["A"], dtype=float)
        eps = float(jd.get("epsilon", 1e-6))
        print("\n1.4 Метод вращений Якоби")
        print(f"Заданная точность ε = {eps}")
        print_matrix("Исходная матрица A", A)
        vals, V, rotations = jacobi_eigen(A, eps)
        Lambda = np.diag(vals)
        print_matrix("Найденные СЗн", vals)
        print_matrix("Матрица СВ V", V)
        print_matrix("Λ (диагональная матрица СЗн)", Lambda)
        print_matrix("A*V", A @ V)
        print_matrix("V*Lambda", V @ Lambda)
        print("Проверка A*V = V*Lamda:", np.allclose(A @ V, V @ Lambda))
        print(f"Количество вращений: {rotations}")
    else:
        print("\nСекция jacobi отсутствует в input.json")
    # 1.5
    if "qr" in data:
        qd = data["qr"]
        A = np.array(qd["A"], dtype=float)
        print("\n1.5 QR")
        print_matrix("Исх. матрица A", A)
        Q, R = qr_householder(A)
        print_matrix("Матрица Q", Q)
        print_matrix("Матрица R", R)
        print_matrix("Q*R", Q @ R)
        print("Проверка Q*R = A:", np.allclose(Q @ R, A))
        vals = qr_eigenvalues(A)
        print_matrix("Найденные СЗн (QR)", vals)
        if "complex_example" in qd:
            C = np.array(qd["complex_example"], dtype=float)
            print("\nПример матрицы с комплексно сопряжёнными СЗн")
            print_matrix("Матрица C", C)
            eig_np = np.linalg.eigvals(C)
            print_matrix("Собственные значения", eig_np)
            found = False
            for i in range(len(eig_np)):
                for j in range(i + 1, len(eig_np)):
                    if abs(eig_np[i] - np.conj(eig_np[j])) < 1e-8:
                        print("Найдена комплексно сопряжённая пара:", eig_np[i], "и", eig_np[j])
                        found = True
            if not found:
                print("Комплексно сопряжённая пара не обнаружена.")
    else:
        print("\nСекции qr нет в input.json")
    print("\n=====")

if __name__ == "__main__":
    main()
