from matplotlib import pyplot as plt, rcParams
from numpy import matlib
from scipy import signal
import json
import numpy as np
import warnings

warnings.filterwarnings("ignore") # Пока matplotlib < 3.3, будет рекомендовать
# use_line_collection в stem


def xcov(x, lags):
    mean = np.mean(x)
    return [
        np.sum([
            (x[n] - mean) * (x[n+lag] - mean)
            for n in range(len(x) - np.abs(lag))
        ])
        for lag in lags
    ]


rcParams['lines.linewidth'] = 1
# rcParams['scatter.edgecolors'] = 'r'
rcParams['axes.axisbelow'] = True
rcParams['axes.grid'] = True

print('ЛР №7. Дискретные сигналы')

if bool(input('Загрузить данные из data.json? [1/0]: ')):
    with open('data.json', 'r') as file:
        data = json.load(file)

    for key, value in data.items(): # Задать глобальные переменные из словаря
        globals()[key] = value # Не делайте в серьезных проектах

    B = np.array([B_1, B_2, B_3])
    w = np.array([w_1, w_2, w_3])
    A = np.array([a_1, a_2, a_3])

else:
    Nb = int(input('Nb = '))       # НОМЕР БРИГАДЫ
    N = int(input('N = '))         # ДЛИНА ПОСЛЕДОВАТЕЛЬНОСТИ
    T = float(input('T = '))       # ПЕРИОД ДИСКРЕТИЗАЦИИ
    a = float(input('a = '))       # ОСНОВАНИЕ ДИСКРЕТНОЙ ЭКСПОНЕНТЫ
    C = int(input('C = '))         # АМПЛИТУДА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
    w0 = float(input('w0 = '))     # ЧАСТОТА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
    m = int(input('m = '))         # ВЕЛИЧИНА ЗАДЕРЖКИ
    U = int(input('U = '))         # АМПЛИТУДА ИМПУЛЬСА
    n0 = int(input('n0 = '))       # МОМЕНТ НАЧАЛА ИМПУЛЬСА
    n_imp = int(input('n_imp = ')) # ДЛИТЕЛЬНОСТЬ ИМПУЛЬСА
    B = np.array(input('B = '))    # ВЕКТОР АМПЛИТУД
    w = np.array(input('w = '))    # ВЕКТОР ЧАСТОТ
    A = np.array(input('A = '))    # ВЕКТОР КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ КОМБИНАЦИИ
    Mean = int(input('Mean = '))   # ЗАДАННОЕ МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ШУМА
    Var = int(input('Var = '))     # ЗАДАННАЯ ДИСПЕРСИЯ ШУМА

print('п1. Цифровой единичный импульс')
input('Для вывода ГРАФИКОВ цифрового единичного импульса нажмите <ENTER>')
n = np.arange(N) # Дискретное нормированное время
nT = T * n # Дискретное ненормированное время
u0 = np.concatenate(([1], np.zeros(N-1)))
plt.subplot(1, 2, 1)
plt.gcf().canvas.set_window_title('Digital Unit Impulse') # Заголовок окна
plt.stem(nT, u0, basefmt='')
plt.title('Digital Unit Impulse u0(nT)')
plt.xlabel('nT')
plt.subplot(1, 2, 2)
plt.stem(n, u0, basefmt='')
plt.title('Digital Unit Impulse u0(n)')
plt.xlabel('n')
plt.tight_layout() # Иначе заголовки налезают друг на друга
plt.show()

print('\n----------------------------------------')
print('п2. Цифровой единичный скачок')
input('Для вывода ГРАФИКОВ цифрового единичного скачка нажмите <ENTER>')
u1 = np.ones(N) # FIXME Может время меньше 0 показать?
plt.subplot(1, 2, 1)
plt.gcf().canvas.set_window_title('Digital Unit Step')
plt.stem(nT, u1, basefmt='')
plt.title('Digital Unit Impulse u0(nT)')
plt.xlabel('nT')
plt.subplot(1, 2, 2)
plt.stem(n, u1, basefmt='')
plt.title('Digital Unit Impulse u0(n)')
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п3. Дискретная экспонента')
input('Для вывода ГРАФИКОВ дискретной экспоненты нажмите <ENTER>')
x1 = a ** n
plt.subplot(1, 2, 1)
plt.gcf().canvas.set_window_title('Discrete Exponent')
plt.stem(nT, x1, basefmt='')
plt.title('Discrete exponent x1(nT)')
plt.xlabel('nT')
plt.subplot(1, 2, 2)
plt.stem(n, x1, basefmt='')
plt.title('Discrete exponent x1(n)')
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п4. Дискретный комплексный гармонический сигнал')
print('Для вывода ГРАФИКОВ вещественной и мнимой частей')
input('гармонического сигнала нажмите <ENTER>')
x2 = C * np.exp(1j * w0 * n)
plt.subplot(2, 1, 1)
plt.gcf().canvas.set_window_title('Discrete Harmonic Signal')
plt.title('Discrete Harmonic Signal: REAL [x2(n)]')
plt.stem(n, np.real(x2))
plt.subplot(2, 1, 2)
plt.title('Discrete Harmonic Signal: IMAG [x2(n)]')
plt.stem(n, np.imag(x2))
plt.tight_layout()
plt.show()


print('\n----------------------------------------')
print('п5. Задержанные последовательности')
input('Для вывода ГРАФИКОВ задержанных последовательностей нажмите <ENTER>')
u0_m = np.concatenate([np.zeros(m), u0[0:(N-m)]])
u1_m = np.concatenate([np.zeros(m), u1[0:(N-m)]])
x1_m = np.concatenate([np.zeros(m), x1[0:(N-m)]])
plt.subplot(3, 1, 1)
plt.gcf().canvas.set_window_title('Delayed Discrete Signal')
plt.stem(n, u0_m)
plt.title('Delayed Digital Unit Impulse u0(n-m)')
plt.subplot(3, 1, 2)
plt.stem(n, u1_m)
plt.title('Delayed Digital Unit Step u1(n-m)')
plt.subplot(3, 1, 3)
plt.stem(n, x1_m)
plt.title('Delayed Discrete Exponent x1(n-m)')
plt.tight_layout()
plt.show()


print('\n----------------------------------------')
print('п6. Дискретный прямоугольный импульс')
input('Для вывода ГРАФИКОВ дискретного прямоугольного импульса нажмите <ENTER>')
# В Python нет rectpuls
x3_1 = np.zeros(N)
x3_1[n0:n0+n_imp] = U
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Discrete Rectangular Impulse')
plt.stem(n, x3_1)
plt.title('Discrete Rectangular Impulse x3 1(n)')
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п7. Дискретный треугольный имульс')
input('Для вывода ГРАФИКА дискретного треугольного импульса нажмите <ENTER>')
x4 = signal.convolve(x3_1, x3_1) # Дискретный треугольный импульс
L = len(x4) # Длина сверки
n = np.array(range(L)) # Дискретное нормированное время
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Discrete Triangular Impulse')
plt.title('Discrete Triangular Impulse x4(n)')
plt.stem(n, x4)
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п8. Линейная комбинация дискретных гармонических сигналов')
input('Для вывода ГРАФИКОВ гармонических сигналов и их линейной комбинации нажмите <ENTER>')
n = np.array(range(5*N - 1))
xi = np.multiply(matlib.repmat(B, len(n), 1), np.sin(matlib.asmatrix(n).transpose() * w))
ai = matlib.repmat(A, len(n), 1)
x5 = np.sum(np.multiply(ai, xi), axis=1)
plt.subplot(4, 1, 1)
plt.gcf().canvas.set_window_title('Discrete Harmonic Signals and their Linear Combination')
plt.stem(n, xi[:, 0])
plt.title('First Discrete Harmonic Signal')
plt.subplot(4, 1, 2)
plt.stem(n, xi[:, 1])
plt.title('Second Discrete Harmonic Signal')
plt.subplot(4, 1, 3)
plt.stem(n, xi[:, 2])
plt.title('Third Discrete Harmonic Signal')
plt.subplot(4, 1, 4)
plt.title('Linear Combination x5(n)')
plt.stem(n, x5)
plt.xlabel('n')
plt.tight_layout()
plt.show()

input('Для вывода СРЕДНЕГО ЗНАЧЕНИЯ, ЭНЕРГИИ и СРЕДНЕЙ МОЩНОСТИ сигнала x5 нажмите <ENTER>')
mean_x5 = np.mean(x5)
E = np.sum(np.square(x5))
P = np.sum(np.square(x5)) / len(x5)
print(f"mean_x5 = {mean_x5}, E = {E}, P = {P}")


print('\n----------------------------------------')
print('п9. Дискретный гармонический сигнал с экспоненциальной огибающей')
input('Для вывода ГРАФИКА гармонического сигнала с экспоненциальной огибающей нажмите <ENTER>')
n = np.array(range(N)) # ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
x = C * np.sin(w0 * n) # ДИСКРЕТНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ
x6 = np.multiply(x, np.abs(a)**n)
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Harmonic Signal with Exponential Envelope')
plt.stem(n, x6)
plt.title('Harmonic Signal with Exponential Envelope x6(n)')
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п10. Периодическая последовательность дискретных прямоугольных импульсов')
input('Для вывода ГРАФИКА пяти периодов последовательности нажмите <ENTER>')
xp = np.concatenate([U * u1[0:n_imp], np.zeros(n_imp)]) # ПЕРИОД ПОСЛЕДОВАТЕЛЬНОСТИ
p = 5 # ЧИСЛО ПЕРИОДОВ
x7 = matlib.repmat(xp, 1, p)[0] # ПЕРИОДИЧЕСКАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ
n = np.arange(len(x7)) # ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Periodic Sequence of Rectangular Impulses')
plt.stem(n, x7)
plt.title('Periodic Sequence of Rectangular Impulses x7(n)')
plt.show()


print('\n----------------------------------------')
print('п11. Равномерный белый шум')
input('Для вывода ОЦЕНОК МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ и ДИСПЕРСИИ ШУМА нажмите <ENTER>')
r_uniform = np.random.rand(1, 1000)[0] # РАВНОМЕРНЫЙ БЕЛЫЙ ШУМ
mean_uniform = np.mean(r_uniform) # ОЦЕНКА МАТ. ОЖИДАНИЯ ШУМА
var_uniform = np.var(r_uniform) # ОЦЕНКА ДИСПЕРСИИ ШУМА
print(f'mean_uniform={mean_uniform}, var_uniform={var_uniform}')
input('Для вывода графика АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ нажмите <ENTER>')
m = np.arange(-len(r_uniform), len(r_uniform)) # ВЕКТОР ДИСКРЕТНЫХ СДВИГОВ ДЛЯ АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ
r_r_uniform = xcov(r_uniform, m) # ОЦЕНКА АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ РАВНОМЕРНОГО БЕЛОГО ШУМА
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Autocovariance Function of Uniform White Noise')
plt.stem(m, r_r_uniform, use_line_collection=True)
plt.title('Autocovariance Function of Uniform White Noise')
plt.xlabel('m')
plt.show()


print('\n----------------------------------------')
print('п12. Нормальный белый шум')
input('Для вывода ОЦЕНОК МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ и ДИСПЕРСИИ шума нажмите <ENTER>')
r_norm = np.random.randn(1000) # РАВНОМЕРНЫЙ БЕЛЫЙ ШУМ
mean_norm = np.mean(r_norm) # ОЦЕНКА МАТ. ОЖИДАНИЯ ШУМА
var_norm = np.var(r_norm) # ОЦЕНКА ДИСПЕРСИИ ШУМА
print(f'mean_norm={mean_norm}, var_uniform={var_norm}')
input('Для вывода графика АКФ нажмите <ENTER>')
R_r_norm = np.correlate(r_norm, r_norm, mode='full') / len(r_norm)
m = np.arange(-len(r_norm), len(r_norm)-1)
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('ACF of White Gaussian Noise')
plt.stem(m, R_r_norm, use_line_collection=True)
plt.title('ACF of White Gaussian Noise')
plt.xlabel('m')
plt.show()


print('\n----------------------------------------')
print('п13. Аддитивная смесь дискретного гармонического сигнала с нормальным белым шумом')
input('Для вывода ГРАФИКА аддитивной смеси сигнала с шумом нажмите <ENTER>')
n = np.arange(N) # ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
x8 = x + np.random.randn(N)
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('Mixture of Harmonic Signal')
plt.stem(n, x8)
plt.title('Mixture of Harmonic Signal and White Gaussian Noise x8(n)')
plt.xlabel('n')
plt.show()


print('\n----------------------------------------')
print('п14. АКФ аддитивной смеси дискретного гармонического сигнала с нормальным белым шумом')
input('Для вывода ГРАФИКА АКФ нажмите <ENTER>')
R = (1 / N) * np.correlate(x8, x8, mode='full')
m = np.arange(-(N), N-1)
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title('White Gaussian Noise ACF')
plt.stem(m, R)
plt.title('ACF R(m)')
plt.xlabel('m')
plt.show()

input('Для вывода ДИСПЕРСИИ аддитивной смеси сигнала с шумом и АКФ R(N) нажмите <ENTER>')
print(f"var_x8 = {np.var(x8)}")
print(f"R(N) = {R[N]}")


print('\n----------------------------------------')
print('п15. Нормальный белый шум с заданными статистическими характеристиками')
r_normMean = np.random.randn(1000) + Mean # НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННЫМ МАТЕМАТИЧЕСКИМ ОЖИДАНИЕМ
r_normVar = np.sqrt(Var) * np.random.randn(1000) # НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННОЙ ДИСПЕРСИЕЙ
r_normMeanVar = np.sqrt(Var) * np.random.randn(1000) + Mean
max_ = np.max([r_norm, r_normMean, r_normVar, r_normMeanVar]) # МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ ШУМА СРЕДИ ЧЕТЫРЕХ ЕГО РАЗНОВИДНОСТЕЙ
input('Для вывода ГРАФИКОВ нормального белого шума нажмите <ENTER>')
plt.subplot(4, 1, 1)
plt.gcf().canvas.set_window_title('White Gaussian Noises with different statistics')
plt.plot(r_norm)
plt.title(f"Mean value = {np.mean(r_norm):.4f}, Variance = {np.var(r_norm):.4f}")
plt.ylim((-max_, max_))

plt.subplot(4, 1, 2)
plt.plot(r_normMean)
plt.title(f"Mean value = {np.mean(r_normMean):.4f}, Variance = {np.var(r_normMean):.4f}")
plt.ylim((-max_, max_))

plt.subplot(4, 1, 3)
plt.plot(r_normVar)
plt.title(f"Mean value = {np.mean(r_normVar):.4f}, Variance = {np.var(r_normVar):.4f}")
plt.ylim((-max_, max_))

plt.subplot(4, 1, 4)
plt.plot(r_normMeanVar)
plt.title(f"Mean value = {np.mean(r_normMeanVar):.4f}, Variance = {np.var(r_normMeanVar):.4f}")
plt.ylim((-max_, max_))

plt.tight_layout()
plt.show()
