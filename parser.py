import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import seaborn as sns


class Parser:
    def __init__(self, debug=False):
        self.debug = debug
        self.waveguide_mse_errors = []
        self.waveguide_max_errors = []
        self.freespace_mse_errors = []
        self.freespace_max_errors = []

    def parse(self, fs_dirname, wg_dirname, result_filename):
        if not (os.path.exists(fs_dirname) and os.path.exists(wg_dirname)):
            raise FileNotFoundError(f"Директория {fs_dirname} или {wg_dirname} не существует")
        for (_, dir_names, filenames) in os.walk(wg_dirname):
            for el in dir_names:
                for (_, dir_names1, filenames1) in os.walk(os.path.join(wg_dirname, el)):
                    for ell in dir_names1:
                        for (_, dir_names2, filenames2) in os.walk(os.path.join(wg_dirname, el, ell)):
                            for filename in filenames2:
                                fs_path = glob.glob(fs_dirname + '/' + el + '/' + ell + '/' + filename.replace("ph_", "").split("L=")[0] + '*')[0]  # Восстанавливаем соответствующее имя файла для свободного пространства
                                wg_path = os.path.join(wg_dirname, el, ell, filename)

                                if not (os.path.isfile(fs_path) and os.path.isfile(wg_path)):
                                    print(f"L отличается")
                                    raise Exception('Нарушена структура датасета')
                                # >>> стандартный вид файла: patch_f=1_w=0.8_A=1.1_L=94.9758.txt - для fs <<<
                                # >>> стандартный вид файла: ph_patch_f=1_w=0.8_A=1.1_L=18.9968.txt - для wg <<<
                                base_name = os.path.splitext(filename)[0]  # Убираем .txt
                                parts = base_name.split('_')
                                params = f"{parts[2]}GHz {el} {ell} {parts[3]} {parts[4]} {parts[5]}"

                                try:
                                    coefs2, freespace_mse_error, freespace_max_error = self.approximate(self.parse_fs(fs_path))
                                    coefs1, waveguide_mse_error, waveguide_max_error = self.approximate(self.parse_waveguide(wg_path))

                                    self.freespace_mse_errors.append(freespace_mse_error)
                                    self.freespace_max_errors.append(freespace_max_error)
                                    self.waveguide_mse_errors.append(waveguide_mse_error)
                                    self.waveguide_max_errors.append(waveguide_max_error)

                                    if self.debug:  # or freespace_mse_error > 1 or waveguide_mse_error > 1 or freespace_max_error > 5 or waveguide_max_error > 5:
                                        # plt.figure().set_fullscreen(True)
                                        plt.xlabel('f, GHz', fontsize=40)
                                        plt.ylabel('ph, deg', fontsize=40)
                                        plt.legend()
                                        plt.xticks(size=40)
                                        plt.yticks(size=40)
                                        plt.grid(True)
                                        plt.show()

                                    if freespace_mse_error < 1 and waveguide_mse_error < 1:
                                        with open(f"ph_post_processed_data_new2_{parts[2]}GHz.txt", mode="a") as f:
                                            f.write(
                                                f"{params} {' '.join(map(str, coefs1))} {' '.join(map(str, coefs2))}\n")

                                except Exception as e:
                                    print(f"!!!{fs_path}")

        self._plot_approximation_error_distribution()
        return

    def _parse_free_space(self, filename):
        return self._parse_file(filename, skip_lines=2, step=1)

    def _parse_file(self, filename, skip_lines, step, last_is_first=False):
        al = []
        shift = 0
        prev_x = None
        prev_y = None

        with open(filename, 'r') as f:
            lines = f.readlines()
            data_lines = lines[skip_lines:]
            if step > 1:
                data_lines = data_lines[::step]

            for line in data_lines:
                if not line.strip():
                    continue

                try:
                    values = [float(x) for x in line.split()]
                    if len(values) < 2:
                        continue

                    x, y_orig = values[0], values[1]
                    y = y_orig + shift

                    if prev_x and prev_y:
                        delta_y = y_orig - prev_y
                        delta_x = x - prev_x

                        if abs(delta_y) > 300 and abs(delta_x) < 0.2:
                            if delta_y > 300:
                                shift -= 360
                            elif delta_y < -300:
                                shift += 360
                            y = y_orig + shift

                    al.append([x, y])
                    prev_x, prev_y = x, y_orig

                except Exception as e:
                    print(f"Ошибка парсинга строки в {filename}: {line.strip()} - {str(e)}")
                    raise e
                    continue
        if last_is_first:
            al[0][1] += shift
        return al

    def _parse_waveguide(self, filename):
        al = []
        shift = 0
        prev_x = None
        prev_y = None

        with open(filename, mode='r') as f:
            lines = f.readlines()
            for i in range(25):
                values = [float(x) for x in lines[2 + 3 * i + i * 1001 + round(i * 1000 / 24)].split()]  # Циклический сдвиг для правильного выбора точек
                if len(values) < 2:
                    continue

                x, y_orig = values[0], values[1]
                y = y_orig + shift

                if prev_x and prev_y:
                    delta_y = y_orig - prev_y
                    delta_x = x - prev_x

                    if abs(delta_y) > 300 and abs(delta_x) < 0.2:
                        if delta_y > 300:
                            shift -= 360
                        elif delta_y < -300:
                            shift += 360
                        y = y_orig + shift

                al.append([x, y])
                prev_x, prev_y = x, y_orig
        return al[1:]

    def approximate(self, al):  # функция возвращает коэффициенты аппроксимации вида y=a*arctg(b(x-c))+d
        data = np.array(al)
        x = data[:, 0]
        y = data[:, 1]

        def arctg_func(x, a, b, c, d):  # функция аппроксимации арктангенсом
            return a * np.arctan(b * (x - c)) + d

        try:
            params, covariance = curve_fit(arctg_func, x, y, maxfev=10000, bounds=([-300, 0, 0, -200], [-50, 600, 65, 200]))  # TODO: написать подбор приблизительных значений для увеличения скорости работы
            y_pred = arctg_func(x, *params)
            mse = np.mean((y_pred - y) ** 2)
            max_error = np.max(np.abs(y_pred - y))

            if self.debug and mse > 1:  # Отрисовка случаев плохой аппроксимации
                x_new = np.linspace(np.min(x), np.max(x), 1000)
                y_new = arctg_func(x_new, params[0], params[1], params[2], params[3])
                plt.scatter(x, y)
                sns.set(style="whitegrid", palette="muted", font_scale=2.5)
                if len(x) < 50:
                    plt.plot(x_new, y_new, label='Аппроксимация в волноводе', color='blue', linewidth=1)
                else:
                    plt.plot(x_new, y_new, label='Аппроксимация в свободном пространстве', color='red', linewidth=1)

            return params, mse, max_error
        except Exception as e:
            print(f"Ошибка аппроксимации: {e}")
            return [0, 0, 0, 0], float('nan'), float('nan')

    def _plot_approximation_error_distribution(self):  # Отрисовка распределения ошибок аппроксимации
        os.makedirs('approximation_plots', exist_ok=True)
        wg_mse = [e for e in self.waveguide_mse_errors if not np.isnan(e)]
        wg_max = [e for e in self.waveguide_max_errors if not np.isnan(e)]
        fs_mse = [e for e in self.freespace_mse_errors if not np.isnan(e)]
        fs_max = [e for e in self.freespace_max_errors if not np.isnan(e)]

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        if wg_mse:
            sns.histplot(wg_mse, bins=100, kde=True, color="blue")
            plt.title('MSE аппроксимаций (волновод)')
            plt.xlabel('MSE')
            plt.ylabel('Количество')

        plt.subplot(2, 2, 2)
        if fs_mse:
            sns.histplot(fs_mse, bins=100, kde=True, color="green")
            plt.title('MSE аппроксимаций (своб. пространство)')
            plt.xlabel('MSE')
            plt.ylabel('Количество')

        # Графики для максимальных ошибок
        plt.subplot(2, 2, 3)
        if wg_max:
            sns.histplot(wg_max, bins=100, kde=True, color="blue")
            plt.title('Макс. ошибка аппроксимаций (волновод)')
            plt.xlabel('Максимальная ошибка')
            plt.ylabel('Количество')

        plt.subplot(2, 2, 4)
        if fs_max:
            sns.histplot(fs_max, bins=100, kde=True, color="green")
            plt.title('Макс. ошибка аппроксимаций (своб. пространство)')
            plt.xlabel('Максимальная ошибка')
            plt.ylabel('Количество')

        plt.tight_layout()
        plt.savefig('approximation_plots/errors_distribution.svg')
        plt.show()
        plt.close()
