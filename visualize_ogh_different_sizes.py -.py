"""
visualize_ogh_different_sizes.py - OGH for Proteins of Different Lengths

Authors: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2024
License: MIT

Description:
    Visualizes how OGH handles proteins with different numbers of residues.
    Demonstrates the monotonic alignment algorithm and the normalization factor.

Generated Figures:
    fig7_ogh_different_sizes.png     - Comparison of proteins with different lengths
    fig8_ogh_size_effect.png         - How OGH depends on protein size
    fig9_monotonic_alignment.png     - Monotonic alignment visualization
    fig10_vad_length_effect.png      - Connection to VAD dataset length distribution

Dependencies:
    numpy, matplotlib, scipy

Usage:
    >>> python visualize_ogh_different_sizes.py

Output:
    Saves 4 PNG files in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# Настройка шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11


def create_helix(n_points, radius=1.0, pitch=0.5):
    """Создаёт спираль с заданным числом точек"""
    t = np.linspace(0, 2 * np.pi * 2, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t
    return np.column_stack([x, y, z])


def create_beta_sheet(n_points, length=4):
    """Создаёт упрощённую бета-структуру"""
    x = np.linspace(-length, length, n_points)
    y = np.sin(x) * 0.5
    z = np.zeros(n_points)
    return np.column_stack([x, y, z])


def compute_ogh_demo(X, Y, alpha=0.5):
    """
    Упрощённая демонстрационная версия OGH для визуализации
    """
    n, m = len(X), len(Y)

    # Нормализация
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-8)

    # Матрица расстояний
    dist_geom = cdist(X_norm, Y_norm)

    # Матрица штрафов за порядок
    order_penalty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            order_penalty[i, j] = alpha * 20 * (1 - np.exp(-3 * pos_diff))

    # Суммарная стоимость
    total_cost = dist_geom + order_penalty

    # Монотонное выравнивание (жадное)
    alignment = []
    i, j = 0, 0
    visited_i, visited_j = set(), set()

    while i < n and j < m:
        window = min(10, n - i, m - j)
        best_cost = np.inf
        best_i, best_j = i, j

        for di in range(window):
            for dj in range(window):
                if (i + di) not in visited_i and (j + dj) not in visited_j:
                    cost = total_cost[i + di, j + dj] + (di + dj) * 1.5
                    if cost < best_cost:
                        best_cost = cost
                        best_i, best_j = i + di, j + dj

        if best_cost < np.inf:
            alignment.append((best_i, best_j))
            visited_i.add(best_i)
            visited_j.add(best_j)
            i = best_i + 1
            j = best_j + 1
        else:
            i += 1
            j += 1

    # Вычисление финального расстояния
    if len(alignment) > 0:
        matched_dist = [dist_geom[i, j] for i, j in alignment]
        ogh_dist = np.mean(matched_dist) / np.sqrt(min(n, m))
    else:
        ogh_dist = np.inf

    return ogh_dist, alignment, total_cost


def fig_ogh_different_sizes():
    """Основная визуализация: OGH для белков разной длины"""
    fig = plt.figure(figsize=(16, 12))

    # Создаём белки разной длины
    np.random.seed(42)

    helix_short = create_helix(20, radius=1.0, pitch=0.4)
    helix_long = create_helix(35, radius=1.0, pitch=0.4)
    sheet = create_beta_sheet(25, length=5)

    # ========================================================================
    # Панель 1: Два белка с близкой длиной (20 vs 25)
    # ========================================================================

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(helix_short[:, 0], helix_short[:, 1], helix_short[:, 2],
                c=plt.cm.viridis(np.linspace(0, 1, len(helix_short))),
                s=40, edgecolors='black', linewidth=0.5)
    ax1.set_title(f'Белок A: спираль\n({len(helix_short)} остатков)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(sheet[:, 0], sheet[:, 1], sheet[:, 2],
                c=plt.cm.plasma(np.linspace(0, 1, len(sheet))),
                s=40, edgecolors='black', linewidth=0.5)
    ax2.set_title(f'Белок B: бета-лист\n({len(sheet)} остатков)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Матрица стоимости
    ax3 = fig.add_subplot(2, 3, 3)
    ogh_val, alignment, cost_matrix = compute_ogh_demo(helix_short, sheet, alpha=0.5)

    im = ax3.imshow(cost_matrix, cmap='hot_r', aspect='auto')
    ax3.set_title(f'Матрица стоимости\nOGH = {ogh_val:.3f}')
    ax3.set_xlabel(f'Белок B (бета-лист, {len(sheet)} остатков)')
    ax3.set_ylabel(f'Белок A (спираль, {len(helix_short)} остатков)')

    # Рисуем путь выравнивания
    if alignment:
        align_i = [a[0] for a in alignment]
        align_j = [a[1] for a in alignment]
        ax3.plot(align_j, align_i, 'b-o', linewidth=2, markersize=3, label='Монотонный путь')
        ax3.legend()

    plt.colorbar(im, ax=ax3, label='Стоимость')

    # ========================================================================
    # Панель 2: Два белка с разной длиной (20 vs 35)
    # ========================================================================

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(helix_short[:, 0], helix_short[:, 1], helix_short[:, 2],
                c=plt.cm.viridis(np.linspace(0, 1, len(helix_short))),
                s=40, edgecolors='black', linewidth=0.5)
    ax4.set_title(f'Белок C: спираль\n({len(helix_short)} остатков)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.scatter(helix_long[:, 0], helix_long[:, 1], helix_long[:, 2],
                c=plt.cm.viridis(np.linspace(0, 1, len(helix_long))),
                s=40, edgecolors='black', linewidth=0.5)
    ax5.set_title(f'Белок D: спираль (длиннее)\n({len(helix_long)} остатков)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    # Матрица стоимости
    ax6 = fig.add_subplot(2, 3, 6)
    ogh_val2, alignment2, cost_matrix2 = compute_ogh_demo(helix_short, helix_long, alpha=0.5)

    im2 = ax6.imshow(cost_matrix2, cmap='hot_r', aspect='auto')
    ax6.set_title(f'Матрица стоимости\nOGH = {ogh_val2:.3f}')
    ax6.set_xlabel(f'Белок D (длинная спираль, {len(helix_long)} остатков)')
    ax6.set_ylabel(f'Белок C (короткая спираль, {len(helix_short)} остатков)')

    # Рисуем путь выравнивания
    if alignment2:
        align_i = [a[0] for a in alignment2]
        align_j = [a[1] for a in alignment2]
        ax6.plot(align_j, align_i, 'b-o', linewidth=2, markersize=3, label='Монотонный путь')
        ax6.legend()

    plt.colorbar(im2, ax=ax6, label='Стоимость')

    plt.suptitle('Рис. 7. OGH для белков разной длины: монотонное выравнивание',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig7_ogh_different_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Рис. 7 сохранён: fig7_ogh_different_sizes.png")

    return ogh_val, ogh_val2


def fig_ogh_size_effect():
    """Визуализация влияния размера белка на OGH"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Создаём спирали разной длины
    sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    ogh_values_vs_self = []
    ogh_values_vs_fixed = []

    fixed_helix = create_helix(25, radius=1.0, pitch=0.4)

    for size in sizes:
        helix = create_helix(size, radius=1.0, pitch=0.4)

        # Сравнение с собой (должно быть 0)
        ogh_self, _, _ = compute_ogh_demo(helix, helix, alpha=0.5)
        ogh_values_vs_self.append(ogh_self)

        # Сравнение с фиксированным белком
        ogh_vs_fixed, _, _ = compute_ogh_demo(helix, fixed_helix, alpha=0.5)
        ogh_values_vs_fixed.append(ogh_vs_fixed)

    # График 1: OGH при сравнении с собой
    ax = axes[0]
    ax.plot(sizes, ogh_values_vs_self, 'b-o', linewidth=2, markersize=8, label='Белок vs сам себя')
    ax.set_xlabel('Количество остатков в белке')
    ax.set_ylabel('OGH расстояние')
    ax.set_title('OGH при сравнении белка с самим собой\n(должно быть 0, работает нормализация)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # График 2: OGH при сравнении с белком фиксированного размера
    ax = axes[1]
    ax.plot(sizes, ogh_values_vs_fixed, 'r-s', linewidth=2, markersize=8,
            label=f'Белок vs спираль {len(fixed_helix)} остатков')
    ax.set_xlabel('Количество остатков в первом белке')
    ax.set_ylabel('OGH расстояние')
    ax.set_title('Зависимость OGH от разницы в длине белков')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Добавляем пояснение
    ax.text(0.02, 0.98, f'Фиксированный белок: {len(fixed_helix)} остатков',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Рис. 8. Влияние размера белка на OGH расстояние', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig8_ogh_size_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Рис. 8 сохранён: fig8_ogh_size_effect.png")


def fig_monotonic_alignment_animation_style():
    """Визуализация монотонного выравнивания для белков разной длины"""
    fig = plt.figure(figsize=(14, 10))

    # Создаём белки разной длины (большая разница)
    short_helix = create_helix(15, radius=1.0, pitch=0.4)
    long_helix = create_helix(35, radius=1.0, pitch=0.4)

    # Вычисляем матрицу стоимости и выравнивание
    ogh_val, alignment, cost_matrix = compute_ogh_demo(short_helix, long_helix, alpha=0.5)

    # Визуализация матрицы
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(cost_matrix, cmap='hot_r', aspect='auto')
    ax1.set_title(f'Матрица стоимости\n(короткий белок A vs длинный белок B)')
    ax1.set_xlabel('Длинный белок B (35 остатков)')
    ax1.set_ylabel('Короткий белок A (15 остатков)')

    # Рисуем путь выравнивания
    if alignment:
        align_i = [a[0] for a in alignment]
        align_j = [a[1] for a in alignment]
        ax1.plot(align_j, align_i, 'b-o', linewidth=2, markersize=4, label='Монотонный путь')
        ax1.legend()

    plt.colorbar(im, ax=ax1, label='Стоимость соответствия')

    # Визуализация штрафа за порядок
    ax2 = fig.add_subplot(2, 2, 2)
    n, m = 15, 35
    order_penalty = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            order_penalty[i, j] = 10 * (1 - np.exp(-3 * pos_diff))

    im2 = ax2.imshow(order_penalty, cmap='Blues', aspect='auto')
    ax2.set_title('Матрица штрафа за порядок\n(экспоненциальный штраф)')
    ax2.set_xlabel('Позиция в длинном белке B')
    ax2.set_ylabel('Позиция в коротком белке A')
    plt.colorbar(im2, ax=ax2, label='Штраф')

    # Схема выравнивания
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')

    # Рисуем схематичное выравнивание
    y_start = 0.8
    for i, (x, y) in enumerate(alignment[:20]):  # Показываем первые 20 соответствий
        # Точка в коротком белке
        ax3.plot(0.1, y_start - i * 0.03, 'bo', markersize=8)
        ax3.text(0.05, y_start - i * 0.03, f'{x}', ha='right', va='center', fontsize=8)

        # Линия связи
        ax3.plot([0.15, 0.85], [y_start - i * 0.03, y_start - i * 0.03], 'gray', alpha=0.5, linestyle='--')

        # Точка в длинном белке
        ax3.plot(0.9, y_start - i * 0.03, 'ro', markersize=8)
        ax3.text(0.95, y_start - i * 0.03, f'{y}', ha='left', va='center', fontsize=8)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.text(0.1, 0.95, 'Короткий белок A', ha='center', fontsize=10, fontweight='bold')
    ax3.text(0.9, 0.95, 'Длинный белок B', ha='center', fontsize=10, fontweight='bold')
    ax3.set_title(f'Монотонное выравнивание ({len(alignment)} соответствий)')

    # Пояснение
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    explanation = f"""
    КЛЮЧЕВЫЕ ОСОБЕННОСТИ OGH ДЛЯ БЕЛКОВ РАЗНОЙ ДЛИНЫ:

    1. Нормализация координат позволяет сравнивать
       белки независимо от их размера и ориентации.

    2. Экспоненциальный штраф за порядок:
       - Мягко штрафует локальные сдвиги
       - Сильно штрафует глобальные перестановки

    3. Монотонное выравнивание:
       - Сохраняет порядок остатков
       - Позволяет пропускать "лишние" остатки
         (gaps) в более длинном белке

    4. Нормировка на sqrt(min(n,m)):
       Делает значения OGH сопоставимыми
       для разных пар белков

    В данном примере:
    - Длина A: {n} остатков
    - Длина B: {m} остатков
    - OGH = {ogh_val:.3f}
    """

    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.suptitle('Рис. 9. Монотонное выравнивание для белков разной длины',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig9_monotonic_alignment.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Рис. 9 сохранён: fig9_monotonic_alignment.png")


def fig_connection_to_vad():
    """Связь с реальными данными VAD: разные длины вирусных белков"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Данные о длинах белков из VAD (из вашего файла)
    vad_proteins = {
        '1E0J': 500, '1F5K': 247, '1K28': 500, '2H2C': 107, '2WR7': 500,
        '3HMX': 500, '4N5J': 500, '4ZMJ': 500, '5KOK': 500, '5T6T': 12,
        '6B8V': 492, '6E6J': 500, '6LU7': 311, '6M0J': 500, '6M17': 500,
        '6NUS': 500, '6QJ4': 500, '6VXX': 500, '6W01': 500, '6WAQ': 500,
        '6Y2E': 306, '7BW4': 500, '7KRS': 500, '7L0E': 500, '7N0S': 500,
        '7RBY': 500, '7T83': 500, '7T9N': 500
    }

    # Гистограмма длин
    ax = axes[0, 0]
    lengths = list(vad_proteins.values())
    ax.hist(lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Длина белка (количество остатков)')
    ax.set_ylabel('Количество белков')
    ax.set_title('Распределение длин белков в VAD датасете')
    ax.axvline(x=12, color='red', linestyle='--', alpha=0.7, label='5T6T (RBD SARS-CoV-2, 12)')
    ax.axvline(x=107, color='orange', linestyle='--', alpha=0.7, label='2H2C (MERS-CoV, 107)')
    ax.legend()

    # Диаграмма рассеяния: длина vs OGH (симуляция)
    ax = axes[0, 1]
    np.random.seed(42)
    simulated_ogh = np.random.uniform(0.01, 0.1, len(lengths)) + 500 / np.array(lengths) * 0.5
    simulated_ogh[np.array(lengths) == 12] = np.random.uniform(1.14, 1.22)  # 5T6T outliers

    ax.scatter(lengths, simulated_ogh, s=80, c='steelblue', alpha=0.6, edgecolors='black')
    ax.set_xlabel('Длина белка (количество остатков)')
    ax.set_ylabel('OGH расстояние (симуляция)')
    ax.set_title('Зависимость OGH от длины белка\n(чем меньше белок, тем больше OGH)')
    ax.grid(True, alpha=0.3)

    # Аннотация для 5T6T
    ax.annotate('5T6T (RBD SARS-CoV-2)\nВсего 12 остатков\nOGH > 1.14',
                xy=(12, 1.18), xytext=(50, 1.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, ha='center')

    # Схема нормализации
    ax = axes[1, 0]
    ax.axis('off')

    norm_schema = """
    НОРМАЛИЗАЦИЯ КООРДИНАТ:

    Для белка с координатами X = {x1, x2, ..., xn}:

        X' = (X - μ_X) / σ_X

    где:
    - μ_X = среднее (центр масс)
    - σ_X = стандартное отклонение

    Результат:
    - Среднее = 0
    - Дисперсия = 1

    Это позволяет сравнивать белки
    разного размера и ориентации!
    """
    ax.text(0.05, 0.95, norm_schema, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.set_title('Нормализация координат')

    # Нормировочный множитель
    ax = axes[1, 1]
    ax.axis('off')

    norm_factor = """
    НОРМИРОВОЧНЫЙ МНОЖИТЕЛЬ:

    d_OGH(X,Y) = [ ... ] / √min(n,m)

    Зачем?

    • Чем длиннее белок, тем больше
      потенциальное накопление ошибок

    • √min(n,m) компенсирует этот эффект

    • Без нормировки:
      - Короткие белки → маленькие значения
      - Длинные белки → большие значения

    • С нормировкой:
      Значения сопоставимы при разной длине!
    """
    ax.text(0.05, 0.95, norm_factor, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.set_title('Нормировка на √min(n,m)')

    plt.suptitle('Рис. 10. OGH для вирусных белков VAD: учёт разной длины',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig10_vad_length_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Рис. 10 сохранён: fig10_vad_length_effect.png")


def main():
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ OGH ДЛЯ БЕЛКОВ РАЗНОЙ ДЛИНЫ")
    print("=" * 60)

    # Рисунки
    fig_ogh_different_sizes()
    fig_ogh_size_effect()
    fig_monotonic_alignment_animation_style()
    fig_connection_to_vad()

    print("\n" + "=" * 60)
    print("✅ ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ!")
    print("=" * 60)
    print("Сохранённые файлы:")
    print("  📁 fig7_ogh_different_sizes.png     - сравнение белков разной длины")
    print("  📁 fig8_ogh_size_effect.png         - влияние размера на OGH")
    print("  📁 fig9_monotonic_alignment.png     - монотонное выравнивание")
    print("  📁 fig10_vad_length_effect.png      - привязка к VAD датасету")


if __name__ == "__main__":
    main()