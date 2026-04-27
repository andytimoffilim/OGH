"""
visualize_ogh_simple.py - Simple OGH Illustration for Non-Experts

Authors: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2024
License: MIT

Description:
    Creates two simple, intuitive figures to explain OGH to non-experts:
    - Which residues are compared with which
    - How the order penalty works (three cases)

Generated Figures:
    fig_ogh_simple.png   - Main illustration: two proteins and their matching
    fig_ogh_penalty.png  - Three cases: small/medium/large order penalty

Dependencies:
    numpy, matplotlib

Usage:
    >>> python visualize_ogh_simple.py

Output:
    Saves 2 PNG files in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings('ignore')

# Настройка шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


def fig_ogh_simple_comparison():
    """Самая простая иллюстрация: два белка и их сравнение"""

    fig = plt.figure(figsize=(14, 10))

    # ============================================================
    # БЕЛОК 1 (слева) - короткая спираль из 6 точек
    # ============================================================

    # Создаём простую спираль
    t = np.linspace(0, 2 * np.pi, 6)
    x1 = np.cos(t)
    y1 = np.sin(t)
    z1 = t

    # ============================================================
    # БЕЛОК 2 (справа) - длинная спираль из 9 точек
    # ============================================================

    t2 = np.linspace(0, 2 * np.pi, 9)
    x2 = np.cos(t2) * 1.2
    y2 = np.sin(t2) * 1.2
    z2 = t2 * 1.5

    # ============================================================
    # 3D ВИЗУАЛИЗАЦИЯ БЕЛКОВ
    # ============================================================

    # Белок 1
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(x1)))
    for i in range(len(x1)):
        ax1.scatter(x1[i], y1[i], z1[i], c=[colors1[i]], s=150, edgecolors='black', linewidth=2)
        ax1.text(x1[i], y1[i], z1[i] + 0.2, str(i + 1), fontsize=10, ha='center', fontweight='bold')

    # Соединяем точки линией (пептидная связь)
    for i in range(len(x1) - 1):
        ax1.plot([x1[i], x1[i + 1]], [y1[i], y1[i + 1]], [z1[i], z1[i + 1]], 'gray', linewidth=2, alpha=0.5)

    ax1.set_title('БЕЛОК A (короткий)\n6 остатков, пронумерованы по порядку', fontsize=11)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)

    # Белок 2
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(x2)))
    for i in range(len(x2)):
        ax2.scatter(x2[i], y2[i], z2[i], c=[colors2[i]], s=150, edgecolors='black', linewidth=2)
        ax2.text(x2[i], y2[i], z2[i] + 0.2, str(i + 1), fontsize=10, ha='center', fontweight='bold')

    # Соединяем точки линией
    for i in range(len(x2) - 1):
        ax2.plot([x2[i], x2[i + 1]], [y2[i], y2[i + 1]], [z2[i], z2[i + 1]], 'gray', linewidth=2, alpha=0.5)

    ax2.set_title('БЕЛОК B (длинный)\n9 остатков, пронумерованы по порядку', fontsize=11)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=45)

    # ============================================================
    # МАТРИЦА СООТВЕТСТВИЙ И ШТРАФОВ
    # ============================================================

    # Создаём матрицу для визуализации
    n, m = 6, 9
    dist_matrix = np.zeros((n, m))
    penalty_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # Позиционное различие (штраф за порядок)
            pos_diff = abs(i / n - j / m)
            # Экспоненциальный штраф
            penalty_matrix[i, j] = 1 - np.exp(-3 * pos_diff)

            # Цвет: зелёный если близко к диагонали, красный если далеко
            dist_matrix[i, j] = pos_diff

    # Матрица штрафов
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(penalty_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_title('ШТРАФ ЗА НАРУШЕНИЕ ПОРЯДКА\n(чем краснее, тем больше штраф)', fontsize=10)
    ax3.set_xlabel('Остатки в белке B (9 шт.)')
    ax3.set_ylabel('Остатки в белке A (6 шт.)')

    # Рисуем идеальную диагональ
    ax3.plot([0, m - 1], [0, n - 1], 'b--', linewidth=2, alpha=0.7, label='Идеальная диагональ')
    ax3.legend()

    # Добавляем номера на оси
    ax3.set_xticks(range(m))
    ax3.set_xticklabels(range(1, m + 1))
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(range(1, n + 1))

    plt.colorbar(im, ax=ax3, label='Величина штрафа')

    # ============================================================
    # ОПТИМАЛЬНОЕ СООТВЕТСТВИЕ (МОНОТОННОЕ)
    # ============================================================

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(penalty_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Оптимальный путь (монотонное выравнивание)
    path = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 6), (5, 7)]
    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]

    ax4.plot(path_j, path_i, 'b-o', linewidth=3, markersize=8,
             markerfacecolor='white', markeredgecolor='blue',
             label='Оптимальный путь')
    ax4.set_title('ОПТИМАЛЬНОЕ МОНОТОННОЕ СООТВЕТСТВИЕ\n(путь только вправо и вниз)', fontsize=10)
    ax4.set_xlabel('Остатки в белке B (9 шт.)')
    ax4.set_ylabel('Остатки в белке A (6 шт.)')
    ax4.set_xticks(range(m))
    ax4.set_xticklabels(range(1, m + 1))
    ax4.set_yticks(range(n))
    ax4.set_yticklabels(range(1, n + 1))
    ax4.legend()

    # ============================================================
    # СХЕМА СООТВЕТСТВИЯ МЕЖДУ БЕЛКАМИ
    # ============================================================

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')

    # Левая колонка (белок A)
    y_start = 0.9
    for i in range(n):
        color = colors1[i]
        ax5.add_patch(plt.Circle((0.15, y_start - i * 0.12), 0.05, color=color, ec='black', linewidth=1.5))
        ax5.text(0.25, y_start - i * 0.12, f'Остаток {i + 1}', fontsize=10, va='center')

    # Правая колонка (белок B)
    for j in range(m):
        color = colors2[j]
        ax5.add_patch(plt.Circle((0.85, y_start - j * 0.08), 0.05, color=color, ec='black', linewidth=1.5))
        ax5.text(0.75, y_start - j * 0.08, f'Остаток {j + 1}', fontsize=10, ha='right', va='center')

    # Линии соответствия
    for (i, j) in path:
        y_a = y_start - i * 0.12
        y_b = y_start - j * 0.08
        # Цвет линии зависит от штрафа
        pos_diff = abs(i / n - j / m)
        if pos_diff < 0.2:
            line_color = 'green'
            line_width = 2
        elif pos_diff < 0.4:
            line_color = 'orange'
            line_width = 2
        else:
            line_color = 'red'
            line_width = 2
        ax5.plot([0.2, 0.8], [y_a, y_b], color=line_color, linewidth=line_width, alpha=0.6)

    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('СООТВЕТСТВИЕ МЕЖДУ ОСТАТКАМИ\n(зелёные линии = малый штраф, красные = большой)', fontsize=10)

    # ============================================================
    # ПОЯСНЕНИЕ
    # ============================================================

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    explanation = """
    КАК РАБОТАЕТ OGH:

    1. Сравниваются ПОПАРНО все остатки:
       - Остаток i из белка A
       - Остаток j из белка B

    2. Каждой паре (i,j) назначается ШТРАФ:
       - Маленький штраф, если i/n ≈ j/m
         (остатки близки к началу/концу)
       - Большой штраф, если |i/n - j/m| большое
         (порядок нарушен)

    3. Ищется МОНОТОННЫЙ путь:
       i₁ < i₂ < ... и j₁ < j₂ < ...
       (нельзя возвращаться назад)

    4. Итоговое расстояние OGH:
       Усреднение штрафов + нормализация

    📌 Главное отличие от GH:
       GH смотрит только на геометрию,
       OGH добавляет ШТРАФ ЗА ПОРЯДОК!
    """

    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('УПРОЩЁННАЯ ИЛЛЮСТРАЦИЯ OGH: что с чем сравнивается и за что штраф',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_ogh_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Рисунок сохранён: fig_ogh_simple.png")


def fig_ogh_penalty_explanation():
    """Отдельная картинка: объяснение штрафа за порядок"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ============================================================
    # СЛУЧАЙ 1: ХОРОШИЙ ПОРЯДОК (малый штраф)
    # ============================================================

    ax = axes[0]
    ax.axis('off')

    # Белок A (слева)
    for i in range(5):
        circle = plt.Circle((0.2, 0.8 - i * 0.15), 0.05, color='lightgreen', ec='green', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.1, 0.8 - i * 0.15, f'{i + 1}', ha='right', va='center', fontsize=9)

    # Белок B (справа)
    for j in range(5):
        circle = plt.Circle((0.8, 0.8 - j * 0.15), 0.05, color='lightgreen', ec='green', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.9, 0.8 - j * 0.15, f'{j + 1}', ha='left', va='center', fontsize=9)

    # Линии (диагональ)
    for k in range(5):
        ax.plot([0.25, 0.75], [0.8 - k * 0.15, 0.8 - k * 0.15], 'green', linewidth=2, alpha=0.7)

    ax.text(0.5, 0.05, '✅ МАЛЫЙ ШТРАФ\ni/n ≈ j/m\n(остатки на одинаковых позициях)',
            ha='center', fontsize=10, color='green', fontweight='bold')
    ax.set_title('Случай 1: Порядок сохраняется', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ============================================================
    # СЛУЧАЙ 2: СРЕДНИЙ ШТРАФ
    # ============================================================

    ax = axes[1]
    ax.axis('off')

    # Белок A
    for i in range(5):
        circle = plt.Circle((0.2, 0.8 - i * 0.15), 0.05, color='lightyellow', ec='orange', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.1, 0.8 - i * 0.15, f'{i + 1}', ha='right', va='center', fontsize=9)

    # Белок B
    for j in range(5):
        circle = plt.Circle((0.8, 0.8 - j * 0.15), 0.05, color='lightyellow', ec='orange', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.9, 0.8 - j * 0.15, f'{j + 1}', ha='left', va='center', fontsize=9)

    # Сдвинутые линии
    ax.plot([0.25, 0.75], [0.8, 0.65], 'orange', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.65, 0.5], 'orange', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.5, 0.35], 'orange', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.35, 0.2], 'orange', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.2, 0.05], 'orange', linewidth=2, alpha=0.7)

    ax.text(0.5, 0.05, '⚠️ СРЕДНИЙ ШТРАФ\nНебольшое смещение порядка',
            ha='center', fontsize=10, color='orange', fontweight='bold')
    ax.set_title('Случай 2: Умеренное нарушение', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ============================================================
    # СЛУЧАЙ 3: ПЕРЕСТАВЛЕННЫЙ ПОРЯДОК (большой штраф)
    # ============================================================

    ax = axes[2]
    ax.axis('off')

    # Белок A
    for i in range(5):
        circle = plt.Circle((0.2, 0.8 - i * 0.15), 0.05, color='lightcoral', ec='red', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.1, 0.8 - i * 0.15, f'{i + 1}', ha='right', va='center', fontsize=9)

    # Белок B (переставленный)
    order = [3, 1, 4, 2, 5]
    for j, pos in enumerate(order):
        circle = plt.Circle((0.8, 0.8 - (pos - 1) * 0.15), 0.05, color='lightcoral', ec='red', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.9, 0.8 - (pos - 1) * 0.15, f'{pos}', ha='left', va='center', fontsize=9)

    # Перекрёстные линии
    ax.plot([0.25, 0.75], [0.8, 0.5], 'red', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.65, 0.8], 'red', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.5, 0.35], 'red', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.35, 0.65], 'red', linewidth=2, alpha=0.7)
    ax.plot([0.25, 0.75], [0.2, 0.2], 'red', linewidth=2, alpha=0.7)

    ax.text(0.5, 0.05, '❌ БОЛЬШОЙ ШТРАФ\nПорядок полностью нарушен\n(перестановка остатков)',
            ha='center', fontsize=10, color='red', fontweight='bold')
    ax.set_title('Случай 3: Переставленный порядок', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle('ШТРАФ ЗА ПОРЯДОК В OGH: чем сильнее нарушен порядок, тем больше штраф',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_ogh_penalty.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Рисунок сохранён: fig_ogh_penalty.png")


def main():
    print("=" * 60)
    print("ПРОСТАЯ ВИЗУАЛИЗАЦИЯ OGH")
    print("=" * 60)

    fig_ogh_simple_comparison()
    fig_ogh_penalty_explanation()

    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print("Сохранённые файлы:")
    print("  📁 fig_ogh_simple.png   - основная иллюстрация OGH")
    print("  📁 fig_ogh_penalty.png  - объяснение штрафа за порядок")


if __name__ == "__main__":
    main()