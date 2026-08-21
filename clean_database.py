#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_database.py — Clean and curate the homologs database

Description:
    Filters and cleans the homologs database by removing invalid sequences,
    duplicates, and filling missing classifications.

Input:
    homologs_database.csv — Raw database from parse_homologs.py

Output:
    homologs_database_clean.csv — Curated database
    cleaning_report.txt — Detailed cleaning report
    unique_families.txt — List of unique families

Dependencies:
    pandas

Usage:
    python clean_database.py

Author: Andrey V. Timofeev, Alexandr S. Anufriev
Date: 2026
License: MIT
"""

import pandas as pd
import re
from pathlib import Path


def clean_homologs_database(input_csv, output_csv, report_file):
    """
    Очищает базу данных гомологов от ошибочных записей
    """
    print("📂 Загрузка данных...")
    df = pd.read_csv(input_csv)
    print(f"Загружено записей: {len(df)}")

    # Создаем копию для очистки
    df_clean = df.copy()

    # 1. Удаляем записи с пустой последовательностью
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['sequence'].notna() & (df_clean['sequence'] != '')]
    print(f"Удалено записей с пустой последовательностью: {initial_count - len(df_clean)}")

    # 2. Удаляем записи с очень короткими последовательностями (< 20 аминокислот)
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['sequence_length'] >= 20]
    print(f"Удалено записей с длиной < 20: {initial_count - len(df_clean)}")

    # 3. Удаляем записи с подозрительно длинными последовательностями (> 5000)
    #    (обычные белки редко превышают 5000 аминокислот)
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['sequence_length'] <= 5000]
    print(f"Удалено записей с длиной > 5000: {initial_count - len(df_clean)}")

    # 4. Проверяем последовательности на валидные аминокислоты
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

    def is_valid_sequence(seq):
        if not seq or not isinstance(seq, str):
            return False
        # Убираем возможные пробелы и спецсимволы
        seq_clean = ''.join(seq.split())
        # Проверяем, что все символы - валидные аминокислоты
        return all(c.upper() in valid_aa for c in seq_clean)

    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['sequence'].apply(is_valid_sequence)]
    print(f"Удалено записей с невалидными аминокислотами: {initial_count - len(df_clean)}")

    # 5. Удаляем дубликаты по ID
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')
    print(f"Удалено дубликатов по ID: {initial_count - len(df_clean)}")

    # 6. Заполняем пропуски в классификации
    #    Пытаемся извлечь классификацию из protein_name, если она пустая
    def extract_class_from_name(row):
        if pd.isna(row['class_family']) or row['class_family'] == '':
            # Ищем паттерн a.1.1.1 в protein_name
            match = re.search(r'([a-z]\.[\d]+\.[\d]+\.[\d]+)', str(row['protein_name']))
            if match:
                return match.group(1)
        return row['class_family']

    df_clean['class_family'] = df_clean.apply(extract_class_from_name, axis=1)

    # 7. Обновляем superfamily и family на основе class_family
    def extract_superfamily(class_family):
        if pd.isna(class_family) or class_family == '':
            return ''
        parts = str(class_family).split('.')
        return '.'.join(parts[:2]) if len(parts) >= 2 else ''

    def extract_family(class_family):
        if pd.isna(class_family) or class_family == '':
            return ''
        parts = str(class_family).split('.')
        return '.'.join(parts[:3]) if len(parts) >= 3 else ''

    df_clean['superfamily'] = df_clean['class_family'].apply(extract_superfamily)
    df_clean['family'] = df_clean['class_family'].apply(extract_family)

    # 8. Сортируем по семействам и ID
    df_clean = df_clean.sort_values(['family', 'id']).reset_index(drop=True)

    # 9. Добавляем статистику
    print(f"\n📊 Итоговая статистика:")
    print(f"  - Всего записей после очистки: {len(df_clean)}")
    print(f"  - Уникальных семейств: {df_clean['family'].nunique()}")
    print(f"  - Уникальных суперсемейств: {df_clean['superfamily'].nunique()}")
    print(f"  - Средняя длина последовательности: {df_clean['sequence_length'].mean():.0f}")
    print(f"  - Мин. длина: {df_clean['sequence_length'].min()}")
    print(f"  - Макс. длина: {df_clean['sequence_length'].max()}")

    # 10. Сохраняем очищенный CSV
    df_clean.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ Очищенный CSV сохранен: {output_csv}")

    # 11. Создаем отчет
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== ОТЧЕТ ОБ ОЧИСТКЕ БАЗЫ ДАННЫХ ГОМОЛОГОВ ===\n\n")

        f.write("1. ОБЩАЯ СТАТИСТИКА\n")
        f.write("-" * 40 + "\n")
        f.write(f"Всего записей: {len(df_clean)}\n")
        f.write(f"Уникальных ID: {df_clean['id'].nunique()}\n")
        f.write(f"Уникальных семейств: {df_clean['family'].nunique()}\n")
        f.write(f"Уникальных суперсемейств: {df_clean['superfamily'].nunique()}\n")
        f.write(f"Уникальных организмов: {df_clean['organism'].nunique()}\n\n")

        f.write("2. СТАТИСТИКА ПО ДЛИНЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ\n")
        f.write("-" * 40 + "\n")
        f.write(f"Средняя: {df_clean['sequence_length'].mean():.0f}\n")
        f.write(f"Медиана: {df_clean['sequence_length'].median():.0f}\n")
        f.write(f"Мин.: {df_clean['sequence_length'].min()}\n")
        f.write(f"Макс.: {df_clean['sequence_length'].max()}\n")
        f.write(f"Станд. отклонение: {df_clean['sequence_length'].std():.0f}\n\n")

        f.write("3. РАСПРЕДЕЛЕНИЕ ПО СЕМЕЙСТВАМ (ТОП-20)\n")
        f.write("-" * 40 + "\n")
        family_counts = df_clean['family'].value_counts()
        for family, count in family_counts.head(20).items():
            if family and family != '':
                f.write(f"{family}: {count} записей\n")

        f.write("\n4. РАСПРЕДЕЛЕНИЕ ПО СУПЕРСЕМЕЙСТВАМ (ТОП-20)\n")
        f.write("-" * 40 + "\n")
        superfamily_counts = df_clean['superfamily'].value_counts()
        for superfamily, count in superfamily_counts.head(20).items():
            if superfamily and superfamily != '':
                f.write(f"{superfamily}: {count} записей\n")

        f.write("\n5. ТОП-20 ОРГАНИЗМОВ\n")
        f.write("-" * 40 + "\n")
        organism_counts = df_clean['organism'].value_counts()
        for organism, count in organism_counts.head(20).items():
            if organism and organism != 'Unknown' and organism != '':
                f.write(f"{organism}: {count} записей\n")

        f.write("\n6. ЗАПИСИ БЕЗ КЛАССИФИКАЦИИ\n")
        f.write("-" * 40 + "\n")
        no_class = df_clean[df_clean['class_family'] == '']
        f.write(f"Количество: {len(no_class)}\n")
        if len(no_class) > 0:
            f.write("Примеры:\n")
            for _, row in no_class.head(10).iterrows():
                f.write(f"  {row['id']}: {row['protein_name']}\n")

    print(f"📊 Отчет сохранен: {report_file}")

    return df_clean


def create_sequence_files(df, output_dir):
    """
    Создает FASTA-файлы по семействам
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n📁 Создание FASTA-файлов по семействам...")

    # Группируем по семействам
    for family, group in df.groupby('family'):
        if family and family != '':
            fasta_file = output_dir / f"{family.replace('.', '_')}.fasta"

            with open(fasta_file, 'w', encoding='utf-8') as f:
                for _, row in group.iterrows():
                    header = f">{row['id']} {row['protein_name']} [{row['organism']}]"
                    f.write(f"{header}\n")
                    # Форматируем последовательность по 60 символов
                    seq = row['sequence']
                    for i in range(0, len(seq), 60):
                        f.write(f"{seq[i:i + 60]}\n")

            print(f"  Создан {fasta_file.name} ({len(group)} записей)")

    print(f"\n✅ Все FASTA-файлы сохранены в {output_dir}")


def create_summary_statistics(df):
    """
    Создает сводную статистику
    """
    stats = {
        'total_sequences': len(df),
        'unique_families': df['family'].nunique(),
        'unique_superfamilies': df['superfamily'].nunique(),
        'unique_organisms': df['organism'].nunique(),
        'avg_length': df['sequence_length'].mean(),
        'median_length': df['sequence_length'].median(),
        'min_length': df['sequence_length'].min(),
        'max_length': df['sequence_length'].max(),
        'std_length': df['sequence_length'].std(),
        'families_distribution': df['family'].value_counts().to_dict(),
        'organisms_distribution': df['organism'].value_counts().head(20).to_dict()
    }

    # Создаем читаемый отчет
    print("\n" + "=" * 60)
    print("📊 СВОДНАЯ СТАТИСТИКА БАЗЫ ДАННЫХ")
    print("=" * 60)
    print(f"Всего последовательностей: {stats['total_sequences']}")
    print(f"Уникальных семейств: {stats['unique_families']}")
    print(f"Уникальных суперсемейств: {stats['unique_superfamilies']}")
    print(f"Уникальных организмов: {stats['unique_organisms']}")
    print(f"\nДлины последовательностей:")
    print(f"  Средняя: {stats['avg_length']:.0f}")
    print(f"  Медиана: {stats['median_length']:.0f}")
    print(f"  Мин.: {stats['min_length']}")
    print(f"  Макс.: {stats['max_length']}")
    print(f"  Станд. отклонение: {stats['std_length']:.0f}")
    print("=" * 60)

    return stats


def main():
    project_root = Path(__file__).parent
    input_csv = project_root / 'homologs_database.csv'
    output_csv = project_root / 'homologs_database_clean.csv'
    report_file = project_root / 'cleaning_report.txt'
    fasta_dir = project_root / 'fasta_files_by_family'

    print("🧹 НАЧАЛО ОЧИСТКИ БАЗЫ ДАННЫХ ГОМОЛОГОВ")
    print("=" * 60)

    if not input_csv.exists():
        print(f"❌ Файл не найден: {input_csv}")
        print("Сначала запустите основной скрипт для создания базы данных.")
        return

    # Очищаем базу данных
    df_clean = clean_homologs_database(input_csv, output_csv, report_file)

    # Создаем сводную статистику
    stats = create_summary_statistics(df_clean)

    # Создаем FASTA-файлы по семействам (опционально)
    create_fasta = input("\nСоздать FASTA-файлы по семействам? (y/n): ").lower() == 'y'
    if create_fasta:
        create_sequence_files(df_clean, fasta_dir)

    # Показываем примеры очищенных данных
    print("\n📝 Примеры очищенных данных:")
    print(df_clean[['id', 'family', 'protein_name', 'sequence_length']].head(10))

    # Сохраняем также список всех уникальных семейств
    families_file = project_root / 'unique_families.txt'
    with open(families_file, 'w', encoding='utf-8') as f:
        f.write("УНИКАЛЬНЫЕ СЕМЕЙСТВА В БАЗЕ ДАННЫХ\n")
        f.write("=" * 50 + "\n")
        for family in sorted(df_clean['family'].unique()):
            if family and family != '':
                count = len(df_clean[df_clean['family'] == family])
                f.write(f"{family}: {count} записей\n")

    print(f"\n✅ Список семейств сохранен: {families_file}")
    print("\n✨ Очистка завершена успешно!")


if __name__ == "__main__":
    main()