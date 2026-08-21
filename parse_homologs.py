#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_homologs.py — Extract protein records from Supplementary Material S1.pdf

Description:
    Parses the PDF file containing SCOP/ASTRAL protein sequences and
    generates a structured CSV database with sequence, family, and
    superfamily annotations.

Input:
    Supplementary Material S1.pdf (PDF with FASTA-like format)

Output:
    homologs_database.csv — Raw database with all records
    extracted_text_debug.txt — Extracted text for debugging
    database_statistics.txt — Summary statistics

Dependencies:
    pdfplumber, pandas

Usage:
    python parse_homologs.py

Author: Andrey V. Timofeev, Alexandr S. Anufriev
Date: 2026
License: MIT
"""

import re
import csv
from pathlib import Path
import pdfplumber
import pandas as pd


def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF-файла с подробной отладкой
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Всего страниц в PDF: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                print(f"Страница {i + 1}: извлечено {len(text)} символов")
                all_text.append(text)
            else:
                print(f"Страница {i + 1}: текст не извлечен")

    full_text = "\n".join(all_text)
    print(f"Всего извлечено символов: {len(full_text)}")
    return full_text


def parse_fasta_like_text(content):
    """
    Парсит текст с улучшенным распознаванием записей
    """
    records = []
    current_record = {}

    # Разбиваем на строки и чистим
    lines = content.split('\n')
    print(f"Всего строк для парсинга: {len(lines)}")

    # Ищем все строки, начинающиеся с '>'
    header_lines = [i for i, line in enumerate(lines) if line.strip().startswith('>')]
    print(f"Найдено заголовков (строк с '>'): {len(header_lines)}")

    if not header_lines:
        # Если не нашли '>', попробуем поискать по шаблону d1...
        print("Строки с '>' не найдены. Ищем по шаблону d1...")
        for i, line in enumerate(lines):
            if re.search(r'd\d\w+_', line):
                print(f"Найдена потенциальная запись в строке {i}: {line[:100]}")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Проверяем, является ли строка заголовком
        is_header = False

        # Основной паттерн для заголовка
        header_match = re.match(r'>(d\d\w+_)\s+([\w.]+)\s+\(([^)]*)\)\s+(.+?)\s+\{(.+)\}', line)
        if header_match:
            is_header = True
            if current_record:
                records.append(current_record)
                current_record = {}

            current_record['id'] = header_match.group(1)
            current_record['class_family'] = header_match.group(2)
            current_record['chain'] = header_match.group(3) if header_match.group(3) else '-'
            current_record['protein_name'] = header_match.group(4).strip()
            current_record['organism'] = header_match.group(5).strip()
            current_record['sequence'] = ''
            continue

        # Альтернативный паттерн
        alt_match = re.match(r'>(d\d\w+_)\s+([\w.]+)\s+(.+?)\s+\{(.+)\}', line)
        if alt_match:
            is_header = True
            if current_record:
                records.append(current_record)
                current_record = {}

            current_record['id'] = alt_match.group(1)
            current_record['class_family'] = alt_match.group(2)
            current_record['chain'] = '-'
            current_record['protein_name'] = alt_match.group(3).strip()
            current_record['organism'] = alt_match.group(4).strip()
            current_record['sequence'] = ''
            continue

        # Простой паттерн для заголовка
        simple_match = re.match(r'>(d\d\w+_)\s+(.+?)(?:\s+\{(.+)\})?$', line)
        if simple_match:
            is_header = True
            if current_record:
                records.append(current_record)
                current_record = {}

            current_record['id'] = simple_match.group(1)
            current_record['class_family'] = ''
            current_record['chain'] = '-'
            current_record['protein_name'] = simple_match.group(2).strip()
            current_record['organism'] = simple_match.group(3) if simple_match.group(3) else ''
            current_record['sequence'] = ''
            continue

        # Если это не заголовок и у нас есть текущая запись, добавляем последовательность
        if not is_header and current_record:
            # Проверяем, что это последовательность аминокислот
            clean_line = ''.join(line.split())
            if clean_line and re.match(r'^[ACDEFGHIKLMNPQRSTVWY-]+$', clean_line, re.IGNORECASE):
                current_record['sequence'] += clean_line

    # Добавляем последнюю запись
    if current_record:
        records.append(current_record)

    print(f"Всего распарсено записей: {len(records)}")

    # Показываем первые несколько записей для проверки
    if records:
        print("\nПример первых записей:")
        for i, rec in enumerate(records[:3]):
            print(f"{i + 1}. ID: {rec.get('id', '')}, "
                  f"Class: {rec.get('class_family', '')}, "
                  f"Seq_len: {len(rec.get('sequence', ''))}")

    return records


def extract_families_info(records):
    """
    Извлекает информацию о семействах
    """
    families = {}
    superfamilies = {}

    for record in records:
        class_family = record.get('class_family', '')
        if class_family:
            parts = class_family.split('.')
            if len(parts) >= 3:
                superfamily_key = '.'.join(parts[:2])
                family_key = '.'.join(parts[:3])

                if superfamily_key not in superfamilies:
                    superfamilies[superfamily_key] = []
                superfamilies[superfamily_key].append(record['id'])

                if family_key not in families:
                    families[family_key] = []
                families[family_key].append(record['id'])

    return families, superfamilies


def create_csv_database(input_file, output_file):
    """
    Создает CSV-файл с данными о гомологах
    """
    # Извлекаем текст из PDF
    print("Извлечение текста из PDF...")
    content = extract_text_from_pdf(input_file)

    if not content:
        raise ValueError("Не удалось извлечь текст из PDF")

    # Сохраняем извлеченный текст для отладки
    debug_file = input_file.parent / 'extracted_text_debug.txt'
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📄 Извлеченный текст сохранен в {debug_file}")

    # Парсим записи
    print("\nПарсинг записей...")
    records = parse_fasta_like_text(content)

    if not records:
        print("\n⚠️ ВНИМАНИЕ: Не найдено ни одной записи!")
        print("Проверьте файл extracted_text_debug.txt")
        print("Возможно, структура PDF отличается от ожидаемой.")

        # Показываем первые несколько строк текста
        print("\nПервые 500 символов извлеченного текста:")
        print(content[:500])

        return [], {}, {}

    # Извлекаем информацию о семействах
    families, superfamilies = extract_families_info(records)

    # Создаем CSV
    print(f"\nСоздание CSV файла...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'id', 'class_family', 'superfamily', 'family',
            'chain', 'protein_name', 'organism', 'sequence_length',
            'sequence'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            class_family = record.get('class_family', '')
            parts = class_family.split('.')
            superfamily = '.'.join(parts[:2]) if len(parts) >= 2 else ''
            family = '.'.join(parts[:3]) if len(parts) >= 3 else ''

            writer.writerow({
                'id': record.get('id', ''),
                'class_family': class_family,
                'superfamily': superfamily,
                'family': family,
                'chain': record.get('chain', ''),
                'protein_name': record.get('protein_name', ''),
                'organism': record.get('organism', ''),
                'sequence_length': len(record.get('sequence', '')),
                'sequence': record.get('sequence', '')
            })

    print(f"✅ CSV-файл создан: {output_file}")
    return records, families, superfamilies


def create_statistics_report(records, families, superfamilies, output_file):
    """
    Создает отчет со статистикой
    """
    stats = {
        'total_sequences': len(records),
        'total_families': len(families),
        'total_superfamilies': len(superfamilies),
        'families_distribution': {},
        'organisms': {},
        'sequences_without_class': 0
    }

    for family, members in families.items():
        stats['families_distribution'][family] = len(members)

    for record in records:
        if not record.get('class_family'):
            stats['sequences_without_class'] += 1

        organism = record.get('organism', 'Unknown')
        if organism:
            stats['organisms'][organism] = stats['organisms'].get(organism, 0) + 1
        else:
            stats['organisms']['Unknown'] = stats['organisms'].get('Unknown', 0) + 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== СТАТИСТИКА БАЗЫ ДАННЫХ ГОМОЛОГОВ ===\n\n")
        f.write(f"Всего последовательностей: {stats['total_sequences']}\n")
        f.write(f"Всего семейств: {stats['total_families']}\n")
        f.write(f"Всего суперсемейств: {stats['total_superfamilies']}\n")
        f.write(f"Без классификации: {stats['sequences_without_class']}\n\n")

        if stats['families_distribution']:
            f.write("--- Распределение по семействам ---\n")
            for family, count in sorted(stats['families_distribution'].items(),
                                        key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"  {family}: {count}\n")

        if stats['organisms']:
            f.write("\n--- Топ-20 организмов ---\n")
            for organism, count in sorted(stats['organisms'].items(),
                                          key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"  {organism}: {count}\n")

    return stats


def main():
    project_root = Path(__file__).parent
    input_file = project_root / 'Supplementary Material S1.pdf'
    output_csv = project_root / 'homologs_database.csv'
    output_report = project_root / 'database_statistics.txt'

    print(f"Обработка файла: {input_file}")
    print(f"Текущая директория: {project_root}\n")

    if not input_file.exists():
        print(f"❌ Файл не найден: {input_file}")
        print("Убедитесь, что файл 'Supplementary Material S1.pdf' находится в корне проекта.")
        return

    try:
        records, families, superfamilies = create_csv_database(input_file, output_csv)

        if not records:
            print("\n⚠️ Не удалось распарсить записи. Проверьте extracted_text_debug.txt")
            return

        stats = create_statistics_report(records, families, superfamilies, output_report)

        print(f"\n📊 Отчет сохранен: {output_report}")
        print(f"\n📈 Статистика:")
        print(f"  - Всего последовательностей: {stats['total_sequences']}")
        print(f"  - Всего семейств: {stats['total_families']}")
        print(f"  - Всего суперсемейств: {stats['total_superfamilies']}")
        print(f"  - Без классификации: {stats['sequences_without_class']}")

        # Проверяем результат
        if stats['total_sequences'] > 0:
            df = pd.read_csv(output_csv)
            print(f"\n📝 Первые 5 записей:")
            print(df[['id', 'class_family', 'protein_name', 'sequence_length']].head())

            # Проверяем длины последовательностей
            print(f"\n📊 Статистика по длинам последовательностей:")
            print(f"  Средняя длина: {df['sequence_length'].mean():.0f}")
            print(f"  Мин. длина: {df['sequence_length'].min()}")
            print(f"  Макс. длина: {df['sequence_length'].max()}")
        else:
            print("\n⚠️ В CSV нет записей. Проверьте парсинг.")

    except ImportError as e:
        print(f"❌ Ошибка: {e}")
        print("\nУстановите необходимые библиотеки:")
        print("pip install pdfplumber pandas")
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()