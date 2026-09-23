#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты генератора синтетических данных (scripts/generate_data.py).

Функции генератора вызываются напрямую с уменьшенными объёмами и пишут во
временный каталог pytest (tmp_path), поэтому весь набор тестов выполняется
за секунды. Проверяются свойства, критичные для дисциплины:

1) детерминированность — одинаковый вариант даёт побайтно одинаковые файлы;
2) корректность logs_answers.json — внедрённые аномалии действительно
   присутствуют в журнале на указанных позициях (самопроверка ДЗ 3);
3) целостность embeddings_ground_truth.json — все указанные id существуют
   в CSV-файлах (самопроверка ДЗ 4);
4) операции ЛР 2 опустошают структуры и пересекают границы роста ёмкости,
   а заготовка ЛР 2 читает их и без решения завершается отчётом self_check.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_data as gd  # noqa: E402

SEED = gd.BASE_SEED + 7  # вариант 7 — произвольный, важна лишь фиксация

# Уменьшенные объёмы для быстрых тестов
SMALL_LOGS = dict(n_events=1_500, n_bursts=2, burst_size=25, n_signatures=2)
SMALL_EMB = dict(n_resumes=60, n_vacancies=60, dim=12, k=5, noise=0.5)
SMALL_TEXTS = dict(length=4_000, n_patterns=6)
SMALL_OPS = dict(n_ops=20_000, max_size=300)

LAB02_STARTER = (REPO_ROOT / "M1-intro-and-basic-structures" / "attachments"
                 / "lab02-recursion-structures-starter.py")


def _sha256_by_name(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _read_log_rows(out_dir: Path) -> list[list[str]]:
    lines = (out_dir / "logs_events.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp;process;event_type;details"
    rows = [line.split(";") for line in lines[1:]]
    assert all(len(row) == 4 for row in rows), "в строке журнала должно быть 4 поля"
    return rows


def _read_csv_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split(",", 1)[0] for line in lines[1:]]


def test_same_variant_gives_identical_bytes(tmp_path: Path) -> None:
    """Повторная генерация с тем же seed совпадает побайтно (logs и embeddings)."""
    hashes = []
    for sub in ("run1", "run2"):
        out_dir = tmp_path / sub
        out_dir.mkdir()
        paths = gd.generate_logs(out_dir, SEED, **SMALL_LOGS)
        paths += gd.generate_embeddings(out_dir, SEED, **SMALL_EMB)
        paths += gd.generate_ops(out_dir, SEED, **SMALL_OPS)
        hashes.append(_sha256_by_name(paths))
    assert hashes[0] == hashes[1]


def test_different_variants_give_different_bytes(tmp_path: Path) -> None:
    """Смена варианта меняет содержимое журнала."""
    hashes = []
    for sub, seed in (("v1", SEED), ("v2", SEED + 1)):
        out_dir = tmp_path / sub
        out_dir.mkdir()
        hashes.append(_sha256_by_name(gd.generate_logs(out_dir, seed, **SMALL_LOGS)))
    assert hashes[0]["logs_events.csv"] != hashes[1]["logs_events.csv"]


def test_logs_answers_describe_real_anomalies(tmp_path: Path) -> None:
    """logs_answers.json — валидный JSON, аномалии присутствуют в журнале."""
    gd.generate_logs(tmp_path, SEED, **SMALL_LOGS)
    rows = _read_log_rows(tmp_path)
    answers = json.loads((tmp_path / "logs_answers.json").read_text(encoding="utf-8"))

    bursts = answers["bursts"]
    assert len(bursts) == SMALL_LOGS["n_bursts"]
    for burst in bursts:
        segment = rows[burst["start_index"]:burst["end_index"] + 1]
        assert len(segment) == burst["count"]
        assert {row[2] for row in segment} == {burst["event_type"]}
        t_first = datetime.fromisoformat(segment[0][0])
        t_last = datetime.fromisoformat(segment[-1][0])
        assert (t_last - t_first).total_seconds() <= burst["window_seconds"]
        assert segment[0][0] == burst["t_start"]
        assert segment[-1][0] == burst["t_end"]

    signatures = answers["signatures"]
    assert len(signatures) == SMALL_LOGS["n_signatures"]
    for signature in signatures:
        assert len(signature["indices"]) == len(signature["substrings"])
        assert signature["indices"] == sorted(signature["indices"])
        for index, substring in zip(signature["indices"], signature["substrings"]):
            assert substring in rows[index][3], (
                f"подстрока сигнатуры {signature['name']!r} не найдена "
                f"в details строки {index}"
            )


def test_embeddings_ground_truth_ids_exist(tmp_path: Path) -> None:
    """Все id в embeddings_ground_truth.json существуют в CSV-файлах."""
    gd.generate_embeddings(tmp_path, SEED, **SMALL_EMB)
    resume_ids = _read_csv_ids(tmp_path / "embeddings_resumes.csv")
    vacancy_ids = set(_read_csv_ids(tmp_path / "embeddings_vacancies.csv"))
    ground_truth = json.loads(
        (tmp_path / "embeddings_ground_truth.json").read_text(encoding="utf-8"))

    matches = ground_truth["matches"]
    assert set(matches) == set(resume_ids)
    for resume_id, vacancies in matches.items():
        assert vacancies, f"для {resume_id} не указано ни одной вакансии"
        assert set(vacancies) <= vacancy_ids


def test_manifest_records_variant_and_merges_sets(tmp_path: Path) -> None:
    """manifest.json хранит вариант и накапливает наборы; смена варианта его обнуляет."""
    arrays = gd.generate_arrays(tmp_path, SEED, sizes=(100,))
    gd.write_manifest(tmp_path, 7, SEED, ["arrays"], arrays)
    manifest = json.loads((tmp_path / gd.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["variant"] == 7
    assert manifest["seed"] == SEED
    assert manifest["sets"] == ["arrays"]
    assert "arrays_random_100.txt" in manifest["files"]

    # тот же вариант, другой набор — сведения накапливаются
    pairs = gd.generate_pairs(tmp_path, SEED, n_keys=50)
    gd.write_manifest(tmp_path, 7, SEED, ["pairs"], pairs)
    manifest = json.loads((tmp_path / gd.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["sets"] == ["arrays", "pairs"]
    assert {"arrays_random_100.txt", "pairs_keys.txt"} <= set(manifest["files"])

    # другой вариант — манифест начинается заново
    gd.write_manifest(tmp_path, 8, SEED + 1, ["pairs"], pairs)
    manifest = json.loads((tmp_path / gd.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["variant"] == 8
    assert manifest["sets"] == ["pairs"]
    assert "arrays_random_100.txt" not in manifest["files"]


def test_patterns_occur_in_texts(tmp_path: Path) -> None:
    """Каждый шаблон из patterns.txt встречается в своём тексте."""
    gd.generate_texts(tmp_path, SEED, **SMALL_TEXTS)
    texts = {
        name: (tmp_path / name).read_text(encoding="utf-8")
        for name in ("texts_small_alphabet.txt", "texts_natural.txt")
    }
    lines = (tmp_path / "patterns.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == SMALL_TEXTS["n_patterns"]
    for line in lines:
        name, pattern = line.split("\t")
        assert 3 <= len(pattern) <= 30
        assert pattern in texts[name]


def test_ops_exercise_linear_structures(tmp_path: Path) -> None:
    """Операции ЛР 2 корректны по формату, опустошают структуры и переходят границы ёмкости."""
    gd.generate_ops(tmp_path, SEED, **SMALL_OPS)
    stack: list[int] = []
    deque: collections.deque[int] = collections.deque()
    references = {
        "ops_stack.txt": (gd.STACK_OPS, stack,
                          {"push": stack.append, "pop": stack.pop, "peek": lambda: stack[-1]}),
        "ops_deque.txt": (gd.DEQUE_OPS, deque,
                          {"push_front": deque.appendleft, "push_back": deque.append,
                           "pop_front": deque.popleft, "pop_back": deque.pop}),
    }
    for name, (arity, ref, methods) in references.items():
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) == SMALL_OPS["n_ops"]
        emptied = raised = max_size = 0
        for line in lines:
            op, *args = line.split()
            assert op in arity and len(args) == arity[op], f"{name}: строка {line!r}"
            before = len(ref)
            try:
                methods[op](*map(int, args))
            except IndexError:
                raised += 1
            emptied += before > 0 and not ref
            max_size = max(max_size, len(ref))
        assert {line.split()[0] for line in lines} == set(arity)
        assert emptied >= 10, f"{name}: структура опустошается лишь {emptied} раз"
        assert raised > 0, f"{name}: нет операций над пустой структурой"
        assert max_size > 64, f"{name}: размер не превышает {max_size}"

    text = (tmp_path / "ops_append_sizes.txt").read_text(encoding="utf-8")
    sizes = [int(line) for line in text.split()]
    lo, hi = gd.APPEND_SIZE_RANGE
    assert sizes == sorted(set(sizes)) and lo <= sizes[0] and sizes[-1] <= hi
    assert any(n & (n - 1) == 0 and n + 1 in sizes for n in sizes), "нет пары 2**k, 2**k + 1"


def test_lab02_starter_reads_variant_ops(tmp_path: Path) -> None:
    """Заготовка ЛР 2 читает ops варианта и сверяет вариант; без решения — отчёт, не зависание."""
    created = gd.generate_ops(tmp_path, SEED, **SMALL_OPS)
    gd.write_manifest(tmp_path, 7, SEED, ["ops"], created)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def run(variant: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-O", str(LAB02_STARTER), "--variant", str(variant),
             "--data", str(tmp_path), "--out", str(tmp_path)],
            capture_output=True, encoding="utf-8", env=env, timeout=120)

    result = run(7)
    assert result.returncode == 1
    assert "Данные варианта 7" in result.stdout
    assert result.stdout.count("не реализовано") == 5, result.stdout
    assert "Traceback" not in result.stdout + result.stderr

    result = run(8)
    assert result.returncode == 1
    assert "сгенерированы для варианта 7" in result.stderr
