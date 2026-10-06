#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 2. Рекурсивные функции; динамический массив, стек и дек — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-02 и правилами использования генеративного ИИ
(docs/ai-verification.md). Каркас загрузки данных, самопроверки, замеров
и построения графиков уже готов — учебная задача в реализации структур
и в объяснении результатов.

Работа выполняется на данных СВОЕГО варианта. Перед первым запуском их нужно
сгенерировать (из корня репозитория курса):

    python scripts/generate_data.py --variant N --only ops

Запуск (N — ваш номер варианта):

    python lab02-recursion-structures-starter.py --variant N

Если заготовка скопирована в личный репозиторий и каталог с данными не находится
автоматически, укажите его явно:

    python lab02-recursion-structures-starter.py --variant N --data ~/data-structures-and-algorithms/data/generated
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path


FRONT_SIZES = (1_000, 3_000, 10_000, 30_000, 100_000)
FRONT_OPS = 1_000
FIB_NS = (5, 10, 15, 20, 25, 30)
REPEATS = 5

CALLS = {"fib_naive": 0, "fib_memo": 0}


# ============================================================
# 1. РЕКУРСИЯ
# ============================================================

def factorial(n: int) -> int:
    """Факториал n >= 0 рекурсивно."""
    if n < 0:
        raise ValueError("factorial определён только для n >= 0")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fib_naive(n: int) -> int:
    """Число Фибоначчи наивной рекурсией."""
    if n < 0:
        raise ValueError("fib_naive определена только для n >= 0")

    CALLS["fib_naive"] += 1

    if n <= 1:
        return n

    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n: int, memo: dict[int, int] | None = None) -> int:
    """Число Фибоначчи с мемоизацией."""
    if n < 0:
        raise ValueError("fib_memo определена только для n >= 0")

    CALLS["fib_memo"] += 1

    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        memo[n] = n
        return n

    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)

    return memo[n]


def hanoi(
    n: int,
    src: str = "A",
    dst: str = "C",
    aux: str = "B",
    moves: list[tuple[str, str]] | None = None,
) -> int:
    """Ханойские башни."""

    if n < 0:
        raise ValueError("число дисков n должно быть неотрицательным")

    if moves is None:
        moves = []

    if n == 0:
        return 0

    left = hanoi(n - 1, src, aux, dst, moves)

    moves.append((src, dst))

    right = hanoi(n - 1, aux, dst, src, moves)

    return left + 1 + right


# ============================================================
# 2. DYNAMIC ARRAY
# ============================================================

class DynamicArray:
    """Динамический массив с увеличением ёмкости в 2 раза."""

    INITIAL_CAPACITY = 4

    def __init__(self) -> None:
        self._capacity = self.INITIAL_CAPACITY
        self._size = 0
        self._buffer: list = [None] * self._capacity
        self.copies = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def _grow(self) -> None:
        """Увеличение ёмкости в 2 раза."""
        new_capacity = self._capacity * 2
        new_buffer = [None] * new_capacity

        for i in range(self._size):
            new_buffer[i] = self._buffer[i]

        self.copies += self._size
        self._buffer = new_buffer
        self._capacity = new_capacity

    def append(self, value) -> None:
        """Добавить элемент в конец."""
        if self._size == self._capacity:
            self._grow()

        self._buffer[self._size] = value
        self._size += 1

    def pop(self):
        """Удалить последний элемент."""
        if self._size == 0:
            raise IndexError("pop из пустого DynamicArray")

        index = self._size - 1
        value = self._buffer[index]

        self._buffer[index] = None
        self._size -= 1

        return value

    def get(self, index: int):
        """Получить элемент."""
        if index < 0 or index >= self._size:
            raise IndexError(
                f"индекс {index} вне диапазона 0..{self._size - 1}"
            )

        return self._buffer[index]

    def set(self, index: int, value) -> None:
        """Изменить элемент."""
        if index < 0 or index >= self._size:
            raise IndexError(
                f"индекс {index} вне диапазона 0..{self._size - 1}"
            )

        self._buffer[index] = value


# ============================================================
# 3. STACK
# ============================================================

class Stack:
    """Стек LIFO на базе DynamicArray."""

    def __init__(self) -> None:
        self._data = DynamicArray()

    def __len__(self) -> int:
        return len(self._data)

    def push(self, value) -> None:
        self._data.append(value)

    def pop(self):
        return self._data.pop()

    def peek(self):
        if len(self._data) == 0:
            raise IndexError("peek из пустого стека")

        return self._data.get(len(self._data) - 1)


# ============================================================
# 4. DEQUE
# ============================================================

class _Node:
    """Узел двусвязного списка."""

    __slots__ = ("value", "prev", "next")

    def __init__(self, value, prev=None, next=None) -> None:
        self.value = value
        self.prev = prev
        self.next = next


class Deque:
    """Дек на базе двусвязного списка."""

    def __init__(self) -> None:
        self._head: _Node | None = None
        self._tail: _Node | None = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push_front(self, value) -> None:
        node = _Node(value, prev=None, next=self._head)

        if self._head is None:
            self._head = self._tail = node
        else:
            self._head.prev = node
            self._head = node

        self._size += 1

    def push_back(self, value) -> None:
        node = _Node(value, prev=self._tail, next=None)

        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node

        self._size += 1

    def pop_front(self):
        if self._head is None:
            raise IndexError("pop_front из пустого дека")

        node = self._head
        new_head = node.next

        if new_head is None:
            self._head = self._tail = None
        else:
            new_head.prev = None
            self._head = new_head

        node.next = None
        self._size -= 1

        return node.value

    def pop_back(self):
        if self._tail is None:
            raise IndexError("pop_back из пустого дека")

        node = self._tail
        new_tail = node.prev

        if new_tail is None:
            self._head = self._tail = None
        else:
            new_tail.next = None
            self._tail = new_tail

        node.prev = None
        self._size -= 1

        return node.value


# ============================================================
# 5. РАБОТА С ДАННЫМИ ВАРИАНТА
# ============================================================

Op = tuple[str, int | None]


def find_data_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise SystemExit(f"Каталог не найден: {explicit}")
        return explicit

    candidates = []

    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (base, *base.parents):
            candidates.append(parent / "data" / "generated")

    for path in candidates:
        if path.is_dir():
            return path

    raise SystemExit(
        "Не найден каталог data/generated с данными варианта.\n"
        "Сначала выполните:\n"
        "python scripts/generate_data.py --variant 14 --only ops"
    )


def check_variant(data_dir: Path, variant: int) -> None:
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.is_file():
        print(
            f"ВНИМАНИЕ: в {data_dir} нет manifest.json"
        )
        return

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    actual = manifest.get("variant")

    if actual != variant:
        raise SystemExit(
            f"Данные относятся к варианту {actual}, "
            f"а выбран вариант {variant}."
        )

    print(
        f"Данные варианта {variant} "
        f"(seed={manifest.get('seed')})"
    )


def read_data_lines(data_dir: Path, name: str) -> list[str]:
    path = data_dir / name

    if not path.is_file():
        raise SystemExit(
            f"Не найден файл данных: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).splitlines()


def load_append_sizes(data_dir: Path) -> list[int]:
    return [
        int(line)
        for line in read_data_lines(
            data_dir,
            "ops_append_sizes.txt"
        )
    ]


def load_ops(
    data_dir: Path,
    kind: str,
) -> list[Op]:

    ops: list[Op] = []

    for line in read_data_lines(
        data_dir,
        f"ops_{kind}.txt"
    ):
        name, *arg = line.split()

        ops.append(
            (
                name,
                int(arg[0]) if arg else None
            )
        )

    return ops


# ============================================================
# 6. ПРОВЕРКИ
# ============================================================

class SelfCheckError(Exception):
    pass


def expect(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise SelfCheckError(message)


def expect_index_error(
    call,
    what: str,
) -> None:

    try:
        call()
    except IndexError:
        return

    raise SelfCheckError(
        f"{what}: ожидался IndexError"
    )


def stack_methods(st: Stack) -> dict:
    return {
        "push": st.push,
        "pop": st.pop,
        "peek": st.peek,
        "len": st.__len__,
    }


def list_stack_methods(a: list) -> dict:
    return {
        "push": a.append,
        "pop": a.pop,
        "peek": lambda: a[-1],
        "len": a.__len__,
    }


def deque_methods(dq: Deque) -> dict:
    return {
        "push_front": dq.push_front,
        "push_back": dq.push_back,
        "pop_front": dq.pop_front,
        "pop_back": dq.pop_back,
        "len": dq.__len__,
    }


def std_deque_methods(d: collections.deque) -> dict:
    return {
        "push_front": d.appendleft,
        "push_back": d.append,
        "pop_front": d.popleft,
        "pop_back": d.pop,
        "len": d.__len__,
    }


def apply_op(
    methods: dict,
    op: str,
    arg: int | None,
):

    try:
        if arg is None:
            return methods[op]()

        return methods[op](arg)

    except IndexError:
        return "IndexError"


def compare_with_reference(
    ops: list[Op],
    own: dict,
    ref: dict,
    label: str,
) -> None:

    for i, (op, arg) in enumerate(
        ops,
        start=1,
    ):

        call = (
            op
            if arg is None
            else f"{op} {arg}"
        )

        got = apply_op(
            own,
            op,
            arg,
        )

        expected = apply_op(
            ref,
            op,
            arg,
        )

        expect(
            got == expected,
            f"{label}, операция №{i} ({call}): "
            f"получено {got!r}, "
            f"эталон {expected!r}",
        )

        expect(
            own["len"]()
            == ref["len"](),
            f"{label}, операция №{i}: "
            f"не совпадает размер",
        )


def random_ops(
    rng: random.Random,
    pushes: tuple[str, ...],
    others: tuple[str, ...],
    count: int,
    max_size: int,
) -> list[Op]:

    ops = []
    size = 0

    for _ in range(count):

        op = rng.choice(
            pushes + others
        )

        if (
            op in pushes
            and size >= max_size
        ):
            op = rng.choice(others)

        if op in pushes:
            ops.append(
                (
                    op,
                    rng.randrange(100)
                )
            )
            size += 1

        else:
            ops.append(
                (
                    op,
                    None
                )
            )

            if op.startswith("pop"):
                size = max(
                    0,
                    size - 1
                )

    return ops


def expected_capacity(size: int) -> int:
    cap = DynamicArray.INITIAL_CAPACITY

    while cap < size:
        cap *= 2

    return cap


def expected_copies(size: int) -> int:
    copies = 0
    cap = DynamicArray.INITIAL_CAPACITY

    while cap < size:
        copies += cap
        cap *= 2

    return copies


def check_dynamic_invariant(
    arr: DynamicArray,
) -> None:

    expect(
        0 <= arr._size <= arr._capacity,
        "DynamicArray: нарушен size <= capacity",
    )

    expect(
        len(arr._buffer)
        == arr._capacity,
        "DynamicArray: длина буфера "
        "не совпадает с capacity",
    )


# ============================================================
# 7. ПРОВЕРКА РЕКУРСИИ
# ============================================================

def check_recursion() -> None:

    for n in range(21):
        got = factorial(n)

        expect(
            got == math.factorial(n),
            f"factorial({n}) некорректен",
        )

    fib = [0, 1]

    while len(fib) <= 90:
        fib.append(
            fib[-1] + fib[-2]
        )

    for n in range(21):
        got = fib_naive(n)

        expect(
            got == fib[n],
            f"fib_naive({n}) некорректен",
        )

    CALLS["fib_naive"] = 0

    fib_naive(20)

    expect(
        CALLS["fib_naive"] >= fib[20],
        "fib_naive: слишком мало вызовов",
    )

    for n in (
        0,
        1,
        2,
        10,
        30,
        90,
    ):

        CALLS["fib_memo"] = 0

        got = fib_memo(n)

        expect(
            got == fib[n],
            f"fib_memo({n}) некорректен",
        )

        expect(
            CALLS["fib_memo"]
            <= 2 * n + 1,
            f"fib_memo({n}): слишком много вызовов",
        )

    for n, towers_names in (
        (0, "ACB"),
        (1, "ACB"),
        (2, "ACB"),
        (3, "XZY"),
        (6, "ACB"),
        (7, "PQR"),
    ):

        src = towers_names[0]
        dst = towers_names[1]
        aux = towers_names[2]

        moves = []

        count = hanoi(
            n,
            src,
            dst,
            aux,
            moves,
        )

        expected = 2 ** n - 1

        expect(
            count == expected,
            f"hanoi({n}) вернул {count}"
        )

        expect(
            len(moves) == count,
            "неправильное число ходов",
        )


# ============================================================
# 8. ПРОВЕРКА DYNAMIC ARRAY
# ============================================================

def check_dynamic_array() -> None:

    arr = DynamicArray()

    expect(
        len(arr) == 0,
        "пустой DynamicArray должен иметь len=0",
    )

    expect_index_error(
        lambda: arr.get(0),
        "get(0)"
    )

    expect_index_error(
        lambda: arr.set(0, 1),
        "set(0)"
    )

    expect_index_error(
        arr.pop,
        "pop()"
    )

    ref = []

    for i in range(100):

        value = i * i

        arr.append(value)
        ref.append(value)

        check_dynamic_invariant(arr)

        expect(
            len(arr) == len(ref),
            "неверный размер массива"
        )

        expect(
            arr.capacity
            == expected_capacity(
                len(ref)
            ),
            "неправильная ёмкость"
        )

    for i, value in enumerate(ref):

        expect(
            arr.get(i) == value,
            f"get({i}) некорректен",
        )

    arr.set(0, -1)
    arr.set(99, -99)

    expect(
        arr.get(0) == -1,
        "set(0) не работает"
    )

    expect(
        arr.get(99) == -99,
        "set(99) не работает"
    )

    while ref:

        got = arr.pop()
        expected = ref.pop()

        expect(
            got == expected,
            "pop() возвращает неверное значение",
        )

        expect(
            len(arr) == len(ref),
            "неверный размер после pop",
        )

        check_dynamic_invariant(arr)

    expect_index_error(
        arr.pop,
        "pop() пустого массива"
    )


# ============================================================
# 9. ПРОВЕРКА STACK
# ============================================================

def check_stack() -> None:

    st = Stack()

    expect(
        len(st) == 0,
        "пустой стек"
    )

    expect_index_error(
        st.pop,
        "pop() пустого стека"
    )

    expect_index_error(
        st.peek,
        "peek() пустого стека"
    )

    values = list(range(20))

    for x in values:

        st.push(x)

        expect(
            st.peek() == x,
            "peek() должен возвращать вершину",
        )

    for expected in reversed(values):

        got = st.pop()

        expect(
            got == expected,
            "нарушен принцип LIFO",
        )


# ============================================================
# 10. ПРОВЕРКА DEQUE
# ============================================================

def check_deque_invariant(
    dq: Deque,
) -> None:

    expect(
        dq._size >= 0,
        "Deque: отрицательный размер"
    )

    if dq._size == 0:

        expect(
            dq._head is None
            and dq._tail is None,
            "пустой deque должен иметь "
            "head=tail=None",
        )

        return

    expect(
        dq._head is not None
        and dq._tail is not None,
        "непустой deque должен иметь "
        "head и tail",
    )

    expect(
        dq._head.prev is None,
        "head.prev должен быть None",
    )

    expect(
        dq._tail.next is None,
        "tail.next должен быть None",
    )


def check_deque() -> None:

    dq = Deque()

    expect_index_error(
        dq.pop_front,
        "pop_front()"
    )

    expect_index_error(
        dq.pop_back,
        "pop_back()"
    )

    dq.push_front(1)
    dq.push_back(2)
    dq.push_front(0)

    check_deque_invariant(dq)

    expect(
        dq.pop_front() == 0,
        "pop_front() ошибка"
    )

    expect(
        dq.pop_back() == 2,
        "pop_back() ошибка"
    )

    expect(
        dq.pop_front() == 1,
        "pop_front() ошибка"
    )

    expect(
        len(dq) == 0,
        "deque должен быть пустым"
    )

    own = Deque()
    ref = collections.deque()

    ops = random_ops(
        random.Random(2),
        ("push_front", "push_back"),
        ("pop_front", "pop_back"),
        5000,
        12,
    )

    compare_with_reference(
        ops,
        deque_methods(own),
        std_deque_methods(ref),
        "Deque",
    )


# ============================================================
# 11. ПРОВЕРКА ОПЕРАЦИЙ ВАРИАНТА
# ============================================================

def check_variant_ops(
    stack_ops: list[Op],
    deque_ops: list[Op],
) -> None:

    compare_with_reference(
        stack_ops,
        stack_methods(Stack()),
        list_stack_methods([]),
        "ops_stack.txt",
    )

    compare_with_reference(
        deque_ops,
        deque_methods(Deque()),
        std_deque_methods(
            collections.deque()
        ),
        "ops_deque.txt",
    )


def self_check(
    stack_ops: list[Op],
    deque_ops: list[Op],
) -> None:

    checks = (
        ("рекурсия", check_recursion),
        ("DynamicArray", check_dynamic_array),
        ("Stack", check_stack),
        ("Deque", check_deque),
        (
            "операции варианта",
            lambda: check_variant_ops(
                stack_ops,
                deque_ops
            ),
        ),
    )

    failed = 0

    print("self_check:")

    for name, check in checks:

        try:
            check()

        except SelfCheckError as err:

            print(
                f"  {name}: ОШИБКА — {err}"
            )

            failed += 1

        except Exception:

            print(
                f"  {name}: ИСКЛЮЧЕНИЕ"
            )

            traceback.print_exc(
                file=sys.stdout
            )

            failed += 1

        else:

            print(
                f"  {name}: OK"
            )

    if failed:
        raise SystemExit(
            f"self_check: ошибок — {failed}"
        )

    print("self_check: OK")


# ============================================================
# 12. БЕНЧМАРКИ
# ============================================================

def bench(call) -> float:

    call()

    times = []

    for _ in range(REPEATS):

        t0 = time.perf_counter()

        call()

        times.append(
            time.perf_counter() - t0
        )

    return statistics.median(times)


def log_log_slope(
    points: list[tuple[int, float]]
) -> float:

    xs = [
        math.log10(n)
        for n, _ in points
    ]

    ys = [
        math.log10(t)
        for _, t in points
    ]

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)

    return (
        sum(
            (x - mx) * (y - my)
            for x, y in zip(xs, ys)
        )
        /
        sum(
            (x - mx) ** 2
            for x in xs
        )
    )


def appends(
    add,
    n: int,
) -> None:

    for i in range(n):
        add(i)


def steady_ops(
    add,
    remove,
    k: int = FRONT_OPS,
) -> None:

    for i in range(k):
        add(i)
        remove()


def replay(
    ops: list[Op],
    methods: dict,
) -> None:

    for op, arg in ops:

        try:

            if arg is None:
                methods[op]()
            else:
                methods[op](arg)

        except IndexError:
            pass


def filled_structure(
    kind: str,
    n: int,
):

    if kind == "list":
        return list(range(n))

    if kind == "deque":
        return collections.deque(
            range(n)
        )

    dq = Deque()

    appends(
        dq.push_back,
        n
    )

    return dq


FRONT_CASES = {
    "insert": (
        (
            "list.insert(0, x) + pop()",
            "list",
            lambda a: (
                functools.partial(
                    a.insert,
                    0
                ),
                a.pop
            ),
        ),
        (
            "deque.appendleft + pop",
            "deque",
            lambda d: (
                d.appendleft,
                d.pop
            ),
        ),
        (
            "Deque.push_front + pop_back",
            "Deque",
            lambda d: (
                d.push_front,
                d.pop_back
            ),
        ),
    ),
    "remove": (
        (
            "list.append + pop(0)",
            "list",
            lambda a: (
                a.append,
                functools.partial(
                    a.pop,
                    0
                ),
            ),
        ),
        (
            "deque.append + popleft",
            "deque",
            lambda d: (
                d.append,
                d.popleft
            ),
        ),
        (
            "Deque.push_back + pop_front",
            "Deque",
            lambda d: (
                d.push_back,
                d.pop_front
            ),
        ),
    ),
}


FRONT_TITLES = {
    "insert": "Вставка в начало",
    "remove": "Удаление из начала",
}


def run_benchmarks(
    append_sizes: list[int],
    stack_ops: list[Op],
    deque_ops: list[Op],
) -> dict:

    results = {}

    print(
        "\nЧисло вызовов: "
        "наивная рекурсия против мемоизации"
    )

    for n in FIB_NS:

        CALLS["fib_naive"] = 0
        CALLS["fib_memo"] = 0

        value = fib_naive(n)

        fib_memo(n)

        print(
            f"  n={n:>2} "
            f"F(n)={value:>6} "
            f"fib_naive: "
            f"{CALLS['fib_naive']:>7} вызовов "
            f"fib_memo: "
            f"{CALLS['fib_memo']:>2} вызовов"
        )

    print(
        "\nСерии append:"
    )

    results["append"] = []

    for n in append_sizes:

        t_own = (
            bench(
                lambda: appends(
                    DynamicArray().append,
                    n
                )
            )
            / n
        )

        t_list = (
            bench(
                lambda: appends(
                    [].append,
                    n
                )
            )
            / n
        )

        arr = DynamicArray()

        appends(
            arr.append,
            n
        )

        cost = (
            n + arr.copies
        ) / n

        results["append"].append(
            (
                n,
                t_own,
                t_list,
                cost,
            )
        )

        print(
            f"  n={n:>7} "
            f"DynamicArray={t_own:.2e} c "
            f"list={t_list:.2e} c "
            f"копирований={arr.copies}"
        )

    for key, title in FRONT_TITLES.items():

        print(
            f"\n{title}:"
        )

        results[key] = {}

        for label, kind, methods in FRONT_CASES[key]:

            points = []

            for n in FRONT_SIZES:

                structure = filled_structure(
                    kind,
                    n
                )

                add, remove = methods(
                    structure
                )

                points.append(
                    (
                        n,
                        bench(
                            lambda: steady_ops(
                                add,
                                remove
                            )
                        )
                        / FRONT_OPS,
                    )
                )

            results[key][label] = points

            print(
                f"  {label}:"
                f" наклон="
                f"{log_log_slope(points):.2f}"
            )

    return results


# ============================================================
# 13. ГРАФИКИ
# ============================================================

def plot_results(
    results: dict,
    out_dir: Path,
) -> None:

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "matplotlib не установлен."
        )

        return

    rows = results["append"]

    ns = [
        row[0]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        ns,
        [row[1] for row in rows],
        marker="o",
        label="DynamicArray.append",
    )

    ax.plot(
        ns,
        [row[2] for row in rows],
        marker="s",
        label="list.append",
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "число операций n"
    )

    ax.set_ylabel(
        "время на операцию, с"
    )

    ax.set_title(
        "Средняя стоимость append"
    )

    ax.grid(
        True,
        which="both"
    )

    ax.legend()

    fig.tight_layout()

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    append_path = (
        out_dir
        / "lab02_append.png"
    )

    fig.savefig(
        append_path,
        dpi=150
    )

    plt.close(fig)

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    for key, title in FRONT_TITLES.items():

        for label, points in results[key].items():

            ax2.plot(
                [
                    n
                    for n, _ in points
                ],
                [
                    t
                    for _, t in points
                ],
                marker="o",
                label=f"{label} ({title})",
            )

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_xlabel(
        "размер структуры n"
    )

    ax2.set_ylabel(
        "время операции, с"
    )

    ax2.set_title(
        "Операции в начале структуры"
    )

    ax2.grid(
        True,
        which="both"
    )

    ax2.legend(
        fontsize=8
    )

    fig2.tight_layout()

    front_path = (
        out_dir
        / "lab02_front.png"
    )

    fig2.savefig(
        front_path,
        dpi=150
    )

    plt.close(fig2)

    print(
        "\nГрафики сохранены:"
    )

    print(
        append_path
    )

    print(
        front_path
    )


# ============================================================
# 14. MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    data_dir = find_data_dir(
        args.data
    )

    check_variant(
        data_dir,
        args.variant
    )

    append_sizes = load_append_sizes(
        data_dir
    )

    stack_ops = load_ops(
        data_dir,
        "stack"
    )

    deque_ops = load_ops(
        data_dir,
        "deque"
    )

    self_check(
        stack_ops,
        deque_ops
    )

    results = run_benchmarks(
        append_sizes,
        stack_ops,
        deque_ops
    )

    plot_results(
        results,
        args.out
    )


if __name__ == "__main__":
    main()
