import streamlit as st
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from deap import base, creator, tools, algorithms, gp, cma, benchmarks

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    layout="wide",
    page_title="Анализ эволюционных алгоритмов",
    page_icon="🔬"
)

# --- ОПРЕДЕЛЕНИЕ ТИПОВ DEAP (делается один раз) ---
# Для Части 1
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Для Части 2 (используем тот же Individual, но с другим фитнесом)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualCMA", list, fitness=creator.FitnessMin)


# --- ФУНКЦИИ-ВЫЧИСЛИТЕЛИ (с кэшированием) ---

@st.cache_data
def run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size=50):
    """Запускает Генетический Алгоритм для Части 1."""
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_func(individual):
        return len(individual) - abs(sum(individual) - target_sum),

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    result_pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                               stats=stats, verbose=False)
    best_individual = tools.selBest(result_pop, k=1)[0]
    return best_individual, logbook

@st.cache_data
def run_cma_es(selected_func, ngen_cma, sigma, num_individuals=10):
    """Запускает CMA-ES для Части 2."""
    if selected_func == "Растригин":
        eval_func_cma = benchmarks.rastrigin
        centroid_start = [5.0] * num_individuals
    else: # Розенброк
        eval_func_cma = benchmarks.rosenbrock
        centroid_start = [0.0] * num_individuals

    strategy = cma.Strategy(centroid=centroid_start, sigma=sigma, lambda_=20)
    toolbox_cma = base.Toolbox()
    toolbox_cma.register("evaluate", eval_func_cma)
    
    hall_of_fame = tools.HallOfFame(1)
    logbook_cma = tools.Logbook()
    logbook_cma.header = strategy.fields
    
    for gen in range(ngen_cma):
        population_cma = strategy.generate(creator.IndividualCMA)
        fitnesses = toolbox_cma.map(toolbox_cma.evaluate, population_cma)
        for ind, fit in zip(population_cma, fitnesses):
            ind.fitness.values = fit
        
        strategy.update(population_cma)
        hall_of_fame.update(population_cma)
        logbook_cma.record(gen=gen, evals=len(population_cma), **strategy.getValues())

    # Добавляем min fitness в logbook для графика
    best_fitness_over_time = [min(strategy.past_f_values[i:i+20]) for i in range(len(strategy.past_f_values)-20)]
    logbook_cma.chapters["min_fitness"] = best_fitness_over_time[:ngen_cma]


    return logbook_cma, hall_of_fame

# Функции для Части 3, 4, 5 (если они будут интерактивными) можно добавить сюда

# --- ФУНКЦИИ-ОТРИСОВЩИКИ ГРАФИКОВ ---

def create_fitness_plot(logbook):
    """Создает график динамики фитнеса для Части 1."""
    fig, ax = plt.subplots(figsize=(10, 5))
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")
    
    ax.plot(gen, max_fitness, "b-", label="Максимальный фитнес")
    ax.plot(gen, avg_fitness, "r-", label="Средний фитнес")
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Значение функции приспособленности (Фитнес)")
    ax.set_title("Динамика фитнеса по поколениям")
    ax.legend(loc="lower right")
    ax.grid(True)
    return fig

# --- ФУНКЦИИ-РЕНДЕРЕРЫ ДЛЯ КАЖДОЙ ГЛАВЫ ---

def render_chapter_1():
    st.title("Часть 1: Генерация битовых образов с предопределенными параметрами")
    st.markdown("В данном разделе решается задача генерации строки из 0 и 1 (битового вектора) таким образом, чтобы количество единиц было максимально приближено к заданному значению. Для решения используется генетический алгоритм, ключевые компоненты которого описаны ниже.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Параметры генетического алгоритма")
        st.info("""
        **Целевая функция (приспособленности):** Рассчитывается как `длина строки - abs(сумма элементов - целевая сумма)`. Максимизируется.
        **Начальная популяция:** Создается случайным образом из индивидуумов, представляющих собой битовые векторы заданной длины.
        **Механизм скрещивания:** Двухточечный кроссинговер (`tools.cxTwoPoint`).
        **Механизм мутации:** Инвертирование бита (`tools.mutFlipBit`) с заданной вероятностью.
        **Механизм селекции:** Турнирная селекция (`tools.selTournament`) с размером турнира 3.
        """)
        target_sum = st.slider("Целевое значение суммы", min_value=10, max_value=100, value=45)
        n_attr = st.slider("Длина индивидуума (количество бит)", min_value=20, max_value=150, value=75)
        ngen = st.slider("Число поколений", min_value=10, max_value=200, value=60)
        cxpb = st.slider("Вероятность скрещивания (cxpb)", min_value=0.0, max_value=1.0, value=0.8)
        mutpb = st.slider("Вероятность мутации (mutpb)", min_value=0.0, max_value=1.0, value=0.1)
        run_button = st.button("Запустить генетический алгоритм")

    with col2:
        st.subheader("Результаты моделирования")
        if run_button:
            with st.spinner("Запуск эволюции..."):
                best_individual, logbook = run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb)
                fig = create_fitness_plot(logbook)
                st.pyplot(fig)
                st.subheader("Интерпретация графика динамики фитнеса")
                st.markdown("""
                Данный график является ключевым индикатором сходимости алгоритма. На нем отображаются два показателя, рассчитанные для каждого поколения:
                - **Максимальный фитнес (синяя линия):** Показывает значение функции приспособленности для лучшего индивидуума в текущей популяции. Рост этого показателя демонстрирует, что алгоритм находит все более качественные решения.
                - **Средний фитнес (красная линия):** Отражает средний уровень приспособленности всех индивидуумов в популяции. Устойчивый рост этого показателя свидетельствует о том, что вся популяция в целом эволюционирует в сторону более оптимальных решений, а не только отдельные "особи".
                
                Совместный рост обоих показателей указывает на успешный процесс обучения и сходимость популяции к решению поставленной задачи.
                """)
        else:
            st.info("Нажмите кнопку 'Запустить генетический алгоритм' для отображения результатов.")


def render_chapter_2():
    st.title("Часть 2: Визуализация хода эволюции (CMA-ES)")
    st.markdown("""
    В данном разделе анализируется работа стратегии эволюции с адаптацией матрицы ковариации (CMA-ES) — мощного алгоритма для оптимизации в непрерывном пространстве параметров. Алгоритм демонстрируется на стандартных многомерных тестовых функциях Растригина и Розенброка.
    Четыре графика ниже представляют собой "приборную панель" алгоритма, показывающую, как его внутренние параметры адаптировались в процессе поиска глобального минимума.
    """)

    selected_func = st.selectbox("Тестовая функция", ["Растригин", "Розенброк"], key="cma_func")
    ngen_cma = st.slider("Число поколений (CMA-ES)", 50, 500, 125, key="cma_ngen")
    sigma = st.slider("Начальное значение Sigma", 0.1, 10.0, 5.0, key="cma_sigma")
    run_button_cma = st.button("Запустить CMA-ES")

    if run_button_cma:
        with st.spinner("Запуск оптимизации CMA-ES..."):
            logbook_cma, hall_of_fame = run_cma_es(selected_func, ngen_cma, sigma)
            st.subheader(f"Результаты для функции {selected_func}")
            st.metric("Найденный минимум функции", f"{hall_of_fame[0].fitness.values[0]:.4f}")

            st.subheader("Анализ динамики внутренних параметров стратегии")
            tab1, tab2, tab3, tab4 = st.tabs(["Лучшее значение функции", "Значение Sigma", "Соотношение осей", "Диагональ D"])

            with tab1:
                st.subheader("График лучшего значения функции (Рисунки 1, 7)")
                fig1, ax1 = plt.subplots()
                ax1.plot(logbook_cma.chapters["min_fitness"])
                ax1.set_xlabel("Поколение")
                ax1.set_ylabel("Минимальное значение f(x)")
                ax1.set_title("Сходимость: Лучшее значение функции")
                ax1.grid(True)
                st.pyplot(fig1)
                st.markdown("""**Описание:** График демонстрирует сходимость алгоритма...""")

            with tab2:
                st.subheader("График значения Sigma (Рисунки 2, 8)")
                fig2, ax2 = plt.subplots()
                ax2.plot(logbook_cma.select('sigma'))
                ax2.set_xlabel("Поколение")
                ax2.set_ylabel("Значение Sigma")
                ax2.set_title("Адаптация размера шага: Динамика Sigma")
                ax2.grid(True)
                st.pyplot(fig2)
                st.markdown("""**Описание:** Sigma представляет собой глобальный размер шага...""")

            with tab3:
                st.subheader("График соотношения осей (Рисунки 3, 9)")
                fig3, ax3 = plt.subplots()
                ax3.semilogy(logbook_cma.select('axis_ratio'))
                ax3.set_xlabel("Поколение")
                ax3.set_ylabel("Соотношение осей (логарифмическая шкала)")
                ax3.set_title("Адаптация формы: Соотношение осей эллипсоида")
                ax3.grid(True)
                st.pyplot(fig3)
                st.markdown("""**Описание:** График показывает соотношение максимальной и минимальной длины...""")
            
            with tab4:
                st.subheader("График диагонали D (Рисунки 4, 10)")
                fig4, ax4 = plt.subplots()
                ax4.semilogy([l['diagD'] for l in logbook_cma])
                ax4.set_xlabel("Поколение")
                ax4.set_ylabel("Значения diag(D) (логарифмическая шкала)")
                ax4.set_title("Адаптация ориентации: Динамика диагонали D")
                ax4.grid(True)
                st.pyplot(fig4)
                st.markdown("""**Описание:** D содержит собственные значения матрицы ковариации C...""")


def render_placeholder_chapter(title, description):
    """Рендерер-заглушка для глав в разработке."""
    st.title(title)
    st.markdown(description)
    st.warning("Этот раздел находится в разработке.")
    st.image("https://i.imgur.com/3_d.png".replace("_", "u4I4fB"), caption="Скоро здесь будет интерактивная демонстрация!")


# --- ГЛАВНЫЙ РОУТЕР ПРИЛОЖЕНИЯ ---

st.sidebar.title("Структура отчета")
chapter = st.sidebar.radio("Выберите раздел для анализа:", [
    "Часть 1: Генерация битовых образов",
    "Часть 2: Визуализация хода эволюции (CMA-ES)",
    "Часть 3: Символическая регрессия",
    "Часть 4: Создание контроллера интеллектуального робота",
    "Часть 5: Рекомендательная система"
])

if chapter == "Часть 1: Генерация битовых образов":
    render_chapter_1()
elif chapter == "Часть 2: Визуализация хода эволюции (CMA-ES)":
    render_chapter_2()
elif chapter == "Часть 3: Символическая регрессия":
    render_placeholder_chapter(chapter, "Применение генетического программирования для поиска математического выражения, наилучшим образом аппроксимирующего заданный набор точек данных.")
elif chapter == "Часть 4: Создание контроллера интеллектуального робота":
    render_placeholder_chapter(chapter, "Использование генетического программирования для эволюционного синтеза алгоритма управления для виртуального агента.")
elif chapter == "Часть 5: Рекомендательная система":
     render_placeholder_chapter(chapter, "Реализация метода коллаборативной фильтрации типа 'user-based' с использованием коэффициента корреляции Пирсона.")
