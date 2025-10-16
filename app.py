import streamlit as st
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import operator
import math
from deap import base, creator, tools, algorithms, gp, cma, benchmarks

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    layout="wide",
    page_title="Анализ эволюционных алгоритмов",
    page_icon="🔬"
)

# --- ОПРЕДЕЛЕНИЕ ТИПОВ DEAP (делается один раз, чтобы избежать ошибок при перерисовке) ---
# Для Части 1 (Генетический алгоритм)
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

# Для Части 2 (CMA-ES)
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("IndividualCMA", list, fitness=creator.FitnessMin)
except Exception:
    pass

# Для Части 3 (Символическая регрессия)
try:
    creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
    creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMinGP)
except Exception:
    pass


# --- ФУНКЦИИ-ВЫЧИСЛИТЕЛИ (с кэшированием для производительности) ---

@st.cache_data
def run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size=50):
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
    _, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, logbook

@st.cache_data
def run_cma_es(func_name, ngen_cma, sigma, num_individuals=10):
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    eval_func_cma, centroid_start = (benchmarks.rastrigin, [5.0]*num_individuals) if func_name == "Растригин" else (benchmarks.rosenbrock, [0.0]*num_individuals)
    
    strategy = cma.Strategy(centroid=centroid_start, sigma=sigma, lambda_=20)
    toolbox_cma = base.Toolbox()
    toolbox_cma.register("evaluate", eval_func_cma)
    
    hall_of_fame = tools.HallOfFame(1)
    
    # 1. Добавлен объект статистики для сбора данных по минимуму
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    logbook_cma = tools.Logbook()
    # 2. Исправлен способ задания заголовка
    logbook_cma.header = ['gen', 'evals', 'min'] + strategy.getValues().keys()

    for gen in range(ngen_cma):
        population_cma = strategy.generate(creator.IndividualCMA)
        fitnesses = toolbox_cma.map(toolbox_cma.evaluate, population_cma)
        for ind, fit in zip(population_cma, fitnesses):
            ind.fitness.values = fit
        
        strategy.update(population_cma)
        hall_of_fame.update(population_cma)
        
        record = stats.compile(population_cma)
        logbook_cma.record(gen=gen, evals=len(population_cma), **record, **strategy.getValues())
        
    return logbook_cma, hall_of_fame


@st.cache_data
def run_symbolic_regression(ngen, pop_size):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
    pset.renameArguments(ARG0='x')
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.IndividualGP, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    def evalSymbReg(individual, points):
        func = toolbox.compile(expr=individual)
        sqerrors = ((func(x) - (2*x**3 - 3*x**2 + 4*x - 1))**2 for x in points)
        return math.fsum(sqerrors) / len(points),
    points = [x/10. for x in range(-10, 10)]
    toolbox.register("evaluate", evalSymbReg, points=points)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen, stats=stats, halloffame=hof, verbose=False)
    return hof[0], toolbox.compile(expr=hof[0])

@st.cache_data
def load_recommender_data():
    with open('ratings.json', 'r') as f:
        return json.load(f)

def pearson_score(dataset, user1, user2):
    common_movies = {item for item in dataset[user1] if item in dataset[user2]}
    if not common_movies: return 0
    sum1 = sum(dataset[user1][item] for item in common_movies)
    sum2 = sum(dataset[user2][item] for item in common_movies)
    sum1_sq = sum(pow(dataset[user1][item], 2) for item in common_movies)
    sum2_sq = sum(pow(dataset[user2][item], 2) for item in common_movies)
    p_sum = sum(dataset[user1][item] * dataset[user2][item] for item in common_movies)
    num = p_sum - (sum1 * sum2 / len(common_movies))
    den = np.sqrt((sum1_sq - pow(sum1, 2) / len(common_movies)) * (sum2_sq - pow(sum2, 2) / len(common_movies)))
    return 0 if den == 0 else num / den


# --- ФУНКЦИИ-РЕНДЕРЕРЫ ДЛЯ КАЖДОЙ ГЛАВЫ ---

def render_chapter_1():
    st.title("Часть 1: Генерация битовых образов")
    st.markdown("Демонстрация канонического генетического алгоритма на задаче оптимизации битового вектора.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Параметры алгоритма")
        target_sum = st.slider("Целевое значение суммы", 10, 100, 45, key="ch1_target")
        n_attr = st.slider("Длина индивидуума (бит)", 20, 150, 75, key="ch1_len")
        ngen = st.slider("Число поколений", 10, 200, 60, key="ch1_ngen")
        cxpb = st.slider("Вероятность скрещивания", 0.0, 1.0, 0.8, key="ch1_cxpb")
        mutpb = st.slider("Вероятность мутации", 0.0, 1.0, 0.1, key="ch1_mutpb")
    with col2:
        st.subheader("Результаты моделирования")
        best_individual, logbook = run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logbook.select("gen"), logbook.select("max"), "b-", label="Максимальный фитнес")
        ax.plot(logbook.select("gen"), logbook.select("avg"), "r-", label="Средний фитнес")
        ax.set_xlabel("Поколение"); ax.set_ylabel("Фитнес"); ax.set_title("Динамика фитнеса")
        ax.legend(loc="lower right"); ax.grid(True)
        st.pyplot(fig)

def render_chapter_2():
    st.title("Часть 2: Оптимизация с помощью CMA-ES")
    st.markdown("Анализ стратегии эволюции с адаптацией матрицы ковариации (CMA-ES).")
    st.subheader("Параметры CMA-ES (применяются к обоим тестам)")
    ngen_cma = st.slider("Число поколений", 50, 500, 125, key="ch2_ngen")
    sigma = st.slider("Начальное значение Sigma", 0.1, 10.0, 5.0, key="ch2_sigma")
    col1, col2 = st.columns(2)
    
    for col, func_name in zip([col1, col2], ["Растригин", "Розенброк"]):
        with col:
            st.header(func_name)
            logbook_cma, hall_of_fame = run_cma_es(func_name, ngen_cma, sigma)
            st.metric("Найденный минимум", f"{hall_of_fame[0].fitness.values[0]:.4f}")
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.tight_layout(pad=3.0)
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: Используем logbook для построения графика ---
            axes[0, 0].plot(logbook_cma.select('gen'), logbook_cma.select('min'))
            axes[0, 0].set_title("Лучшее значение функции"); axes[0, 0].grid(True)
            
            axes[0, 1].plot(logbook_cma.select('gen'), logbook_cma.select('sigma'))
            axes[0, 1].set_title("Значение Sigma"); axes[0, 1].grid(True)

            axes[1, 0].semilogy(logbook_cma.select('gen'), logbook_cma.select('axis_ratio'))
            axes[1, 0].set_title("Соотношение осей"); axes[1, 0].grid(True)
            
            axes[1, 1].semilogy(logbook_cma.select('gen'), [l['diagD'] for l in logbook_cma])
            axes[1, 1].set_title("Диагональ D"); axes[1, 1].grid(True)
            
            st.pyplot(fig)
            
def render_chapter_3():
    st.title("Часть 3: Символическая регрессия")
    st.markdown("Применение генетического программирования для поиска математического выражения.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Параметры ГП")
        ngen_gp = st.slider("Число поколений", 10, 100, 40, key="ch3_ngen")
        pop_size_gp = st.slider("Размер популяции", 100, 1000, 300, key="ch3_pop")
        best_ind, compiled_func = run_symbolic_regression(ngen_gp, pop_size_gp)
        st.subheader("Найденное выражение")
        st.code(str(best_ind))
        st.subheader("Древовидная структура")
        try:
            nodes, edges, labels = gp.graph(best_ind)
            g = graphviz.Digraph()
            g.attr('node', shape='circle')
            for i, node in enumerate(nodes): g.node(str(i), labels[i])
            for edge in edges: g.edge(str(edge[0]), str(edge[1]))
            st.graphviz_chart(g)
        except Exception as e:
            st.error(f"Не удалось построить граф: {e}")

    with col2:
        st.subheader("Визуальное сравнение функций")
        x = np.linspace(-1, 1, 100)
        y_true = 2*x**3 - 3*x**2 + 4*x - 1
        y_found = [compiled_func(val) for val in x]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_true, 'b-', label="Исходная функция")
        ax.plot(x, y_found, 'r--', label="Найденная функция")
        ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Сравнение функций")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
        
def render_chapter_4():
    st.title("Часть 4: Создание контроллера интеллектуального робота")
    st.markdown("Этот раздел демонстрирует концепцию, но не запускает полную симуляцию из-за высокой вычислительной нагрузки.")
    st.info("""
    **Концепция:** Генетическое программирование используется для эволюции программы управления роботом.
    **Фитнес-функция:** Количество собранных "яблок" на 2D-карте за ограниченное число шагов.
    **Результат:** Ниже представлен пример оптимальной программы-контроллера, которая могла бы быть получена в результате такого процесса.
    """)
    st.subheader("Пример графа лучшего найденного алгоритма")
    # Статичный пример, так как реальный расчет слишком долгий для веб-приложения
    g = graphviz.Digraph()
    g.attr('node', shape='box')
    g.node("0", "if_food_ahead")
    g.node("1", "move_forward")
    g.node("2", "progn2")
    g.node("3", "turn_left")
    g.node("4", "move_forward")
    g.edge("0", "1"); g.edge("0", "2")
    g.edge("2", "3"); g.edge("2", "4")
    st.graphviz_chart(g)
    st.warning("Полная интерактивная симуляция требует значительных вычислительных ресурсов и в данном демо отключена.")

def render_chapter_5():
    st.title("Часть 5: Рекомендательная система")
    st.markdown("Реализация метода коллаборативной фильтрации 'user-based' с использованием коэффициента корреляции Пирсона.")
    data = load_recommender_data()
    users = list(data.keys())
    
    selected_user = st.selectbox("Выберите пользователя для получения рекомендаций:", users)
    
    if selected_user:
        scores = {other: pearson_score(data, selected_user, other) for other in users if other != selected_user}
        similar_users = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        
        total_scores = {}
        similarity_sums = {}
        for other, similarity in similar_users:
            if similarity <= 0: continue
            for item, rating in data[other].items():
                if item not in data[selected_user]:
                    total_scores.setdefault(item, 0)
                    total_scores[item] += rating * similarity
                    similarity_sums.setdefault(item, 0)
                    similarity_sums[item] += similarity
        
        if not total_scores:
            st.warning("Нет рекомендаций для данного пользователя.")
        else:
            rankings = [(score/similarity_sums[item], item) for item, score in total_scores.items()]
            rankings.sort(reverse=True)
            recommendations = [movie for score, movie in rankings]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Рекомендации для {selected_user}")
                st.dataframe(pd.DataFrame(recommendations, columns=["Фильм"]), use_container_width=True)
            with col2:
                st.subheader("Наиболее похожие пользователи")
                st.dataframe(pd.DataFrame(similar_users, columns=["Пользователь", "Сходство (Пирсон)"]), use_container_width=True)

# --- ГЛАВНЫЙ РОУТЕР ПРИЛОЖЕНИЯ ---
st.sidebar.title("Структура отчета")
chapter = st.sidebar.radio("Выберите раздел для анализа:", [
    "Часть 1: Генерация битовых образов",
    "Часть 2: Визуализация хода эволюции (CMA-ES)",
    "Часть 3: Символическая регрессия",
    "Часть 4: Создание контроллера робота",
    "Часть 5: Рекомендательная система"
])

if chapter == "Часть 1: Генерация битовых образов":
    render_chapter_1()
elif chapter == "Часть 2: Визуализация хода эволюции (CMA-ES)":
    render_chapter_2()
elif chapter == "Часть 3: Символическая регрессия":
    render_chapter_3()
elif chapter == "Часть 4: Создание контроллера робота":
    render_chapter_4()
elif chapter == "Часть 5: Рекомендательная система":
    render_chapter_5()
