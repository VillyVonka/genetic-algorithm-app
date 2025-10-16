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

# --- ОПРЕДЕЛЕНИЕ ТИПОВ DEAP (делается один раз) ---
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception: pass
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("IndividualCMA", list, fitness=creator.FitnessMin)
except Exception: pass
try:
    creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
    creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMinGP)
except Exception: pass

# --- ФУНКЦИИ-ВЫЧИСЛИТЕЛИ ---

@st.cache_data
def run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size):
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
    stats.register("min", np.min)
    stats.register("std", np.std)
    _, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, logbook

@st.cache_data
def run_cma_es(func_name, ngen_cma, sigma, num_individuals=10):
    eval_func_cma, centroid_start = (benchmarks.rastrigin, [5.0]*num_individuals) if func_name == "Растригин" else (benchmarks.rosenbrock, [0.0]*num_individuals)
    strategy = cma.Strategy(centroid=centroid_start, sigma=sigma, lambda_=20)
    toolbox_cma = base.Toolbox()
    toolbox_cma.register("evaluate", eval_func_cma)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    logbook_cma = tools.Logbook()
    logbook_cma.header = ['gen', 'evals', 'min'] + list(strategy.values.keys())
    for gen in range(ngen_cma):
        population_cma = strategy.generate(creator.IndividualCMA)
        fitnesses = toolbox_cma.map(toolbox_cma.evaluate, population_cma)
        for ind, fit in zip(population_cma, fitnesses): ind.fitness.values = fit
        strategy.update(population_cma)
        hall_of_fame.update(population_cma)
        record = stats.compile(population_cma)
        logbook_cma.record(gen=gen, evals=len(population_cma), **record, **strategy.values)
    return logbook_cma, hall_of_fame

@st.cache_data
def run_symbolic_regression(ngen, pop_size):
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2); pset.addPrimitive(operator.sub, 2); pset.addPrimitive(operator.mul, 2)
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
    toolbox.register("select", tools.selTournament, tournsize=3); toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen, halloffame=hof, verbose=False)
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
    p_sum = sum(dataset[user1][item] * dataset[user2][item] for item in common_movies)
    sum1_sq = sum(pow(dataset[user1][item], 2) for item in common_movies)
    sum2_sq = sum(pow(dataset[user2][item], 2) for item in common_movies)
    num = p_sum - (sum1 * sum2 / len(common_movies))
    den = np.sqrt((sum1_sq - pow(sum1, 2) / len(common_movies)) * (sum2_sq - pow(sum2, 2) / len(common_movies)))
    return 0 if den == 0 else num / den

# --- ФУНКЦИИ-РЕНДЕРЕРЫ ---

def render_chapter_1():
    st.title("Часть 1: Генерация битовых образов")
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
        best_individual, logbook = run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, 50)
        
        fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
        ax_conv.plot(logbook.select("gen"), logbook.select("max"), "b-", label="Макс. приспособленность")
        ax_conv.plot(logbook.select("gen"), logbook.select("avg"), "r-", label="Сред. приспособленность")
        ax_conv.plot(logbook.select("gen"), logbook.select("min"), "g-", label="Мин. приспособленность")
        ax_conv.set_xlabel("Поколение"); ax_conv.set_ylabel("Приспособленность")
        ax_conv.set_title("Сходимость популяции по поколениям"); ax_conv.legend(); ax_conv.grid(True)
        st.pyplot(fig_conv)

    st.markdown("---")
    st.subheader("Протокол эволюции (Logbook)")
    logbook_df = pd.DataFrame(logbook)
    logbook_df = logbook_df.rename(columns={
        "gen": "Поколение", "evals": "Оценено", "min": "Мин.", 
        "max": "Макс.", "avg": "Сред.", "std": "Ст. откл."
    })
    
    # Визуализация таблицы с цветовой градиентной заливкой
    st.dataframe(logbook_df.style.background_gradient(
        cmap='viridis', subset=['Мин.', 'Макс.', 'Сред.']
    ).background_gradient(
        cmap='plasma_r', subset=['Ст. откл.']
    ))
    st.info("""
    **Интерпретация таблицы:**
    - **Столбцы "Мин.", "Макс.", "Сред.":** Цветовая шкала (от фиолетового к желтому) показывает рост приспособленности. Желтые ячейки в конце говорят о том, что вся популяция достигла высоких показателей.
    - **Столбец "Ст. откл.":** Цветовая шкала (от желтого к фиолетовому) показывает падение генетического разнообразия. Фиолетовые ячейки в конце — признак того, что популяция сошлась к одному решению.
    """)

def render_chapter_2():
    st.title("Часть 2: Оптимизация с помощью CMA-ES")
    st.subheader("Параметры CMA-ES")
    ngen_cma = st.slider("Число поколений", 50, 500, 125, key="ch2_ngen")
    sigma = st.slider("Начальное значение Sigma", 0.1, 10.0, 5.0, key="ch2_sigma")
    col1, col2 = st.columns(2)
    for col, func_name in zip([col1, col2], ["Растригин", "Розенброк"]):
        with col:
            st.header(func_name)
            logbook_cma, hall_of_fame = run_cma_es(func_name, ngen_cma, sigma)
            st.metric("Найденный минимум", f"{hall_of_fame[0].fitness.values[0]:.4f}")
            fig, axes = plt.subplots(2, 2, figsize=(10, 8)); fig.tight_layout(pad=3.0)
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
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Параметры ГП")
        ngen_gp = st.slider("Число поколений", 10, 100, 40, key="ch3_ngen")
        pop_size_gp = st.slider("Размер популяции", 100, 1000, 300, key="ch3_pop")
        best_ind, compiled_func = run_symbolic_regression(ngen_gp, pop_size_gp)
        st.subheader("Найденное выражение"); st.code(str(best_ind))
        st.subheader("Древовидная структура")
        try:
            nodes, edges, labels = gp.graph(best_ind)
            g = graphviz.Digraph(); g.attr('node', shape='circle')
            for i, node in enumerate(nodes): g.node(str(i), labels[i])
            for edge in edges: g.edge(str(edge[0]), str(edge[1]))
            st.graphviz_chart(g)
        except: st.error("Не удалось построить граф.")
    with col2:
        st.subheader("Визуальное сравнение функций")
        x = np.linspace(-1, 1, 100); y_true = 2*x**3 - 3*x**2 + 4*x - 1
        y_found = [compiled_func(val) for val in x]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_true, 'b-', label="Исходная функция")
        ax.plot(x, y_found, 'r--', label="Найденная функция")
        ax.legend(); ax.grid(True); st.pyplot(fig)

def render_chapter_4():
    st.title("Часть 4: Создание контроллера интеллектуального робота")
    st.info("Концепция: Генетическое программирование используется для эволюции программы управления роботом.")
    st.subheader("Пример графа лучшего найденного алгоритма")
    g = graphviz.Digraph(); g.attr('node', shape='box')
    g.node("0", "if_food_ahead"); g.node("1", "move_forward"); g.node("2", "progn2")
    g.node("3", "turn_left"); g.node("4", "move_forward")
    g.edge("0", "1"); g.edge("0", "2"); g.edge("2", "3"); g.edge("2", "4")
    st.graphviz_chart(g)
    st.warning("Полная интерактивная симуляция требует значительных вычислительных ресурсов и в данном демо отключена.")

def render_chapter_5():
    st.title("Часть 5: Рекомендательная система")
    data = load_recommender_data()
    users = list(data.keys())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Получение рекомендаций")
        selected_user = st.selectbox("Выберите пользователя:", users)
        if selected_user:
            scores = {other: pearson_score(data, selected_user, other) for other in users if other != selected_user}
            similar_users = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
            total_scores, similarity_sums = {}, {}
            for other, similarity in similar_users:
                if similarity <= 0: continue
                for item, rating in data[other].items():
                    if item not in data[selected_user]:
                        total_scores.setdefault(item, 0); total_scores[item] += rating * similarity
                        similarity_sums.setdefault(item, 0); similarity_sums[item] += similarity
            if not total_scores:
                st.warning("Нет рекомендаций.")
            else:
                rankings = sorted([(score/similarity_sums[item], item) for item, score in total_scores.items()], reverse=True)
                st.dataframe(pd.DataFrame([m for _, m in rankings], columns=["Рекомендованные фильмы"]), use_container_width=True)
    with col2:
        st.subheader("Метод вычисления сходства")
        st.markdown("Для определения 'похожести' пользователей используется **коэффициент корреляции Пирсона**. Он показывает наличие линейной связи между оценками двух пользователей.")
        st.latex(r'''
        r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
        ''')
        st.markdown(r"""
        - $r$: Коэффициент корреляции (от -1 до 1). Значение `1` означает полную схожесть вкусов.
        - $n$: Количество фильмов, оцененных обоими пользователями.
        - $x_i, y_i$: Оценки $i$-го фильма пользователями X и Y.
        - $\bar{x}, \bar{y}$: Средние оценки пользователей X и Y.
        """)


# --- ГЛАВНЫЙ РОУТЕР ПРИЛОЖЕНИЯ ---
st.sidebar.title("Структура отчета")
chapter = st.sidebar.radio("Выберите раздел для анализа:", [
    "Часть 1: Генерация битовых образов",
    "Часть 2: Визуализация хода эволюции (CMA-ES)",
    "Часть 3: Символическая регрессия",
    "Часть 4: Создание контроллера робота",
    "Часть 5: Рекомендательная система"
])

if chapter == "Часть 1: Генерация битовых образов": render_chapter_1()
elif chapter == "Часть 2: Визуализация хода эволюции (CMA-ES)": render_chapter_2()
elif chapter == "Часть 3: Символическая регрессия": render_chapter_3()
elif chapter == "Часть 4: Создание контроллера робота": render_chapter_4()
elif chapter == "Часть 5: Рекомендательная система": render_chapter_5()
