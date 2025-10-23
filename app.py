import streamlit as st
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import operator
import math
import time # Needed for Himmelblau visualization example, though we won't use the callback directly

from deap import base, creator, tools, algorithms, gp, cma, benchmarks
from deap.algorithms import varAnd # Needed for the copied elitism function

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    layout="wide",
    page_title="Анализ эволюционных алгоритмов",
    page_icon="🔬"
)

# --- ОПРЕДЕЛЕНИЕ ТИПОВ DEAP ---
# Existing definitions remain the same...
try: creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception: pass
try: creator.create("FitnessMin", base.Fitness, weights=(-1.0,)); creator.create("IndividualCMA", list, fitness=creator.FitnessMin)
except Exception: pass
try: creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,)); creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMinGP)
except Exception: pass
# New definition for Himmelblau optimization
try: creator.create("FitnessMinCont", base.Fitness, weights=(-1.0,)); creator.create("IndividualCont", list, fitness=creator.FitnessMinCont)
except Exception: pass


# --- КОПИЯ ФУНКЦИИ ИЗ algelitism.py ---
# We copy this function directly into our script for Streamlit Cloud compatibility
def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This function is copied from algelitism.py
    It implements the eaSimple algorithm with an elitism component.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream) # In Streamlit, this won't show in console but is kept for compatibility

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals (excluding elite)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals (clone, crossover, mutation)
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best individuals from the previous generation (elitism)
        if halloffame is not None:
             offspring.extend(halloffame.items) # Add elites to the offspring

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# --- ФУНКЦИИ-ВЫЧИСЛИТЕЛИ ---
# Existing cached functions remain the same...
@st.cache_data
def run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size):
    # ... (code from previous version) ...
    toolbox = base.Toolbox(); toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    def eval_func(individual): return len(individual) - abs(sum(individual) - target_sum),
    toolbox.register("evaluate", eval_func); toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05); toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=pop_size); stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean); stats.register("max", np.max); stats.register("min", np.min); stats.register("std", np.std)
    _, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, logbook

@st.cache_data
def run_cma_es(func_name, ngen_cma, sigma, num_individuals=10):
    # ... (code from previous version) ...
    eval_func_cma, centroid_start = (benchmarks.rastrigin, [5.0]*num_individuals) if func_name == "Растригин" else (benchmarks.rosenbrock, [0.0]*num_individuals)
    strategy = cma.Strategy(centroid=centroid_start, sigma=sigma, lambda_=20); toolbox_cma = base.Toolbox()
    toolbox_cma.register("evaluate", eval_func_cma); hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values); stats.register("min", np.min); logbook_cma = tools.Logbook()
    logbook_cma.header = ['gen', 'evals', 'min'] + list(strategy.values.keys())
    for gen in range(ngen_cma):
        population_cma = strategy.generate(creator.IndividualCMA)
        fitnesses = toolbox_cma.map(toolbox_cma.evaluate, population_cma)
        for ind, fit in zip(population_cma, fitnesses): ind.fitness.values = fit
        strategy.update(population_cma); hall_of_fame.update(population_cma); record = stats.compile(population_cma)
        logbook_cma.record(gen=gen, evals=len(population_cma), **record, **strategy.values)
    return logbook_cma, hall_of_fame

@st.cache_data
def run_symbolic_regression(ngen, pop_size):
    # ... (code from previous version) ...
    pset = gp.PrimitiveSet("MAIN", 1); pset.addPrimitive(operator.add, 2); pset.addPrimitive(operator.sub, 2); pset.addPrimitive(operator.mul, 2)
    toolbox = base.Toolbox(); toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.IndividualGP, toolbox.expr); toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    def evalSymbReg(individual, points): func = toolbox.compile(expr=individual); sqerrors = ((func(x) - (2*x**3 - 3*x**2 + 4*x - 1))**2 for x in points); return math.fsum(sqerrors) / len(points),
    points = [x/10. for x in range(-10, 10)]; toolbox.register("evaluate", evalSymbReg, points=points)
    toolbox.register("select", tools.selTournament, tournsize=3); toolbox.register("mate", gp.cxOnePoint); toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset); pop = toolbox.population(n=pop_size); hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen, halloffame=hof, verbose=False)
    return hof[0], toolbox.compile(expr=hof[0])

@st.cache_data
def load_recommender_data():
    with open('ratings.json', 'r') as f: return json.load(f)

def pearson_score(dataset, user1, user2):
    # ... (code from previous version) ...
    common_movies = {item for item in dataset[user1] if item in dataset[user2]};
    if not common_movies: return 0
    sum1=sum(dataset[user1][item] for item in common_movies); sum2=sum(dataset[user2][item] for item in common_movies)
    p_sum=sum(dataset[user1][item]*dataset[user2][item] for item in common_movies)
    sum1_sq=sum(pow(dataset[user1][item],2) for item in common_movies); sum2_sq=sum(pow(dataset[user2][item],2) for item in common_movies)
    num=p_sum-(sum1*sum2/len(common_movies)); den=np.sqrt((sum1_sq-pow(sum1,2)/len(common_movies))*(sum2_sq-pow(sum2,2)/len(common_movies)))
    return 0 if den == 0 else num / den

# --- НОВАЯ ФУНКЦИЯ-ВЫЧИСЛИТЕЛЬ для функции Химмельблау ---
@st.cache_data
def run_himmelblau_ga(population_size, p_crossover, p_mutation, max_generations, hall_of_fame_size=5):
    """Запускает ГА с элитизмом для функции Химмельблау."""
    LOW, UP = -5, 5
    ETA = 20
    LENGTH_CHROM = 2

    hof = tools.HallOfFame(hall_of_fame_size)
    random.seed(42) # Ensure reproducibility

    toolbox = base.Toolbox()
    def randomPoint(a, b): return [random.uniform(a, b), random.uniform(a, b)]
    toolbox.register("randomPoint", randomPoint, LOW, UP)
    toolbox.register("individualCreator", tools.initIterate, creator.IndividualCont, toolbox.randomPoint)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    def himmelblau(individual):
        x, y = individual
        f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        return f,
    toolbox.register("evaluate", himmelblau)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

    population = toolbox.populationCreator(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Используем скопированную функцию элитизма
    final_pop, logbook = eaSimpleElitism(population, toolbox,
                                         cxpb=p_crossover,
                                         mutpb=p_mutation,
                                         ngen=max_generations,
                                         halloffame=hof,
                                         stats=stats,
                                         verbose=False)
    
    return final_pop, logbook, hof

# --- ФУНКЦИИ-РЕНДЕРЕРЫ ---
# Existing render functions remain the same...
def render_chapter_1():
    # ... (code from previous version) ...
    st.title("Часть 1: Генерация битовых образов"); col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Параметры алгоритма"); SCENARIOS = {...}; selected_scenario_name = st.radio(...)
        params = SCENARIOS[selected_scenario_name]; st.info(params["description"]); is_disabled = params["disabled"]
        cxpb = st.slider("Вероятность скрещивания", 0.0, 1.0, value=params.get("cxpb", 0.8), disabled=is_disabled)
        mutpb = st.slider("Вероятность мутации", 0.0, 1.0, value=params.get("mutpb", 0.1), disabled=is_disabled)
        target_sum = st.slider("Целевое значение суммы", 10, 100, 45); n_attr = st.slider("Длина индивидуума (бит)", 20, 150, 75); ngen = st.slider("Число поколений", 10, 200, 60)
    with col2:
        st.subheader("Результаты моделирования"); best_individual, logbook = run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, 50)
        fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
        ax_conv.plot(logbook.select("gen"), logbook.select("max"), "b-", label="Макс. приспособленность")
        ax_conv.plot(logbook.select("gen"), logbook.select("avg"), "r-", label="Сред. приспособленность")
        ax_conv.plot(logbook.select("gen"), logbook.select("min"), "g-", label="Мин. приспособленность")
        ax_conv.set_xlabel("Поколение"); ax_conv.set_ylabel("Приспособленность"); ax_conv.set_title("Сходимость популяции"); ax_conv.legend(); ax_conv.grid(True)
        st.pyplot(fig_conv)
    st.markdown("---"); st.subheader("Протокол эволюции (Logbook)"); logbook_df = pd.DataFrame(logbook); logbook_df = logbook_df.rename(...)
    st.dataframe(logbook_df.style.background_gradient(...)); st.info("""...""")

def render_chapter_2():
    # ... (code from previous version) ...
    st.title("Часть 2: Оптимизация с помощью CMA-ES"); st.subheader("Параметры CMA-ES"); ngen_cma = st.slider(...) ; sigma = st.slider(...)
    col1, col2 = st.columns(2)
    for col, func_name in zip([col1, col2], ["Растригин", "Розенброк"]):
        with col:
            st.header(func_name); logbook_cma, hall_of_fame = run_cma_es(...) ; st.metric(...)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8)); fig.tight_layout(pad=3.0)
            axes[0, 0].plot(logbook_cma.select('gen'), logbook_cma.select('min')); axes[0, 0].set_title("Лучшее значение функции"); axes[0, 0].grid(True)
            axes[0, 1].plot(logbook_cma.select('gen'), logbook_cma.select('sigma')); axes[0, 1].set_title("Значение Sigma"); axes[0, 1].grid(True)
            axes[1, 0].semilogy(logbook_cma.select('gen'), logbook_cma.select('axis_ratio')); axes[1, 0].set_title("Соотношение осей"); axes[1, 0].grid(True)
            axes[1, 1].semilogy(logbook_cma.select('gen'), [l['diagD'] for l in logbook_cma]); axes[1, 1].set_title("Диагональ D"); axes[1, 1].grid(True)
            st.pyplot(fig)

def render_chapter_3():
    # ... (code from previous version) ...
    st.title("Часть 3: Символическая регрессия"); col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Параметры ГП"); ngen_gp = st.slider(...); pop_size_gp = st.slider(...)
        best_ind, compiled_func = run_symbolic_regression(...) ; st.subheader("Найденное выражение"); st.code(str(best_ind))
        st.subheader("Древовидная структура");
        try: nodes, edges, labels = gp.graph(best_ind); g = graphviz.Digraph(); ... ; st.graphviz_chart(g)
        except: st.error(...)
    with col2:
        st.subheader("Визуальное сравнение функций"); x = np.linspace(-1, 1, 100); ... ; fig, ax = plt.subplots(...); ... ; st.pyplot(fig)

def render_chapter_4():
    # ... (code from previous version) ...
    st.title("Часть 4: Создание контроллера ..."); st.info(...); st.subheader(...)
    g = graphviz.Digraph(); ... ; st.graphviz_chart(g); st.warning(...)

def render_chapter_5():
    # ... (code from previous version) ...
    st.title("Часть 5: Рекомендательная система"); data = load_recommender_data(); users = list(data.keys()); col1, col2 = st.columns(2)
    with col1: st.subheader("Получение рекомендаций"); selected_user = st.selectbox(...);
    if selected_user: scores = {...}; similar_users = sorted(...); total_scores, similarity_sums = {}, {}; ...
    if not total_scores: st.warning(...); else: rankings = sorted(...); st.dataframe(...)
    with col2: st.subheader("Метод вычисления сходства"); st.markdown(...); st.latex(r'''...'''); st.markdown(r"""...""")


# --- НОВЫЙ РЕНДЕРЕР для функции Химмельблау ---
def render_chapter_himmelblau():
    st.title("Часть 2 (доп.): ГА для функции Химмельблау")
    st.markdown("Демонстрация классического Генетического Алгоритма с элитизмом для поиска минимума функции Химмельблау в непрерывном пространстве.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Параметры ГА")
        pop_size = st.slider("Размер популяции", 50, 500, 200, key="him_pop")
        p_cross = st.slider("Вероятность скрещивания", 0.0, 1.0, 0.9, key="him_cx")
        p_mut = st.slider("Вероятность мутации", 0.0, 1.0, 0.2, key="him_mut")
        max_gen = st.slider("Число поколений", 10, 100, 50, key="him_gen")

        final_pop, logbook, hof = run_himmelblau_ga(pop_size, p_cross, p_mut, max_gen)

        st.subheader("Лучшие найденные решения (Hall of Fame):")
        best_solutions = []
        for i, ind in enumerate(hof):
            best_solutions.append({
                "№": i + 1,
                "X": f"{ind[0]:.4f}",
                "Y": f"{ind[1]:.4f}",
                "Значение функции": f"{ind.fitness.values[0]:.4f}"
            })
        st.dataframe(pd.DataFrame(best_solutions), use_container_width=True)


    with col2:
        st.subheader("Визуализация поиска")
        # Готовим данные для контурного графика
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        xgrid, ygrid = np.meshgrid(x, y)
        f_himmelblau = (xgrid**2 + ygrid - 11)**2 + (xgrid + ygrid**2 - 7)**2

        # Известные минимумы функции Химмельблау
        ptMins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        
        # Рисуем контуры функции
        contour = ax.contour(xgrid, ygrid, f_himmelblau, levels=np.logspace(0, 3, 15), cmap='viridis')
        fig.colorbar(contour)

        # Рисуем известные минимумы
        ax.scatter(*zip(*ptMins), marker='X', color='red', s=100, label="Известные минимумы", zorder=2)
        
        # Рисуем финальную популяцию
        pop_x = [ind[0] for ind in final_pop]
        pop_y = [ind[1] for ind in final_pop]
        ax.scatter(pop_x, pop_y, color='blue', s=10, label="Финальная популяция", zorder=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Функция Химмельблау и финальная популяция ГА")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


# --- ГЛАВНЫЙ РОУТЕР ПРИЛОЖЕНИЯ ---
st.sidebar.title("Структура отчета")
chapter = st.sidebar.radio("Выберите раздел для анализа:", [
    "Часть 1: Генерация битовых образов",
    "Часть 2: Визуализация хода эволюции (CMA-ES)",
    "Часть 2 (доп.): ГА для функции Химмельблау", # <-- Новый пункт
    "Часть 3: Символическая регрессия",
    "Часть 4: Создание контроллера робота",
    "Часть 5: Рекомендательная система"
])

if chapter == "Часть 1: Генерация битовых образов": render_chapter_1()
elif chapter == "Часть 2: Визуализация хода эволюции (CMA-ES)": render_chapter_2()
elif chapter == "Часть 2 (доп.): ГА для функции Химмельблау": render_chapter_himmelblau() # <-- Новый роут
elif chapter == "Часть 3: Символическая регрессия": render_chapter_3()
elif chapter == "Часть 4: Создание контроллера робота": render_chapter_4()
elif chapter == "Часть 5: Рекомендательная система": render_chapter_5()
