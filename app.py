import streamlit as st
from deap import base, algorithms, creator, tools
from deap.algorithms import varAnd  # Импорт, который был в algelitism.py
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Код из файла algelitism.py ---
# Я вставил его прямо сюда

def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, callback=None):
    """Переделанный алгоритм eaSimple с элементом элитизма
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
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

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

        if callback:
            callback[0](*callback[1])

    return population, logbook

# --- Конец кода из algelitism.py ---


# --- Константы и Настройка DEAP ---
LOW, UP = -5, 5
ETA = 20
LENGTH_CHROM = 2
HALL_OF_FAME_SIZE = 5

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Настройка DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]

def himmelblau(individual):
    x, y = individual
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f,

# --- Функции для Streamlit ---

@st.cache_data
def run_ga(pop_size, max_gen, cxpb, mutpb):
    """
    Запускает GA и возвращает историю популяций и логбук.
    """
    toolbox = base.Toolbox()
    toolbox.register("randomPoint", randomPoint, LOW, UP)
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    
    toolbox.register("evaluate", himmelblau)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)
    
    population = toolbox.populationCreator(n=pop_size)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population_history = []
    
    def record_history(population, *args):
        population_history.append(population[:])

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
    # Мы убрали 'algelitism.' и теперь вызываем функцию напрямую
    pop, logbook = eaSimpleElitism(population, toolbox,
                                    cxpb=cxpb,
                                    mutpb=mutpb,
                                    ngen=max_gen,
                                    halloffame=hof,
                                    stats=stats,
                                    callback=(record_history, ()), 
                                    verbose=False)

    return logbook, population_history, hof

def plot_population(population, generation_num, xgrid, ygrid, f_himmelbalu, ax):
    """
    Отрисовывает одно состояние популяции (одно поколение).
    """
    ptMins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    
    ax.clear()
    ax.contour(xgrid, ygrid, f_himmelbalu, levels=20)
    ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=2, s=150, label="Истинные минимумы")
    
    pop_x = [ind[0] for ind in population]
    pop_y = [ind[1] for ind in population]
    ax.scatter(pop_x, pop_y, color='green', s=10, zorder=1, alpha=0.7, label="Популяция")
    
    ax.set_title(f"Поколение: {generation_num}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(LOW-1, UP+1)
    ax.set_ylim(LOW-1, UP+1)
    ax.legend()

def plot_convergence(logbook):
    """
    Отрисовывает график сходимости.
    """
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(minFitnessValues, color='red', label="Мин. (лучшая) приспособленность")
    ax.plot(meanFitnessValues, color='green', label="Средняя приспособленность")
    ax.set_xlabel('Поколение')
    ax.set_ylabel('Приспособленность')
    ax.set_title('График сходимости')
    ax.legend()
    return fig

# --- Интерфейс Streamlit ---

st.set_page_config(layout="wide")
st.title("Интерактивный Генетический Алгоритм 🧬")
st.write("Визуализация оптимизации функции Химмельблау с помощью `DEAP` и `Streamlit`")

# --- 1. Боковая панель с ползунками ---
st.sidebar.header("Параметры Генетического Алгоритма")
P_CROSSOVER = st.sidebar.slider("Вероятность скрещивания (cxpb)", 0.0, 1.0, 0.9, 0.05)
P_MUTATION = st.sidebar.slider("Вероятность мутации (mutpb)", 0.0, 1.0, 0.2, 0.05)
POPULATION_SIZE = st.sidebar.slider("Размер популяции", 50, 500, 200, 10)
MAX_GENERATIONS = st.sidebar.slider("Макс. число поколений (ngen)", 10, 100, 50, 5)

# --- 2. Запуск алгоритма ---
if st.sidebar.button("🚀 Запустить оптимизацию"):
    with st.spinner("Алгоритм выполняется..."):
        logbook, history, hof = run_ga(POPULATION_SIZE, MAX_GENERATIONS, P_CROSSOVER, P_MUTATION)
        
        st.session_state.logbook = logbook
        st.session_state.history = history
        st.session_state.hof = hof
        st.session_state.run_completed = True
        st.success("Оптимизация завершена!")

# --- 3. Отображение результатов ---
if 'run_completed' in st.session_state:
    st.header("Результаты оптимизации")

    if 'xgrid' not in st.session_state:
        x = np.arange(LOW-1, UP+1, 0.1)
        y = np.arange(LOW-1, UP+1, 0.1)
        st.session_state.xgrid, st.session_state.ygrid = np.meshgrid(x, y)
        st.session_state.f_himmelbalu = (st.session_state.xgrid**2 + st.session_state.ygrid - 11)**2 + \
                                       (st.session_state.xgrid + st.session_state.ygrid**2 - 7)**2

    st.subheader("Эволюция популяции")
    history = st.session_state.history
    
    gen_to_show = st.slider("Выберите поколение для отображения:", 0, len(history) - 1, 0)
    
    fig_pop, ax_pop = plt.subplots(figsize=(7, 7))
    plot_population(history[gen_to_show], 
                    gen_to_show, 
                    st.session_state.xgrid, 
                    st.session_state.ygrid, 
                    st.session_state.f_himmelbalu,
                    ax_pop)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.pyplot(fig_pop)
    
    with col2:
        st.subheader("График сходимости")
        fig_conv = plot_convergence(st.session_state.logbook)
        st.pyplot(fig_conv)
        
        st.subheader("Лучшее найденное решение")
        best_ind = st.session_state.hof.items[0]
        best_fitness = himmelblau(best_ind)[0]
        st.metric("Координаты (X, Y)", f"[{best_ind[0]:.4f}, {best_ind[1]:.4f}]")
        st.metric("Значение функции (min)", f"{best_fitness:.6f}")
