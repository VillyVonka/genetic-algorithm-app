import streamlit as st
from deap import base, algorithms
from deap import creator
from deap import tools
import algelitism  # Наш кастомный алгоритм
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Константы и Настройка DEAP ---
# (Почти без изменений из вашего ga_9.py)

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

# @st.cache_data кеширует результат.
# Это значит, что GA не будет перезапускаться каждый раз,
# когда вы двигаете ползунок "Поколение".
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

    # --- Ключевое изменение ---
    # Мы будем сохранять историю популяций здесь
    population_history = []
    
    # Кастомная callback-функция для сохранения истории
    def record_history(population, *args):
        population_history.append(population[:]) # Важно: сохранить копию

    # Запускаем GA с нашим колбэком
    pop, logbook = algelitism.eaSimpleElitism(population, toolbox,
                                            cxpb=cxpb,
                                            mutpb=mutpb,
                                            ngen=max_gen,
                                            halloffame=hof,
                                            stats=stats,
                                            # Передаем нашу функцию
                                            callback=(record_history, ()), 
                                            verbose=False) # Отключаем print в консоль

    return logbook, population_history, hof

def plot_population(population, generation_num, xgrid, ygrid, f_himmelbalu, ax):
    """
    Отрисовывает одно состояние популяции (одно поколение).
    """
    ptMins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    
    ax.clear()
    ax.contour(xgrid, ygrid, f_himmelbalu, levels=20)
    ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=2, s=150, label="Истинные минимумы")
    
    # Разделяем популяцию на X и Y
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
        
        # Сохраняем результаты в 'session_state'
        # чтобы они не пропали при движении другого ползунка
        st.session_state.logbook = logbook
        st.session_state.history = history
        st.session_state.hof = hof
        st.session_state.run_completed = True
        st.success("Оптимизация завершена!")

# --- 3. Отображение результатов ---
if 'run_completed' in st.session_state:
    st.header("Результаты оптимизации")

    # Подготовка данных для графика (делаем 1 раз)
    if 'xgrid' not in st.session_state:
        x = np.arange(LOW-1, UP+1, 0.1)
        y = np.arange(LOW-1, UP+1, 0.1)
        st.session_state.xgrid, st.session_state.ygrid = np.meshgrid(x, y)
        st.session_state.f_himmelbalu = (st.session_state.xgrid**2 + st.session_state.ygrid - 11)**2 + \
                                       (st.session_state.xgrid + st.session_state.ygrid**2 - 7)**2

    # --- 3.1. Интерактивный ползунок поколений ---
    st.subheader("Эволюция популяции")
    
    history = st.session_state.history
    
    # Ползунок для выбора поколения
    gen_to_show = st.slider("Выберите поколение для отображения:", 0, len(history) - 1, 0)
    
    # Создаем холст для графика
    fig_pop, ax_pop = plt.subplots(figsize=(7, 7))
    plot_population(history[gen_to_show], 
                    gen_to_show, 
                    st.session_state.xgrid, 
                    st.session_state.ygrid, 
                    st.session_state.f_himmelbalu,
                    ax_pop)
    
    # --- 3.2. Вывод графиков и результатов ---
    
    col1, col2 = st.columns([1.5, 1]) # Делим экран на 2 колонки
    
    with col1:
        st.pyplot(fig_pop) # График с популяцией
    
    with col2:
        st.subheader("График сходимости")
        fig_conv = plot_convergence(st.session_state.logbook)
        st.pyplot(fig_conv) # График сходимости
        
        st.subheader("Лучшее найденное решение")
        best_ind = st.session_state.hof.items[0]
        best_fitness = himmelblau(best_ind)[0]
        st.metric("Координаты (X, Y)", f"[{best_ind[0]:.4f}, {best_ind[1]:.4f}]")
        st.metric("Значение функции (min)", f"{best_fitness:.6f}")
