import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    layout="wide",
    page_title="Анализ Генетического Алгоритма",
    page_icon="🧬"
)

# --- ОПРЕДЕЛЕНИЕ ТИПОВ DEAP (делается один раз) ---
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass

# --- ФУНКЦИЯ-ВЫЧИСЛИТЕЛЬ (с кэшированием для производительности) ---
@st.cache_data
def run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size):
    """Запускает Генетический Алгоритм и возвращает результаты для визуализации."""
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
    initial_pop = [list(ind) for ind in population] # Сохраняем копию для визуализации

    # Добавляем сбор статистики по минимуму и стандартному отклонению
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    final_pop_obj, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                                  stats=stats, verbose=False)
    
    best_individual = tools.selBest(final_pop_obj, k=1)[0]
    final_pop = [list(ind) for ind in final_pop_obj] # Копия финальной популяции

    return best_individual, logbook, initial_pop, final_pop


# --- ФУНКЦИИ-ОТРИСОВЩИКИ ГРАФИКОВ ---

def create_population_heatmap(population, title):
    """Создает тепловую карту для визуализации популяции."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(population, cmap='gray', interpolation='none', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Гены (биты)")
    ax.set_ylabel("Индивидуумы")
    return fig

def create_convergence_plot(logbook):
    """Создает график сходимости (min, avg, max)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    gen = logbook.select("gen")
    max_vals = logbook.select("max")
    avg_vals = logbook.select("avg")
    min_vals = logbook.select("min")
    
    ax.plot(gen, max_vals, "b-", label="Макс. приспособленность")
    ax.plot(gen, avg_vals, "r-", label="Сред. приспособленность")
    ax.plot(gen, min_vals, "g-", label="Мин. приспособленность")
    
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Значение функции приспособленности")
    ax.set_title("Сходимость популяции по поколениям")
    ax.legend(loc="lower right")
    ax.grid(True)
    return fig

def create_diversity_plot(logbook):
    """Создает график генетического разнообразия (стандартное отклонение)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    gen = logbook.select("gen")
    std_vals = logbook.select("std")
    
    ax.plot(gen, std_vals, "m-", label="Стандартное отклонение")
    
    ax.set_xlabel("Поколение")
    ax.set_ylabel("Стандартное отклонение приспособленности")
    ax.set_title("Динамика генетического разнообразия")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig

# --- ОСНОВНАЯ ЧАСТЬ ПРИЛОЖЕНИЯ ---

st.title("Интерактивный анализ Генетического Алгоритма")
st.markdown("Исследование задачи генерации битовых векторов с помощью эволюционного подхода.")

col1, col2 = st.columns([1, 2]) # Пропорция колонок 1:2

with col1:
    st.subheader("Параметры алгоритма")
    target_sum = st.slider("Целевое значение суммы", 10, 100, 45)
    n_attr = st.slider("Длина индивидуума (бит)", 20, 150, 75)
    ngen = st.slider("Число поколений", 10, 200, 60)
    pop_size = st.slider("Размер популяции", 20, 200, 50)
    cxpb = st.slider("Вероятность скрещивания", 0.0, 1.0, 0.8)
    mutpb = st.slider("Вероятность мутации", 0.0, 1.0, 0.1)
    
    st.info("""
    **Функция приспособленности:** `Длина - abs(Сумма - Цель)`

    **Механизмы:**
    - **Селекция:** Турнирная (`size=3`)
    - **Скрещивание:** Двухточечное
    - **Мутация:** Инвертирование бита
    """)

with col2:
    # Запуск вычислений при любом изменении виджетов
    best_ind, logbook, initial_pop, final_pop = run_genetic_algorithm(target_sum, n_attr, ngen, cxpb, mutpb, pop_size)

    st.header("Результаты")
    
    # --- Блок с итоговыми метриками ---
    sub_col1, sub_col2 = st.columns(2)
    sub_col1.metric("Финальная макс. приспособленность", f"{best_ind.fitness.values[0]:.2f}")
    sub_col2.metric("Количество единиц в лучшем решении", f"{sum(best_ind)}")
    st.write("**Лучший найденный индивидуум:**")
    st.code(''.join(map(str, best_ind)), language=None)

    st.markdown("---")
    
    # --- Блок с визуализацией популяций ---
    st.subheader("Визуализация начальной и конечной популяций")
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.pyplot(create_population_heatmap(initial_pop, "Начальная популяция (Поколение 0)"))
    with vis_col2:
        st.pyplot(create_population_heatmap(final_pop, f"Конечная популяция (Поколение {ngen})"))
    st.markdown("""
    На этих тепловых картах каждая строка представляет одного индивидуума, а каждый столбец — ген (бит). 
    Белый цвет — 1, черный — 0. Видно, как популяция эволюционирует от случайного "шума" к более структурированному и однородному состоянию.
    """)
    
    st.markdown("---")

    # --- Блок с графиками эволюции во вкладках ---
    st.subheader("История эволюции")
    tab1, tab2 = st.tabs(["Сходимость приспособленности", "Динамика генетического разнообразия"])
    
    with tab1:
        st.pyplot(create_convergence_plot(logbook))
        st.info("""
        **Интерпретация графика сходимости:**
        - **Макс. приспособленность** показывает качество лучшего решения в каждом поколении.
        - **Мин. приспособленность** показывает качество худшего решения.
        - По мере работы алгоритма, "плохие" решения отсеиваются, и минимальная приспособленность растет. Сближение всех трех линий указывает на то, что вся популяция сошлась к схожим, высококачественным решениям.
        """)
        
    with tab2:
        st.pyplot(create_diversity_plot(logbook))
        st.info("""
        **Интерпретация графика разнообразия:**
        - **Стандартное отклонение** является мерой генетического разнообразия в популяции.
        - Высокое значение в начале означает большой разброс в качестве решений.
        - Снижение показателя до низких значений говорит о том, что популяция теряет разнообразие и сходится к определенному оптимуму.
        """)
