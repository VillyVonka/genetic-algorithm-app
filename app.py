import streamlit as st
import random
from deap import base, creator, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Заголовок и описание ---
st.set_page_config(layout="wide")
st.title("🔬 Исследование Генетических Алгоритмов")
st.markdown("Это интерактивное приложение демонстрирует работу генетического алгоритма для поиска битовой строки с заданными свойствами.")

# --- Боковая панель с параметрами ---
st.sidebar.header("Панель управления")
target_sum = st.sidebar.slider("Целевая сумма единиц", 10, 100, 45)
num_attributes = st.sidebar.slider("Длина битовой строки", 20, 150, 75)
num_generations = st.sidebar.slider("Количество поколений", 10, 200, 60)
p_crossover = st.sidebar.slider("Вероятность скрещивания", 0.1, 1.0, 0.8, step=0.05)
p_mutation = st.sidebar.slider("Вероятность мутации", 0.01, 0.5, 0.1, step=0.01)

# --- Основная логика (кэшируется для скорости) ---
@st.cache_data # Эта магия кэширует результат!
def run_ga(target, length, ngen, cxpb, mutpb):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_func(individual):
        return len(individual) - abs(sum(individual) - target),

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=50)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    result_pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
                                               stats=stats, verbose=False)
    return result_pop, logbook

# --- Запуск и отображение результатов ---
result_pop, logbook = run_ga(target_sum, num_attributes, num_generations, p_crossover, p_mutation)

best_individual = tools.selBest(result_pop, k=1)[0]

st.header("Результаты")
col1, col2, col3 = st.columns(3)
col1.metric("Лучший фитнес", f"{best_individual.fitness.values[0]:.2f}")
col2.metric("Сумма единиц", f"{sum(best_individual)}")
col3.metric("Целевая сумма", f"{target_sum}")

st.write("**Лучший найденный индивидуум:**")
# Выводим битовую строку в более читаемом виде
st.code(''.join(map(str, best_individual)), language=None)


st.header("Динамика обучения")
# Визуализация
fig, ax1 = plt.subplots(figsize=(10, 4))
gen = logbook.select("gen")
max_fitness = logbook.select("max")
avg_fitness = logbook.select("avg")

ax1.plot(gen, max_fitness, "b-", label="Максимальный фитнес")
ax1.plot(gen, avg_fitness, "r-", label="Средний фитнес")
ax1.set_xlabel("Поколение")
ax1.set_ylabel("Фитнес")
ax1.legend(loc="lower right")
ax1.grid()
st.pyplot(fig)