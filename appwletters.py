import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#streamlit run appwletters.py

st.set_page_config(page_title="Категоріальний аналіз", layout="wide")
st.title("📊 Статистичний аналіз категоріальних (текстових) даних")

st.header("1️⃣ Введення даних")
data_input = st.text_area(
    "Введи категорії через кому або пробіл:",
    height=100,
    placeholder="Наприклад: а, b, c, a, b, a, d"
)

uploaded_file = st.file_uploader(
    "Або завантаж CSV-файл з одним стовпцем категорій",
    type=["csv"]
)

data = []
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    data = df.iloc[:, 0].dropna().astype(str).tolist()
elif data_input:
    data = data_input.replace(",", " ").split()
    data = [val.strip() for val in data if val.strip() != ""]

if data:
    n = len(data)
    st.success(f"Успішно зчитано {n} значень.")
    
    # --- Таблиця частот ---
    st.header("2️⃣ Таблиці частот категорій")
    values, counts = np.unique(data, return_counts=True)
    rel_freqs = counts / n
    cum_freqs = np.cumsum(counts)
    rel_cum_freqs = np.cumsum(rel_freqs)
    
    df_stats = pd.DataFrame({
        "Частота (nᵢ)": counts,
        "Відносна частота (nᵢ/n)": rel_freqs,
        "Накопичувальна частота": cum_freqs,
        "Відносна накопичувальна частота": rel_cum_freqs
    }, index=values)
    
    df_stats_transposed = df_stats.T
    st.dataframe(df_stats_transposed, use_container_width=True)
    
    # --- Графік частот ---
    st.header("3️⃣ Графік частот категорій")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(values, counts, color='skyblue')
    ax.set_title("Частоти категорій")
    ax.set_xlabel("Категорії")
    ax.set_ylabel("Частота")
    ax.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    #Емпірична функція
    st.header("4️⃣ Емпірична функція розподілу")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.step(np.arange(len(values)), rel_cum_freqs, where='post', label='Емпірична функція')
    ax2.set_xticks(np.arange(len(values)))
    ax2.set_xticklabels(values, rotation=45, ha='right')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Категорії")
    ax2.set_ylabel("F(x)")
    ax2.set_title("Емпірична функція розподілу (накопичувальна відносна частота)")
    ax2.grid(True)
    st.pyplot(fig2)
    
    # --- Числові характеристики для категорій ---
    st.header("5️⃣ Основні характеристики категоріальних даних")
    mode = values[np.argmax(counts)]
    unique_count = len(values)
    
    st.markdown(f"""
    - 📌 **Мода (найпоширеніша категорія):** {mode}
    - 📌 **Кількість унікальних категорій:** {unique_count}
    - 📌 **Загальна кількість значень:** {n}
    """)
else:
    st.info("Введи або завантаж категоріальні дані, щоб побачити результати.")
    #hi this is ivan
