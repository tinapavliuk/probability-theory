import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mean_weighted(values, probs):
    # M[X] = Σ xᵢ * pᵢ
    return sum(x * p for x, p in zip(values, probs))

def variance_weighted(values, probs):
    # D[X] = Σ pᵢ * (xᵢ - M[X])²
    mean = mean_weighted(values, probs)
    return sum(p * (x - mean)**2 for x, p in zip(values, probs))

def std_dev_weighted(values, probs):
    return variance_weighted(values, probs)**0.5

def create_empirical_function_text(data):
    """Створює текстове представлення емпіричної функції розподілу"""
    n = len(data)
    sorted_data = sorted(set(data))
    
    function_parts = []
    
    # Додаємо початкову частину (F(x) = 0 для x <= min)
    function_parts.append(f"0, x ≤ {sorted_data[0]:.1f}")
    
    # Обчислюємо кумулятивні частоти
    for i, value in enumerate(sorted_data):
        count = sum(1 for x in data if x <= value)
        freq = count / n
        
        if i < len(sorted_data) - 1:
            next_value = sorted_data[i + 1]
            function_parts.append(f"{freq:.2f}, {value:.1f} < x ≤ {next_value:.1f}")
        else:
            # Останній інтервал
            function_parts.append(f"1, x > {value:.1f}")
    
    return function_parts

# --- Streamlit інтерфейс ---

st.set_page_config(page_title="Статистичний аналіз", layout="wide")
st.title("📊 Статистичний аналіз емпіричних даних")

# Введення даних
st.header("1️⃣ Введення даних")
data_input = st.text_area(
    "Введіть числа через кому або пробіл (якщо бажаєте ввести дробове число, то введіть його через крапку):",
    height=100,
    placeholder="Наприклад: 2, 4, 4, 5, 7, 8, 8, 8, 9"
)

uploaded_file = st.file_uploader(
    "Або завантаж CSV-файл з одним стовпцем чисел",
    type=["csv"]
)

data = []
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    data = df.iloc[:, 0].dropna().astype(float).tolist()
elif data_input:
    try:
        data = list(map(float, data_input.replace(",", " ").split()))
    except:
        st.error("Помилка у введенні. Перевірь, чи всі значення — числа.")

if data:
    data = sorted(data)
    n = len(data)
    st.success(f"Успішно зчитано {n} значень.")
    
    # Таблиці статистичних розподілів
    st.header("2️⃣ Таблиці статистичних розподілів")
    values, counts = np.unique(data, return_counts=True)
    rel_freqs = counts / n
    
    cum_freqs = np.concatenate([[0], np.cumsum(counts)[:-1], [np.sum(counts)]])
    rel_cum_freqs = np.concatenate([[0], np.cumsum(rel_freqs)[:-1], [1.0]])
    
    extended_values = list(values) + [""]
    extended_counts = list(counts) + [""]
    extended_rel_freqs = list(rel_freqs) + [""]
    
    df_stats = pd.DataFrame({
        val: [ni, rfi, cfi, rcfi]
        for val, ni, rfi, cfi, rcfi in zip(extended_values, extended_counts, extended_rel_freqs, cum_freqs, rel_cum_freqs)
    }, index=[
        "Частота (nᵢ)",
        "Відносна частота (nᵢ/n)",
        "Накопичувальна частота",
        "Відносна накопичувальна частота"
    ])
    
    st.dataframe(df_stats, use_container_width=True)
    
    # Полігони частот
    st.header("3️⃣ Графічні характеристики")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values, counts, marker="o", label="Полігон частот")
    ax.plot(values, rel_freqs * n, marker="x", linestyle="--", label="Полігон відносних частот")
    ax.set_title("Полігони частот", fontsize=10)
    ax.set_xlabel("Значення", fontsize=9)
    ax.set_ylabel("Частота", fontsize=9)
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    
    # Емпірична функція розподілу
    st.subheader("Емпірична функція розподілу")
    
    # Текстове представлення функції
    function_parts = create_empirical_function_text(data)
    
    # Форматування функції
    function_text = "F*(x) = {\n"
    for part in function_parts:
        function_text += f"    {part}\n"
    function_text += "}"
    
    st.code(function_text, language="text")
    
    # Графік емпіричної функції
    def empirical_cdf(x, data):
        return sum(1 for val in data if val <= x) / len(data)
    
    # Створюємо точки для стрибкоподібного графіка
    unique_values = sorted(set(data))
    x_points = []
    y_points = []
    
    # Додаємо початкову точку
    x_min = min(data) - 1
    x_points.extend([x_min, unique_values[0]])
    y_points.extend([0, 0])
    
    # Додаємо стрибки для кожного унікального значення
    for i, value in enumerate(unique_values):
        cdf_value = empirical_cdf(value, data)
        
        # Додаємо горизонтальну лінію до стрибка
        if i > 0:
            x_points.append(value)
            y_points.append(y_points[-1])
        
        # Додаємо вертикальний стрибок
        x_points.append(value)
        y_points.append(cdf_value)
        
        # Додаємо горизонтальну лінію після стрибка
        if i < len(unique_values) - 1:
            next_value = unique_values[i + 1]
            x_points.append(next_value)
            y_points.append(cdf_value)
    
    # Додаємо кінцеву точку
    x_max = max(data) + 1
    x_points.append(x_max)
    y_points.append(1.0)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(x_points, y_points, 'b-', linewidth=2)
    
    for value in unique_values:
        cdf_before = empirical_cdf(value - 0.001, data) if value > min(data) else 0
        cdf_at = empirical_cdf(value, data)
        
        if cdf_before < cdf_at:
            ax2.plot(value, cdf_before, 'wo', markersize=4, markeredgecolor='blue')
            ax2.plot(value, cdf_at, 'bo', markersize=4)
        
        ax2.axvline(x=value, color='gray', linestyle=':', alpha=0.5)
        
        if cdf_at < 1.0: 
            arrow_length = (max(data) - min(data)) * 0.02
            ax2.annotate('', xy=(value + arrow_length, cdf_at), 
                        xytext=(value, cdf_at),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    final_value = max(unique_values)
    final_x = final_value + (max(data) - min(data)) * 0.05
    ax2.annotate('', xy=(final_x, 1.0), xytext=(final_value, 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax2.set_title("Емпірична функція розподілу", fontsize=12)
    ax2.set_xlabel("x", fontsize=10)
    ax2.set_ylabel("F*(x)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    st.pyplot(fig2)
    
    # Інтервальний розподіл (гістограма)
    st.header("4️⃣ Інтервальний розподіл")
    
    # Можливість змінювати кількість інтервалів
    st.subheader("Налаштування інтервалів")
    col1, col2 = st.columns(2)
    
    with col1:
        default_m = int(np.floor(np.sqrt(n)))
        m = st.number_input("Кількість інтервалів (m):", min_value=2, max_value=20, value=default_m)
    
    with col2:
        st.write(f"Рекомендована кількість (⌊√n⌋): {default_m}")
    
    # Обчислення параметрів інтервалів
    r = max(data) - min(data)
    h = r / m
    
    st.write(f"**Розмах (R):** {r:.3f}")
    st.write(f"**Довжина інтервалу (h):** {h:.3f}")
    
    # Створення інтервалів
    bin_edges = np.linspace(min(data), max(data), m + 1)
    
    # Таблиця інтервалів
    st.subheader("Таблиця інтервального розподілу")
    
    # Обчислюємо частоти для кожного інтервалу
    hist_counts, _ = np.histogram(data, bins=bin_edges)
    hist_rel_freqs = hist_counts / n
    hist_densities = hist_counts / (n * h)  # Щільність частоти
    
    # Створюємо таблицю
    intervals = []
    midpoints = []
    for i in range(len(bin_edges) - 1):
        interval_str = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        if i == len(bin_edges) - 2:
            interval_str = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]"
        intervals.append(interval_str)
        midpoints.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    # Накопичувальні частоти для інтервалів
    cum_interval_freqs = np.cumsum(hist_counts)
    rel_cum_interval_freqs = cum_interval_freqs / n
    
    df_intervals = pd.DataFrame({
        "Інтервал": intervals,
        "Середина інтервалу": [f"{mp:.2f}" for mp in midpoints],
        "Частота (nᵢ)": hist_counts,
        "Відносна частота": [f"{rf:.3f}" for rf in hist_rel_freqs],
        "Щільність частоти": [f"{d:.3f}" for d in hist_densities],
        "Накопичувальна частота": cum_interval_freqs,
        "Відн. накопич. частота": [f"{rcf:.3f}" for rcf in rel_cum_interval_freqs]
    })
    
    st.dataframe(df_intervals, use_container_width=True)
    
    # Гістограма
    st.subheader("Гістограма")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7, color='pink')
    ax3.set_xticks(bin_edges.round(2))
    ax3.set_xticklabels([f"{edge:.2f}" for edge in bin_edges], rotation=45)
    ax3.set_title("Гістограма частот", fontsize=12)
    ax3.set_xlabel("Значення", fontsize=10)
    ax3.set_ylabel("Частота", fontsize=10)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    
    st.header("5️⃣ Числові характеристики")
    mean = mean_weighted(values, rel_freqs)
    variance = variance_weighted(values, rel_freqs)
    std_dev = std_dev_weighted(values, rel_freqs)
    
    median = np.median(data)
    mode = max(set(data), key=data.count)
    
    st.markdown(f"""
    - 📌 **Мода:** {mode}
    - 📌 **Медіана:** {median:.3f}
    - 📌 **Математичне сподівання (очікуване значення):** {mean:.3f}
    - 📌 **Дисперсія:** {variance:.3f}
    - 📌 **Середньоквадратичне відхилення:** {std_dev:.3f}
    """)
else:
    st.info("Введіть або завантаж дані, щоб побачити результати.")