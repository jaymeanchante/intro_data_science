import json
import statistics
import pandas as pd

configs = json.load(open("./config.json"))
quiz_url = configs["quiz_url"]
df = pd.read_csv(quiz_url)

true_answers = df.iloc[0].copy()
students_answers = df[1:].copy()

def results(first_question, final_question):
    percentage_right_list = []
    for q in range(first_question, final_question):
        right_answers = (students_answers.iloc[:, q] == true_answers.iloc[q]).sum()
        total_answers = len(students_answers)
        percentage_right = (right_answers / total_answers) * 100
        percentage_right_list.append(percentage_right)
    return statistics.mean(percentage_right_list)

avg_python_level = students_answers["Como você classifica seu conhecimento de Python"].mean()
print("Média de classificação de conhecimento:", avg_python_level)

avg_python_years = students_answers["Anos de experiência utilizando Python (profissional, acadêmica, hobby ou outro)"].mean()
print("Média de anos de experiência:", avg_python_years)

avg_python_concepts = results(4, 7)
print("Média acertos de conceitos Python:", avg_python_concepts)

avg_python_basics = results(7, 10)
print("Média acertos de Python básico:", avg_python_basics)

avg_python_functions = results(10, 12)
print("Média acertos de funções:", avg_python_functions)

avg_python_flow = results(12, 16)
print("Média acertos de controle de fluxo:", avg_python_flow)

avg_python_data_structures = results(16, 18)
print("Média acertos de estruturas de dados:", avg_python_data_structures)

avg_python_advanced = results(18, 22)
print("Média acertos de Python avançado:", avg_python_advanced)