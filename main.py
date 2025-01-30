import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedentes (entradas)
nivel_sujeira = ctrl.Antecedent(np.arange(0, 101, 1), 'nivel_sujeira')  # Nível de sujeira (0 a 100)
quantidade_louca = ctrl.Antecedent(np.arange(0, 11, 1), 'quantidade_louca')  # Quantidade de louça (0 a 10 kg)
nivel_gordura = ctrl.Antecedent(np.arange(0, 101, 1), 'nivel_gordura')  # Nível de gordura (0 a 100)

# Consequentes (saídas)
tempo_lavagem = ctrl.Consequent(np.arange(0, 121, 1), 'tempo_lavagem')  # Tempo de lavagem (0 a 120 minutos)
temperatura_agua = ctrl.Consequent(np.arange(0, 81, 1), 'temperatura_agua')  # Temperatura da água (0 a 80 °C)

#Funções de pertinência
nivel_sujeira['leve'] = fuzz.trapmf(nivel_sujeira.universe, [0, 0, 30, 50])  # Leve
nivel_sujeira['moderada'] = fuzz.trapmf(nivel_sujeira.universe, [30, 50, 70, 90])  # Moderada
nivel_sujeira['pesada'] = fuzz.trapmf(nivel_sujeira.universe, [70, 90, 100, 100])  # Pesada

quantidade_louca['pouca'] = fuzz.gaussmf(quantidade_louca.universe, 2.5, 1.5)  # Pouca (centro em 2.5)
quantidade_louca['moderada'] = fuzz.gaussmf(quantidade_louca.universe, 5, 1.5)  # Moderada (centro em 5)
quantidade_louca['muita'] = fuzz.gaussmf(quantidade_louca.universe, 7.5, 1.5)  # Muita (centro em 7.5)

nivel_gordura['baixa'] = fuzz.trimf(nivel_gordura.universe, [0, 0, 50])  # Baixa gordura (0 a 50)
nivel_gordura['media'] = fuzz.trimf(nivel_gordura.universe, [30, 50, 70])  # Média gordura (30 a 70)
nivel_gordura['alta'] = fuzz.trimf(nivel_gordura.universe, [50, 100, 100])  # Alta gordura (50 a 100)

tempo_lavagem['curto'] = fuzz.trimf(tempo_lavagem.universe, [0, 0, 60])  # Curto
tempo_lavagem['medio'] = fuzz.trapmf(tempo_lavagem.universe, [30, 60, 90, 120])  # Médio
tempo_lavagem['longo'] = fuzz.trimf(tempo_lavagem.universe, [90, 120, 120])  # Longo

temperatura_agua['fria'] = fuzz.gaussmf(temperatura_agua.universe, 20, 10)  # Fria (centro em 20 °C)
temperatura_agua['morna'] = fuzz.gaussmf(temperatura_agua.universe, 40, 10)  # Morna (centro em 40 °C)
temperatura_agua['quente'] = fuzz.gaussmf(temperatura_agua.universe, 60, 10)  # Quente (centro em 60 °C)

# Visualização das funções de pertinência
nivel_sujeira.view()
plt.savefig("nivel_sujeira.png")
quantidade_louca.view()
plt.savefig("quantidade_louca.png")
nivel_gordura.view()
plt.savefig("nivel_gordura.png")
tempo_lavagem.view()
plt.savefig("tempo_lavagem.png")
temperatura_agua.view()
plt.savefig("temperatura_agua.png")

# Regras fuzzy
regras = []

# Combinações para nível de sujeira 'leve'
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['pouca'] & nivel_gordura['baixa'],
                        (tempo_lavagem['curto'], temperatura_agua['fria'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['pouca'] & nivel_gordura['media'],
                        (tempo_lavagem['curto'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['pouca'] & nivel_gordura['alta'],
                        (tempo_lavagem['curto'], temperatura_agua['morna'])))

regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['moderada'] & nivel_gordura['baixa'],
                        (tempo_lavagem['medio'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['moderada'] & nivel_gordura['media'],
                        (tempo_lavagem['medio'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['moderada'] & nivel_gordura['alta'],
                        (tempo_lavagem['medio'], temperatura_agua['quente'])))

regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['muita'] & nivel_gordura['baixa'],
                        (tempo_lavagem['medio'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['muita'] & nivel_gordura['media'],
                        (tempo_lavagem['medio'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['leve'] & quantidade_louca['muita'] & nivel_gordura['alta'],
                        (tempo_lavagem['medio'], temperatura_agua['quente'])))

# Combinações para nível de sujeira 'moderada'
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['pouca'] & nivel_gordura['baixa'],
                        (tempo_lavagem['medio'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['pouca'] & nivel_gordura['media'],
                        (tempo_lavagem['medio'], temperatura_agua['morna'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['pouca'] & nivel_gordura['alta'],
                        (tempo_lavagem['medio'], temperatura_agua['quente'])))

regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['moderada'] & nivel_gordura['baixa'],
                        (tempo_lavagem['medio'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['moderada'] & nivel_gordura['media'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['moderada'] & nivel_gordura['alta'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))

regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['muita'] & nivel_gordura['baixa'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['muita'] & nivel_gordura['media'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['moderada'] & quantidade_louca['muita'] & nivel_gordura['alta'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))

# Combinações para nível de sujeira 'pesada'
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['pouca'] & nivel_gordura['baixa'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['pouca'] & nivel_gordura['media'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['pouca'] & nivel_gordura['alta'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))

regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['moderada'] & nivel_gordura['baixa'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['moderada'] & nivel_gordura['media'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['moderada'] & nivel_gordura['alta'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))

regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['muita'] & nivel_gordura['baixa'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['muita'] & nivel_gordura['media'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))
regras.append(ctrl.Rule(nivel_sujeira['pesada'] & quantidade_louca['muita'] & nivel_gordura['alta'],
                        (tempo_lavagem['longo'], temperatura_agua['quente'])))

# Sistema de controle
sistema_controle = ctrl.ControlSystem(regras)
sistema = ctrl.ControlSystemSimulation(sistema_controle)

# Entradas
sistema.input['nivel_sujeira'] = 70  # Nível de sujeira (0 a 100)
sistema.input['quantidade_louca'] = 8  # Quantidade de louça (0 a 10 kg)
sistema.input['nivel_gordura'] = 60  # Nível de gordura (0 a 100)

# Computação
sistema.compute()

# Saídas
print(f"Tempo de lavagem: {sistema.output['tempo_lavagem']} minutos")
print(f"Temperatura da água: {sistema.output['temperatura_agua']} °C")

# Visualização das saídas
nivel_gordura.view(sim=sistema)
plt.savefig("nivel_gordura_teste.png")
nivel_sujeira.view(sim=sistema)
plt.savefig("nivel_sujeira_teste.png")
quantidade_louca.view(sim=sistema)
plt.savefig("quantidade_louca_teste.png")
tempo_lavagem.view(sim=sistema)
plt.savefig("tempo_lavagem_saida.png")
temperatura_agua.view(sim=sistema)
plt.savefig("temperatura_agua_saida.png")
