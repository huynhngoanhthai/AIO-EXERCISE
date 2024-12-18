# -*- coding: utf-8 -*-
"""GA_Advertising_Problem.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XaKiXIAhRjKKBC7HRaZtlAEHYv_sZ64G
"""

# Commented out IPython magic to ensure Python compatibility.
# aivietnam.ai - advertising
import numpy as np
# from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
random.seed(0) # please do not remove this line
# %matplotlib inline

def load_data_from_file(fileName = "advertising.csv"):
  data = np.genfromtxt(fileName, dtype=None, delimiter=',', skip_header=1)
  features_X = data[:, :3]
  sales_Y = data[:, 3]

  # **************** your code here ****************
  features_X = features_X.astype(float)
  sales_Y = sales_Y.astype(float)

  return features_X, sales_Y

#Question 2
features_X, _ = load_data_from_file()
print(features_X[:5,:])

#Question 3
_, sales_Y = load_data_from_file()
print(sales_Y.shape)

def generate_random_value(bound = 10):
    return (random.random() - 0.5)*bound

def create_individual(n=4, bound=10):

  # **************** your code here ****************
  individual = [generate_random_value(bound) for _ in range(n)]

  return individual

individual = create_individual()
print(individual)

def compute_loss(individual):
    theta = np.array(individual)
    y_hat = features_X.dot(theta)
    loss  = np.multiply((y_hat-sales_Y), (y_hat-sales_Y)).mean()
    return loss

def compute_fitness(individual):

    # **************** your code here ****************
    fitness = 1 / (compute_loss(individual) + 1)

    return fitness

#question 4
features_X, sales_Y = load_data_from_file()
individual = [4.09, 4.82, 3.10, 4.02]
fitness_score = compute_fitness(individual)
print(fitness_score)

def crossover(individual1, individual2, crossover_rate = 0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()

   # **************** your code here ****************
    if random.random() < crossover_rate:
        point = random.randint(1, len(individual1) - 1)
        individual1_new[point:], individual2_new[point:] = individual2_new[point:], individual1_new[point:]

    return individual1_new, individual2_new

#question 5
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]

individual1, individual2 = crossover(individual1, individual2, 2.0)
print("individual1: ", individual1)
print("individual2: ", individual2)

def mutate(individual, mutation_rate = 0.05):
    individual_m = individual.copy()

    # **************** your code here ****************
    for i in range(len(individual_m)):
        if random.random() < mutation_rate:
            individual_m[i] = generate_random_value()

    return individual_m

#Question 6
before_individual = [4.09, 4.82, 3.10, 4.02]
after_individual = mutate(before_individual, mutation_rate = 2.0)
print(before_individual == after_individual)

def initializePopulation(m):
  population = [create_individual() for _ in range(m)]
  return population

population = initializePopulation(100)
print(len(population))

def selection(sorted_old_population, m):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if (index2 != index1):
            break

    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s

population = initializePopulation(m=100)
individual_s = selection(population, m = 100)
print(individual_s)

def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)

    if gen%1 == 0:
        print("Best loss:", compute_loss(sorted_population[m-1]), "with chromsome: ", sorted_population[m-1])

    new_population = []
    while len(new_population) < m-elitism:
        # selection
        parent1 = selection(sorted_population, m)
        parent2 = selection(sorted_population, m)

        # crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # mutation
        offspring1 = mutate(offspring1)
        offspring2 = mutate(offspring2)

        new_population.extend([offspring1, offspring2])

    # copy elitism chromosomes that have best fitness score to the next generation
    for ind in sorted_population[m-elitism:]:
        new_population.append(ind)

    return new_population, compute_loss(sorted_population[m-1])

#Question 7
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]
old_population = [individual1, individual2]
new_population, _ = create_new_population(old_population, elitism=2, gen=1)

def run_GA():
  n_generations = 100
  m = 600
  features_X, sales_Y = load_data_from_file()
  population = initializePopulation(m)
  losses_list = []
  for i in range(n_generations):

    # *********** your code here *************
    population, best_loss = create_new_population(population, elitism=2, gen=i)
    losses_list.append(best_loss)

  return losses_list

losses_list = run_GA()

import matplotlib.pyplot as plt

def visualize_loss(losses_list):

      # *********** your code here *************
      plt.plot(losses_list)
      plt.xlabel('Generation')
      plt.ylabel('Loss')
      plt.title('Loss over Generations')
      plt.show()

losses_list = run_GA()
visualize_loss(losses_list)

def visualize_predict_gt():
  # visualization of ground truth and predict value
  sorted_population = sorted(population, key=compute_fitness)
  print(sorted_population[-1])
  theta = np.array(sorted_population[-1])

  estimated_prices = []
  for feature in features_X:
     # ************* your code here *************
     estimated_price = sum(c*x for x, c in zip(feature, theta))
     estimated_prices.append(estimated_price)

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.xlabel('Samples')
  plt.ylabel('Price')
  plt.plot(sales_Y, c='green', label='Real Prices')
  plt.plot(estimated_prices, c='blue', label='Estimated Prices')
  plt.legend()
  plt.show()

visualize_predict_gt()

# visualization of ground truth and predict value
sorted_population = sorted(population, key=compute_fitness)
print(sorted_population[-1])
theta = np.array(sorted_population[-1])

estimated_prices = []
samples = [i for i in range(len(features_X))]
for feature in features_X:
    estimated_price = sum(c*x for x, c in zip(feature, theta))
    estimated_prices.append(estimated_price)
fig, ax = plt.subplots(figsize=(10, 6))
# plt.plot(prices, c='green')
# plt.plot(estimated_prices, c='red')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.scatter(samples, sales_Y, c='green', label='Real Prices')
plt.scatter(samples, estimated_prices, c='blue', label='Estimated Prices')
plt.legend()
plt.show()