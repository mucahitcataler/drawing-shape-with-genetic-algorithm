import random
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from alive_progress import alive_bar

# Read and transform the image
img = cv.imread('IMAGE PATH', 2)
imgX = 20
imgY = 20
img = cv.resize(img, (imgX, imgY), interpolation=cv.INTER_CUBIC)
# 0 --> empty cells, 1--> full cells
img = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
# Calculate number of full cells (paintedArea) in image.
paintedArea = 0
for i in range(imgX):
    for j in range(imgY):
        if img[i][j] == 1:
            paintedArea += 1
# pLength --> population length
pLength = 5000
# iLength --> individual length
iLength = int(0.3*imgX*imgY)
# Calculate the maximum angle that an individual ever has.
maxAngle = (iLength-1)*4
# mutateProbability --> probability of mutation
mutateProbability = 0.3
# maxGeneration --> maximum number of generations
maxGeneration = 200
# crossoverPoint --> number of point that do crossover
crossoverPoint = 2
# numberOfNewIndividual --> number of new individuals added to the new generation
numberOfNewIndividual = int(pLength/2)
# fitnessScoreThresholds[0] --> minimum fitness score, fitnessScoreThresholds[1] --> maximum fitness score
fitnessScoreThresholds = [0.5, 10]

# visulize() --> function that returns the given individual as an image.
def visualize(individual):
    individualImg = np.zeros((imgX, imgY), dtype=np.uint8)
    currentX = imgX - 1
    currentY = 0
    individualImg[currentX][currentY] = 1
    for i in range(iLength):
        if individual[i] == 1:
            currentX -= 1
        elif individual[i] == 2:
            currentX -= 1
            currentY -= 1
        elif individual[i] == 3:
            currentY -= 1
        elif individual[i] == 4:
            currentX += 1
            currentY -= 1
        elif individual[i] == 5:
            currentX += 1
        elif individual[i] == 6:
            currentX += 1
            currentY += 1
        elif individual[i] == 7:
            currentY += 1
        elif individual[i] == 8:  
            currentX -= 1
            currentY += 1
        if currentX < 0 or currentX > imgX-1 or currentY < 0 or currentY > imgY-1:
            break
        individualImg[currentX][currentY] = 1
    return individualImg

# fitness() --> function that returns fittness score of a given individual.
def fitness(individual):
    areaScore = 0
    angleScore = 0
    individualImg = np.zeros((imgX, imgY), dtype=int)
    currentX = imgX - 1
    currentY = 0
    individualImg[currentX][currentY] = 1
    if img[currentX][currentY] == 1:
        areaScore += 1
    for i in range(iLength):
        if individual[i] == 1:
            currentX -= 1
        elif individual[i] == 2:
            currentX -= 1
            currentY -= 1
        elif individual[i] == 3:
            currentY -= 1
        elif individual[i] == 4:
            currentX += 1
            currentY -= 1
        elif individual[i] == 5:
            currentX += 1
        elif individual[i] == 6:
            currentX += 1
            currentY += 1
        elif individual[i] == 7:
            currentY += 1
        elif individual[i] == 8:  
            currentX -= 1
            currentY += 1
        if currentX < 0 or currentX > imgX-1 or currentY < 0 or currentY > imgY-1:
            break
        if i > 0:
            angle = np.abs(individual[i] - individual[i-1])
            if angle > 4:
                angle = 8 - angle
            angleScore += angle
        if individualImg[currentX][currentY] != 1 and img[currentX][currentY] == 1:
            areaScore += 1
        individualImg[currentX][currentY] = 1
    for i in range(imgX):
        for j in range(imgY):
            if img[i][j] != individualImg[i][j]:
                areaScore -= 0.2
    areaScore = float(areaScore)/paintedArea
    angleScore = float(angleScore)/maxAngle
    return (10 * areaScore) - (0.75 * angleScore)

# select() --> function that selects an individual from population using fitness scores.
def select(population, fitnessScores):
    sumOfScores = np.sum(fitnessScores)
    sortedIndices = np.argsort(fitnessScores)
    fitnessScores = np.sort(fitnessScores)
    selectedScore = random.random()*sumOfScores
    currentScore = 0.0
    for i in range(pLength):
        currentScore += fitnessScores[i]
        if currentScore >= selectedScore:
            return population[sortedIndices[i]]
    return

# reproduce() --> function that returns a child from given two parents.
def reproduce(parentA, parentB):
    crossoverPoints = []
    while crossoverPoint > len(crossoverPoints):
        point = random.randint(1, iLength-1)
        if point not in crossoverPoints:
            crossoverPoints.append(point)
    childA = parentA
    childB = parentB
    change = True
    for i in range(iLength):
        if i in crossoverPoints:
            change = not change
        if change == True:
            childA[i], childB[i] = parentB[i], parentA[i]
    if fitness(childA) > fitness(childB):
        return childA
    return childB

# mutate() --> function that mutate a given individual with a probability.
def mutate(individual):
    if not random.random() > mutateProbability:
        individual[random.randint(0,iLength-1)] = random.randint(1,8)
    return individual

# Initialize the first population randomly.
firstSelection = [1,7,8]
population = np.array([[random.randint(1,8) for i in range(iLength-1)] for j in range(pLength)])
population = np.array([np.insert(population[i], 0, firstSelection[random.randint(0,2)]) for i in range(pLength)])
fitnessScores = np.array([fitness(population[i]) for i in range(pLength)])
fitnessScores[fitnessScores < 0] = 0
fitnessEvolution = []
maxFitnessScore = np.max(fitnessScores)
fitnessEvolution.append(maxFitnessScore)
start = time.time()
generationTime = []
generationTime.append(start - start)
generation = 0

# Generate new generations until reach maximum number of generations or maximum fitness score.
with alive_bar(maxGeneration, manual=True) as bar:
    while generation < maxGeneration and maxFitnessScore > fitnessScoreThresholds[0] and maxFitnessScore < fitnessScoreThresholds[1]:
        bar(generation/maxGeneration)
        newGeneration = np.array([np.zeros(iLength) for i in range(numberOfNewIndividual)])
        for i in range(numberOfNewIndividual):
            parentA = select(population, fitnessScores)
            parentB = select(population, fitnessScores)
            child = reproduce(parentA, parentB)
            child = mutate(child)
            newGeneration[i] = child
        sortedIndices = np.argsort(fitnessScores)
        oldGeneration = np.delete(population, sortedIndices[:numberOfNewIndividual], axis=0)
        population = np.concatenate((oldGeneration, newGeneration), axis=0)
        fitnessScores = np.array([fitness(population[i]) for i in range(pLength)])
        fitnessScores[fitnessScores < 0] = 0
        maxFitnessScore = np.max(fitnessScores)
        fitnessEvolution.append(maxFitnessScore)
        generationTime.append(time.time() - start)
        generation += 1
    bar(1.)

# Write why exiting the while loop.
if generation >= maxGeneration:
    print("Reached the maximum number of generations. Number of Generation = " + str(generation))
else:
    if maxFitnessScore >= fitnessScoreThresholds[1]:
        print("Reached the maximum fitness score threshold. Number of Generation = " + str(generation))
    else:
        print("Maximum fitness score is below the threshold.")

# Do calculation for plotting.
sortedIndices = np.argsort(fitnessScores)
bestIndividual = visualize(population[sortedIndices[-1]])
bestIndividual[bestIndividual == 0] = 255
bestIndividual[bestIndividual != 255] = 0
bestIndividual = cv.cvtColor(bestIndividual, cv.COLOR_GRAY2RGB)
img[img == 0] = 255
img[img != 255] = 0
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

# Plot results.
plt.subplot(2, 1, 1)
plt.plot(np.array(generationTime), np.array(fitnessEvolution))
plt.title("Evolution of Maximum Fitness Score")
plt.xlabel("Time (Seconds)")
plt.ylabel("Fitness Score")
plt.subplot(2, 2, 3)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(2, 2, 4)
plt.imshow(bestIndividual)
plt.title("Best Individual")
plt.tight_layout()
plt.show()