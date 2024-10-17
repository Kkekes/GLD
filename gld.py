# Game of Life Devolution

# to devolve a field A n steps back: to find such a field B, that matches A after applying the GoL rules n times

import numpy as np
import scipy.signal as sps
import random

# Genetic algorithm settings
SIZE = 11
BATCH_SIZE = 50
BASE_DENSITY = 0.5 # the density of the initial guess field
N_REPETITION_LIMIT = 5

# for each cells counts the amount of alive neighbours
def mat_nNeighboursMap(mat_field, includeMiddle=False):
	mat_kernel = np.array([[1, 1, 1], 
					   [1, includeMiddle, 1],
					   [1, 1, 1]])
	return sps.convolve2d(mat_field, mat_kernel, mode="same")

# evolves the field n times according to the GoL rules
def evolve(mat_field, n=1):
	if n == 1:
		mat_neighbours = mat_nNeighboursMap(mat_field)
		mat_nextField = np.where((mat_field == 1) & ((mat_neighbours == 2) | (mat_neighbours == 3)) |
	                             (mat_field == 0) & (mat_neighbours == 3), 1, 0)
		return mat_nextField
	return evolve(evolve(mat_field), n - 1)

def fieldSimilarityScore(mat_field1, mat_field2):
	# similarity = 1 - (#differing cells / #cells)
	return float(1 - sum(sum(mat_field1 ^ mat_field2)) / (SIZE ** 2))

def generateRandomField(mat_probMap):
	mat_field = np.zeros((SIZE, SIZE), dtype=bool)
	for i, j in np.ndindex((SIZE, SIZE)):
		mat_field[i, j] = random.random() < mat_probMap[i, j]
	return mat_field

def printBatch(batch):
	for mat_field in batch:
		printField(mat_field.astype(int))
		print()

def printField(field):
	for i in field:
		for j in i:
			print("■ " if j else "□ ", end="")
		print()

def _directDevolve(mat_tef, n):
	# TEF - Target Evolved Field
	if n <= 0: return mat_tef

	def getEvolvedScore(mat_field): return fieldSimilarityScore(evolve(mat_field, n), mat_tef)
	def nextGen(mat_field, n_max_cpg): 
		# n_max_cpg - max possible changes per generation
		batch = []
		for _ in range(BATCH_SIZE):
			mat_newField = mat_field.copy()
			for _ in range(random.randint(1, n_max_cpg)):
				changedPoint = random.choice(changablePoints)
				mat_newField[*changedPoint] ^= True
			batch.append(mat_newField)
		batch.insert(0, mat_field)

		getEvolvedScores = [getEvolvedScore(mat_field) for mat_field in batch]
		mat_bestField = max(zip(getEvolvedScores, batch), key=lambda pair: pair[0])[1]
		return mat_bestField


	mat_zeroZone = mat_nNeighboursMap(mat_tef, True) == 0
	mat_nonZeroZone = ~mat_zeroZone
	mat_probMap = mat_nonZeroZone * BASE_DENSITY
	changablePoints = np.argwhere(mat_nonZeroZone)
	mat_field = generateRandomField(mat_probMap)

	n_scoreRepeated = 0
	evolvedScore = 0
	prevEvolvedScore = 0
	n_iterations = 0

	while True:
		n_iterations += 1

		n_max_cpg = 1 + (n_scoreRepeated // N_REPETITION_LIMIT)
		if n_max_cpg > len(changablePoints) / 2:
			n_max_cpg = 1
			mat_field = generateRandomField(mat_probMap)

		mat_field = nextGen(mat_field, n_max_cpg)

		prevEvolvedScore = evolvedScore
		evolvedScore = getEvolvedScore(mat_field)

		if evolvedScore == prevEvolvedScore: n_scoreRepeated += 1
		else: n_scoreRepeated = 0

		print(f"Iteration {n_iterations}")
		print(f"Score: {evolvedScore}")
		print(f"Score repeated {n_scoreRepeated} times")
		print(f"Improvement: {evolvedScore - prevEvolvedScore}")
		print(f"n_max_cpg: {n_max_cpg}")
		print()

		if evolvedScore == 1: break

	return mat_field.astype(int)


def devolve(mat_tef, n=1, mode="direct"): 
	# TEF - Target Evolved Field
	assert mode	in ("direct", "consecutive", "consecutive_double")

	# MODES:

	# direct - chooses the best candidate by evolving it n times and comparing to the TEF
	# works bad for n > 2

	# consecutive - recursively devolves TEF one step at a time
	# works best for n >= 2
	# for n == 1 (or n == 2 in some cases) the direct mode is faster

	# consecutive double - recursively devolves TEF two steps at a time
	# less stable version of the consecutive mode
	# may be better for simple shapes

	# stopping the recursion in 'consecutive' or 'consecutive_double' mode
	if n == 0: return mat_tef
	if n == 1: return _directDevolve(mat_tef, 1)

	match mode:
		case "direct":
			return _directDevolve(mat_tef, 1)
		case "consecutive":
			return devolve(_directDevolve(mat_tef, 1), n - 1, "consecutive")
		case "consecutive_double":
			return devolve(_directDevolve(mat_tef, 2), n - 2, "consecutive_double")

field = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
				  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(bool)


printField(devolve(field, 3, "consecutive"))