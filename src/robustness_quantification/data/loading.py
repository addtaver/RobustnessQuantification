import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

data_sets = ['adult', 'Australian', 'bank', 'BCW', 'german', 'Healthy_aging', 'heart_disease', 'lymphography', 'mushroom', 'nursery', 'primary_tumor', 'solar_small', 'solar_big', 'soybean', 'spect_heart', 'student_math_bin', 'student_math_fourcl', 'student_port_bin', 'student_port_fourcl', 'zoo']


def load_dataset(set_name, randomize=True):
	X_test, y_test = None, None

	if set_name == 'adult':
		data = np.genfromtxt('./data/UCI/adult.data', delimiter=',',dtype=str, autostrip=True)
		data_test = np.genfromtxt('./data/UCI/adult.test', delimiter=',',dtype=str, autostrip=True)
		X_train, X_test = data[:, :-1], data_test[:, :-1]
		y_train, y_test = data[:, -1], data_test[:, -1]
		y_train = np.array([0 if yi.strip() == '<=50K' else 1 for yi in y_train]).astype(int)
		y_test = np.array([0 if yi.strip() == '<=50K.' else 1 for yi in y_test]).astype(int)


		X, y = X_train, y_train

		X = np.delete(X, [0,2,4,10,11,12,13], 1)
		X_test = np.delete(X_test, [0,2,4,10,11,12,13], 1)

		X, y = X[~(X == '?').any(axis=1)], y[~(X == '?').any(axis=1)]
		X_test, y_test = X_test[~(X_test == '?').any(axis=1)], y_test[~(X_test == '?').any(axis=1)]

		if randomize:
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]

			perm = np.random.permutation(len(y_test))
			X_test, y_test = X_test[perm], y_test[perm]


	elif set_name == 'Australian':
		data = np.genfromtxt('./data/UCI//australian.dat',dtype=str, autostrip=True)
		X, y = data[:, :-1], data[:, -1].astype(int)

		X = np.delete(X, [1, 2, 6, 9, 12, 13], 1)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'bank':
		data = np.genfromtxt('./data/UCI/bank.csv',dtype=str, skip_header=1, delimiter=';', autostrip=True)
		# data = np.genfromtxt('data/bank/bank-full.csv',dtype=str, autostrip=True
		X, y = data[:, :-1], data[:, -1]

		X = np.delete(X, [0, 5, 9, 11, 12, 13, 14], 1)
		# X = np.delete(X, [0, 5, 9, 10, 11, 12, 13, 14], 1)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'BCW':
		F = [10]*9

		data = np.genfromtxt('./data/UCI/breast-cancer-wisconsin.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, 1:-1], data[:, -1].astype(int)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]

		X, y = X[~(X == '?').any(axis=1)].astype(int), y[~(X == '?').any(axis=1)].astype(int)

		feat_vals = [np.unique(X[:, i]) for i in range(X.shape[1])]

		X = make_numeric(X, feat_vals=feat_vals)
		y = np.searchsorted(np.unique(y), y)
		return X, y, None, None, 2, F


	elif set_name == 'german':
		data = np.genfromtxt('./data/UCI/german.data',dtype=str, autostrip=True)
		X, y = data[:, :-1], data[:, -1].astype(int)
		X = np.delete(X, [1,4,7,10,12,15,17], 1)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'Healthy_aging':
		data = np.genfromtxt('./data/UCI/NPHA-doctor-visits.csv',delimiter=',', dtype=str, autostrip=True, skip_header=1)
		X, y = data[:, 1:], data[:, 0].astype(int)

		X = np.delete(X, [0], 1)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'heart_disease':
		data = np.genfromtxt('./data/UCI/processed.cleveland.data', delimiter=',', dtype=str, autostrip=True)
		X, y = data[:, :-1], data[:, -1]

		X = np.delete(X, [0, 3, 4, 7, 9], 1)
		X, y = X[~(X == '?').any(axis=1)].astype(float), y[~(X == '?').any(axis=1)].astype(int)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'lymphography':
		data = np.genfromtxt('./data/UCI/lymphography.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, 1:], data[:, 0]

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'mushroom':
		data = np.genfromtxt('./data/UCI/agaricus-lepiota.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, 1:], data[:, 0]
		X, y = X[~(X == '?').any(axis=1)], y[~(X == '?').any(axis=1)]

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'nursery':
		# has a lot of structure
		data = np.genfromtxt('./data/UCI/nursery.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, :-1], data[:, -1]

		# remove instances of class that is underrepresented
		X, y = X[y != 'recommend'], y[y != 'recommend']

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'primary_tumor':
		data = np.genfromtxt('./data/UCI/primary-tumor.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, 1:], data[:, 0]

		np.delete(X, [2, 3], 1)
		X, y = X[~(X == '?').any(axis=1)].astype(int), y[~(X == '?').any(axis=1)].astype(int)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'solar_small' or set_name == 'solar_big':
		if set_name == 'solar_small':
			data = np.genfromtxt('./data/UCI/flare.data1', delimiter=' ',dtype=str, autostrip=True)
		else:
			data = np.genfromtxt('./data/UCI/flare.data2', delimiter=' ',dtype=str, autostrip=True)
		X, y = data[:, 0:-3], data[:, -3:].astype(int)
		y = np.sum(y, axis=1)
		# X, y = data[:, 0:-3], data[:, -3].astype(int)
		y[y.nonzero()] = 1

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]

	elif set_name == 'soybean':
		data_train = np.loadtxt('./data/UCI/backup-large.data', delimiter=',',dtype=str)
		data_test = np.loadtxt('./data/UCI/backup-large.test', delimiter=',',dtype=str)
		X, y = data_train[:, 1:].astype(int), data_train[:, 0]
		X_test, y_test = data_test[:, 1:].astype(int), data_test[:, 0]

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]
			perm = np.random.permutation(len(y_test))
			X_test, y_test = X_test[perm], y_test[perm]


	elif set_name == 'spect_heart':
		data_train = np.genfromtxt('./data/UCI/SPECT.train', delimiter=',',dtype=int, autostrip=True)
		data_test = np.genfromtxt('./data/UCI/SPECT.test', delimiter=',',dtype=int, autostrip=True)
		X_train, y_train = data_train[:, 1:], data_train[:, 0]
		X_test, y_test = data_test[:, 1:], data_test[:, 0]

		# append test data to training data
		X = np.vstack((X_train, X_test))
		y = np.hstack((y_train, y_test))

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'student_math_bin' or set_name == 'student_port_bin' or  set_name == 'student_math_four' or set_name == 'student_port_four':
		if set_name == 'student_math':
			data = np.genfromtxt('./data/UCI/student-mat.csv', delimiter=';',dtype=str, autostrip=True, skip_header=1)
		else:
			data = np.genfromtxt('./data/UCI/student-por.csv', delimiter=';',dtype=str, autostrip=True, skip_header=1)

		X, y = data[:, :-4], data[:, -1].astype(int)

		if 'bin' in set_name:
			# make classes binary: pass of fail
			y = np.array([0 if yi < 10 else 1 for yi in y])
		else:
			# make 4 classes, <5, 5-10, 10-15, >15
			y = np.array([0 if yi < 5 else 1 if yi < 10 else 2 if yi < 15 else 3 for yi in y])

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]


	elif set_name == 'zoo':
		data = np.genfromtxt('./data/UCI/zoo.data', delimiter=',',dtype=str, autostrip=True)
		X, y = data[:, 1:-1].astype(int), data[:, -1].astype(int)

		if randomize:
			# shuffle data
			perm = np.random.permutation(len(y))
			X, y = X[perm], y[perm]

	else:
		raise ValueError('Invalid dataset name')


	feat_vals = [np.unique(X[:, i]) for i in range(X.shape[1])]
	F = [len(feat_val) for feat_val in feat_vals]

	C = len(np.unique(y))

	X = make_numeric(X, feat_vals=feat_vals)
	y = np.searchsorted(np.unique(y), y)

	if X_test is not None:
		feat_vals = [np.unique(X_test[:, i]) for i in range(X_test.shape[1])]
		F_bis = [len(feat_val) for feat_val in feat_vals]
		X_test = make_numeric(X_test, feat_vals=feat_vals)
		y_test = np.searchsorted(np.unique(y_test), y_test)

	# return X_num, y_num, C, F
	return X, y, X_test, y_test, C, F


def make_numeric(X, feat_vals):
	X_num = np.zeros(X.shape)
	for i in range(len(feat_vals)):
		X_num[:, i] = np.searchsorted(feat_vals[i], X[:, i])
	X_num = X_num.astype(int)
	return X_num