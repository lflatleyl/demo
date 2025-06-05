import csv
import math
import sys

category_maps = {
    'string_id': {},
    'error_code': {},
    'installation_type': {}
}

def to_float(value):
    try:
        if value == '' or value is None:
            return 0.0
        return float(value)
    except ValueError:
        return 0.0

def encode_category(name, value):
    if value == '' or value is None:
        return 0.0
    mapping = category_maps[name]
    if value not in mapping:
        mapping[value] = len(mapping) + 1
    return float(mapping[value])

def read_dataset(path, is_train=False):
    features = []
    targets = []
    ids = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['id']))
            feat = [
                to_float(row['temperature']),
                to_float(row['irradiance']),
                to_float(row['humidity']),
                to_float(row['panel_age']),
                to_float(row['maintenance_count']),
                to_float(row['soiling_ratio']),
                to_float(row['voltage']),
                to_float(row['current']),
                to_float(row['module_temperature']),
                to_float(row['cloud_coverage']),
                to_float(row['wind_speed']),
                to_float(row['pressure']),
                encode_category('string_id', row['string_id']),
                encode_category('error_code', row['error_code']),
                encode_category('installation_type', row['installation_type'])
            ]
            features.append(feat)
            if is_train:
                targets.append(float(row['efficiency']))
    if is_train:
        return ids, features, targets
    return ids, features

def compute_feature_stats(X):
    n = len(X[0])
    m = len(X)
    means = [0.0]*n
    for row in X:
        for j, val in enumerate(row):
            means[j] += val
    for j in range(n):
        means[j] /= m
    stds = [0.0]*n
    for row in X:
        for j, val in enumerate(row):
            diff = val - means[j]
            stds[j] += diff*diff
    for j in range(n):
        stds[j] = math.sqrt(stds[j]/m)
        if stds[j] == 0:
            stds[j] = 1.0
    return means, stds

def standardize(X, means, stds):
    out = []
    for row in X:
        out_row = [(row[j]-means[j])/stds[j] for j in range(len(row))]
        out.append(out_row)
    return out

def train_linear_regression(X, y, lr=0.01, epochs=100):
    m = len(X)
    n = len(X[0])
    w = [0.0]*n
    b = 0.0
    for _ in range(epochs):
        grad_w = [0.0]*n
        grad_b = 0.0
        for i in range(m):
            pred = b
            for j in range(n):
                pred += w[j]*X[i][j]
            err = pred - y[i]
            grad_b += err
            for j in range(n):
                grad_w[j] += err * X[i][j]
        for j in range(n):
            w[j] -= lr * grad_w[j]/m
        b -= lr * grad_b/m
    return w, b

def predict(X, w, b):
    preds = []
    for row in X:
        val = b
        for j in range(len(row)):
            val += w[j]*row[j]
        preds.append(val)
    return preds

def rmse(y_true, y_pred):
    m = len(y_true)
    s = 0.0
    for i in range(m):
        diff = y_pred[i]-y_true[i]
        s += diff*diff
    return math.sqrt(s/m)

def main():
    train_ids, train_X, train_y = read_dataset('train.csv', True)
    test_ids, test_X = read_dataset('test.csv', False)

    means, stds = compute_feature_stats(train_X)
    train_X = standardize(train_X, means, stds)
    test_X = standardize(test_X, means, stds)

    w, b = train_linear_regression(train_X, train_y, lr=0.01, epochs=100)
    train_pred = predict(train_X, w, b)
    error = rmse(train_y, train_pred)
    print('Training RMSE:', error)

    test_pred = predict(test_X, w, b)
    with open('predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'efficiency'])
        for idx, val in zip(test_ids, test_pred):
            writer.writerow([idx, val])

if __name__ == '__main__':
    main()
