import csv
import math

# column definitions
NUMERIC_COLS = [
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed',
    'pressure'
]

CATEGORICAL_COLS = ['string_id', 'error_code', 'installation_type']

# mappings for categorical variables
category_maps = {col: {} for col in CATEGORICAL_COLS}


def build_category_maps(rows):
    for row in rows:
        for col in CATEGORICAL_COLS:
            mapping = category_maps[col]
            val = row[col]
            if val not in mapping:
                mapping[val] = len(mapping)


def parse_float(value):
    """Convert numeric strings to float, returning None for blanks."""
    try:
        if value == '' or value is None:
            return None
        return float(value)
    except ValueError:
        return None


def read_dataset(path, is_train=False):
    """Read CSV data returning ids, feature rows and (optionally) targets."""
    ids = []
    rows = []
    targets = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['id']))
            item = {}
            for col in NUMERIC_COLS:
                item[col] = parse_float(row[col])
            for col in CATEGORICAL_COLS:
                # keep empty string so we can treat it as a category
                item[col] = row[col] if row[col] != '' else 'missing'
            rows.append(item)
            if is_train:
                targets.append(float(row['efficiency']))
    if is_train:
        return ids, rows, targets
    return ids, rows

def compute_numeric_means(rows):
    sums = {col: 0.0 for col in NUMERIC_COLS}
    counts = {col: 0 for col in NUMERIC_COLS}
    for row in rows:
        for col in NUMERIC_COLS:
            val = row[col]
            if val is not None:
                sums[col] += val
                counts[col] += 1
    means = {col: (sums[col] / counts[col] if counts[col] > 0 else 0.0)
             for col in NUMERIC_COLS}
    return means


def fill_missing_numeric(rows, means):
    new_rows = []
    for row in rows:
        item = {}
        for col in NUMERIC_COLS:
            val = row[col]
            if val is None:
                val = means[col]
            item[col] = val
        for col in CATEGORICAL_COLS:
            item[col] = row[col]
        new_rows.append(item)
    return new_rows


def encode_rows(rows):
    encoded = []
    for row in rows:
        feat = []
        numeric_vals = [row[col] for col in NUMERIC_COLS]
        feat.extend(numeric_vals)
        for val in numeric_vals:
            feat.append(val * val)
            feat.append(val * val * val)
        for i in range(len(numeric_vals)):
            for j in range(i + 1, len(numeric_vals)):
                feat.append(numeric_vals[i] * numeric_vals[j])
        cat_indicators = []
        for col in CATEGORICAL_COLS:
            mapping = category_maps[col]
            val = row[col]
            index = mapping.get(val)
            indicators = [1.0 if k == index else 0.0 for k in range(len(mapping))]
            feat.extend(indicators)
            cat_indicators.append(indicators)
        encoded.append(feat)
    return encoded


def compute_feature_stats(X):
    n = len(X[0])
    m = len(X)
    means = [0.0] * n
    for row in X:
        for j, val in enumerate(row):
            means[j] += val
    for j in range(n):
        means[j] /= m
    stds = [0.0] * n
    for row in X:
        for j, val in enumerate(row):
            diff = val - means[j]
            stds[j] += diff * diff
    for j in range(n):
        stds[j] = math.sqrt(stds[j] / m)
        if stds[j] == 0:
            stds[j] = 1.0
    return means, stds

def standardize(X, means, stds):
    out = []
    for row in X:
        out_row = [(row[j]-means[j])/stds[j] for j in range(len(row))]
        out.append(out_row)
    return out

def transpose(mat):
    return [list(row) for row in zip(*mat)]


def matmul(a, b):
    res = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            for j in range(len(b[0])):
                res[i][j] += aik * b[k][j]
    return res


def invert_matrix(mat):
    n = len(mat)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(mat)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(pivot):
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
        if abs(pivot) < 1e-12:
            pivot = 1e-12
        scale = 1.0 / pivot
        for j in range(2 * n):
            aug[i][j] *= scale
        for k in range(n):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(2 * n):
                aug[k][j] -= factor * aug[i][j]
    inv = [row[n:] for row in aug]
    return inv


def train_linear_regression_closed_form(X, y, reg=0.1):
    # add bias column
    Xb = [row + [1.0] for row in X]
    Xt = transpose(Xb)
    XtX = matmul(Xt, Xb)
    for i in range(len(XtX)-1):
        XtX[i][i] += reg
    XtX_inv = invert_matrix(XtX)
    XtY = matmul(Xt, [[val] for val in y])
    W = matmul(XtX_inv, XtY)
    w = [row[0] for row in W]
    b = w[-1]
    return w[:-1], b

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
    train_ids, train_rows, train_y = read_dataset('train.csv', True)
    test_ids, test_rows = read_dataset('test.csv', False)

    # impute missing numeric values using training means
    num_means = compute_numeric_means(train_rows)
    train_rows = fill_missing_numeric(train_rows, num_means)
    test_rows = fill_missing_numeric(test_rows, num_means)

    # determine categorical mappings from training data
    build_category_maps(train_rows)

    # encode rows using one-hot categories
    train_X = encode_rows(train_rows)
    test_X = encode_rows(test_rows)

    means, stds = compute_feature_stats(train_X)
    train_X = standardize(train_X, means, stds)
    test_X = standardize(test_X, means, stds)

    w, b = train_linear_regression_closed_form(train_X, train_y, reg=0.1)
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
