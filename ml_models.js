/**
 * Spotify ML Studio - Client-Side Machine Learning Library
 * Implements CSV Parsing, Feature Scaling, Train-Test Splitting,
 * and 4 core Classifiers: Decision Tree, Naive Bayes, KNN, and Logistic Regression.
 */

// --- 1. UTILS & DATA PREPROCESSING ---

/**
 * Robust CSV parser that handles quotes and commas correctly.
 */
function parseCSV(text) {
    const lines = [];
    let row = [""];
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        const next = text[i + 1];

        if (c === '"') {
            if (inQuotes && next === '"') {
                row[row.length - 1] += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c === ',' && !inQuotes) {
            row.push('');
        } else if ((c === '\r' || c === '\n') && !inQuotes) {
            if (c === '\r' && next === '\n') {
                i++;
            }
            lines.push(row);
            row = [''];
        } else {
            row[row.length - 1] += c;
        }
    }
    if (row.length > 1 || row[0] !== '') {
        lines.push(row);
    }
    return lines;
}

/**
 * Min-Max Scaler to normalize features to [0, 1] range.
 * Critical for KNN and Logistic Regression stability and performance.
 */
class MinMaxScaler {
    constructor() {
        self.mins = null;
        self.maxs = null;
    }

    fit(X) {
        const numFeatures = X[0].length;
        this.mins = new Array(numFeatures).fill(Infinity);
        this.maxs = new Array(numFeatures).fill(-Infinity);

        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < numFeatures; j++) {
                if (X[i][j] < this.mins[j]) this.mins[j] = X[i][j];
                if (X[i][j] > this.maxs[j]) this.maxs[j] = X[i][j];
            }
        }
    }

    transform(X) {
        return X.map(row => {
            return row.map((val, colIdx) => {
                const range = this.maxs[colIdx] - this.mins[colIdx];
                return range === 0 ? 0 : (val - this.mins[colIdx]) / range;
            });
        });
    }

    transformRow(row) {
        return row.map((val, colIdx) => {
            const range = this.maxs[colIdx] - this.mins[colIdx];
            return range === 0 ? 0 : (val - this.mins[colIdx]) / range;
        });
    }
}

/**
 * Helper to split dataset into train and test sets.
 */
function trainTestSplit(X, y, testSize = 0.2, randomState = 42) {
    // Simple seeded LCG random generator for reproducibility
    let seed = randomState;
    const random = () => {
        const x = Math.sin(seed++) * 10000;
        return x - Math.floor(x);
    };

    const indices = Array.from({ length: X.length }, (_, i) => i);
    
    // Shuffle indices
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    const splitIdx = Math.floor(X.length * (1 - testSize));
    const trainIndices = indices.slice(0, splitIdx);
    const testIndices = indices.slice(splitIdx);

    const X_train = trainIndices.map(i => X[i]);
    const X_test = testIndices.map(i => X[i]);
    const y_train = trainIndices.map(i => y[i]);
    const y_test = testIndices.map(i => y[i]);

    return { X_train, X_test, y_train, y_test };
}

// --- 2. MACHINE LEARNING CLASSIFIERS ---

/**
 * 2.1 Decision Tree Classifier
 * Built recursively using Gini Impurity. Supports maxDepth hyperparameter.
 */
class DecisionTreeClassifier {
    constructor(maxDepth = 5) {
        this.maxDepth = maxDepth;
        this.root = null;
    }

    fit(X, y) {
        this.root = this._buildTree(X, y, 0);
    }

    _gini(y) {
        const len = y.length;
        if (len === 0) return 0;
        const counts = {};
        for (const val of y) {
            counts[val] = (counts[val] || 0) + 1;
        }
        let sumSq = 0;
        for (const key in counts) {
            const p = counts[key] / len;
            sumSq += p * p;
        }
        return 1 - sumSq;
    }

    _buildTree(X, y, depth) {
        const numSamples = X.length;
        const numFeatures = numSamples > 0 ? X[0].length : 0;
        
        // Count classes
        const classes = {};
        for (const val of y) {
            classes[val] = (classes[val] || 0) + 1;
        }

        const classKeys = Object.keys(classes);
        
        // Base cases: pure node, no features, or max depth reached
        if (classKeys.length === 1 || numSamples <= 2 || depth >= this.maxDepth) {
            let mostCommonClass = 0;
            let maxCount = -1;
            for (const key in classes) {
                if (classes[key] > maxCount) {
                    maxCount = classes[key];
                    mostCommonClass = parseInt(key);
                }
            }
            return { isLeaf: true, class: mostCommonClass, distribution: classes, samples: numSamples };
        }

        // Find best split
        let bestGini = Infinity;
        let bestSplit = null;

        for (let col = 0; col < numFeatures; col++) {
            // Find unique values
            const values = X.map(row => row[col]);
            const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

            // Test thresholds
            for (let i = 0; i < uniqueValues.length - 1; i++) {
                const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

                const leftIdx = [];
                const rightIdx = [];

                for (let r = 0; r < numSamples; r++) {
                    if (X[r][col] <= threshold) {
                        leftIdx.push(r);
                    } else {
                        rightIdx.push(r);
                    }
                }

                if (leftIdx.length === 0 || rightIdx.length === 0) continue;

                const yLeft = leftIdx.map(idx => y[idx]);
                const yRight = rightIdx.map(idx => y[idx]);

                const giniLeft = this._gini(yLeft);
                const giniRight = this._gini(yRight);

                const weightedGini = (leftIdx.length / numSamples) * giniLeft + (rightIdx.length / numSamples) * giniRight;

                if (weightedGini < bestGini) {
                    bestGini = weightedGini;
                    bestSplit = {
                        feature: col,
                        threshold: threshold,
                        leftIdx,
                        rightIdx
                    };
                }
            }
        }

        // If no split was found that improves Gini, make a leaf
        if (!bestSplit) {
            let mostCommonClass = 0;
            let maxCount = -1;
            for (const key in classes) {
                if (classes[key] > maxCount) {
                    maxCount = classes[key];
                    mostCommonClass = parseInt(key);
                }
            }
            return { isLeaf: true, class: mostCommonClass, distribution: classes, samples: numSamples };
        }

        // Recurse left and right
        const XLeft = bestSplit.leftIdx.map(idx => X[idx]);
        const yLeft = bestSplit.leftIdx.map(idx => y[idx]);
        const XRight = bestSplit.rightIdx.map(idx => X[idx]);
        const yRight = bestSplit.rightIdx.map(idx => y[idx]);

        const leftNode = this._buildTree(XLeft, yLeft, depth + 1);
        const rightNode = this._buildTree(XRight, yRight, depth + 1);

        return {
            isLeaf: false,
            feature: bestSplit.feature,
            threshold: bestSplit.threshold,
            left: leftNode,
            right: rightNode,
            samples: numSamples,
            gini: bestGini
        };
    }

    predictRow(row) {
        let node = this.root;
        while (!node.isLeaf) {
            if (row[node.feature] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.class;
    }

    predict(X) {
        return X.map(row => this.predictRow(row));
    }

    predictProbabilityRow(row) {
        let node = this.root;
        while (!node.isLeaf) {
            if (row[node.feature] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        const total = node.samples;
        const positive = node.distribution[1] || 0;
        return total === 0 ? 0 : positive / total;
    }

    predictProbability(X) {
        return X.map(row => this.predictProbabilityRow(row));
    }

    /**
     * Compute Feature Importance by counting the depth and frequency of splits.
     */
    getFeatureImportances(numFeatures) {
        const importances = new Array(numFeatures).fill(0);
        
        function traverse(node) {
            if (node.isLeaf) return;
            // Add feature split importance weighted by sample fraction
            importances[node.feature] += node.samples;
            traverse(node.left);
            traverse(node.right);
        }
        
        if (this.root) {
            traverse(this.root);
        }

        // Normalize
        const sum = importances.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            return importances.map(val => val / sum);
        }
        return importances;
    }
}

/**
 * 2.2 Gaussian Naive Bayes Classifier
 * Calculates conditional probabilities for continuous features assuming normal distribution.
 */
class GaussianNB {
    constructor() {
        this.classes = [];
        this.classPriors = {};
        this.means = {};
        this.variances = {};
    }

    fit(X, y) {
        const numSamples = X.length;
        const numFeatures = X[0].length;
        
        // Identify classes
        this.classes = [...new Set(y)];
        
        // Initialize structures
        for (const c of this.classes) {
            this.classPriors[c] = y.filter(val => val === c).length / numSamples;
            this.means[c] = new Array(numFeatures).fill(0);
            this.variances[c] = new Array(numFeatures).fill(0);

            // Filter rows for this class
            const classRows = X.filter((_, idx) => y[idx] === c);
            const count = classRows.length;

            if (count === 0) continue;

            // Calculate Mean
            for (let f = 0; f < numFeatures; f++) {
                let sum = 0;
                for (let r = 0; r < count; r++) {
                    sum += classRows[r][f];
                }
                this.means[c][f] = sum / count;
            }

            // Calculate Variance
            for (let f = 0; f < numFeatures; f++) {
                let sqSumDiff = 0;
                const m = this.means[c][f];
                for (let r = 0; r < count; r++) {
                    const diff = classRows[r][f] - m;
                    sqSumDiff += diff * diff;
                }
                // Add epsilon (1e-9) to variance to prevent division by zero
                this.variances[c][f] = (sqSumDiff / count) + 1e-9;
            }
        }
    }

    _pdf(x, mean, varValue) {
        const exponent = Math.exp(-((x - mean) ** 2) / (2 * varValue));
        return (1 / Math.sqrt(2 * Math.PI * varValue)) * exponent;
    }

    predictRow(row) {
        let bestProb = -Infinity;
        let bestClass = this.classes[0];

        for (const c of this.classes) {
            let logProb = Math.log(this.classPriors[c]);

            for (let f = 0; f < row.length; f++) {
                const pdfVal = this._pdf(row[f], this.means[c][f], this.variances[c][f]);
                // Log trick to prevent underflow
                logProb += Math.log(pdfVal + 1e-12);
            }

            if (logProb > bestProb) {
                bestProb = logProb;
                bestClass = c;
            }
        }

        return bestClass;
    }

    predict(X) {
        return X.map(row => this.predictRow(row));
    }

    predictProbabilityRow(row) {
        const probs = {};
        let sum = 0;

        for (const c of this.classes) {
            let logProb = Math.log(this.classPriors[c]);
            for (let f = 0; f < row.length; f++) {
                const pdfVal = this._pdf(row[f], this.means[c][f], this.variances[c][f]);
                logProb += Math.log(pdfVal + 1e-12);
            }
            probs[c] = Math.exp(logProb);
            sum += probs[c];
        }

        // Normalize
        if (sum === 0) return this.classPriors[1] || 0.5;
        return (probs[1] || 0) / sum;
    }

    predictProbability(X) {
        return X.map(row => this.predictProbabilityRow(row));
    }

    getFeatureImportances(numFeatures) {
        // For Naive Bayes, relative importance is modeled as the normalized variance of class means.
        // A feature whose class means are very different is highly discriminative.
        const importances = new Array(numFeatures).fill(0);
        if (this.classes.length < 2) return importances;

        const c0 = this.classes[0];
        const c1 = this.classes[1];

        for (let f = 0; f < numFeatures; f++) {
            const diff = Math.abs(this.means[c0][f] - this.means[c1][f]);
            importances[f] = diff;
        }

        const sum = importances.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            return importances.map(val => val / sum);
        }
        return importances;
    }
}

/**
 * 2.3 K-Nearest Neighbors Classifier
 * Calculates Euclidean distance to all training samples. Custom K.
 */
class KNeighborsClassifier {
    constructor(k = 5) {
        this.k = k;
        this.X_train = [];
        this.y_train = [];
    }

    fit(X, y) {
        this.X_train = X;
        this.y_train = y;
    }

    _distance(r1, r2) {
        let sum = 0;
        for (let j = 0; j < r1.length; j++) {
            const diff = r1[j] - r2[j];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    predictRow(row) {
        const distances = [];
        for (let i = 0; i < this.X_train.length; i++) {
            distances.push({
                dist: this._distance(row, this.X_train[i]),
                label: this.y_train[i]
            });
        }
        // Sort ascending
        distances.sort((a, b) => a.dist - b.dist);
        
        // Take top K
        const neighbors = distances.slice(0, this.k);
        const counts = {};
        for (const n of neighbors) {
            counts[n.label] = (counts[n.label] || 0) + 1;
        }

        let bestLabel = 0;
        let maxCount = -1;
        for (const label in counts) {
            if (counts[label] > maxCount) {
                maxCount = counts[label];
                bestLabel = parseInt(label);
            }
        }
        return bestLabel;
    }

    predict(X) {
        return X.map(row => this.predictRow(row));
    }

    predictProbabilityRow(row) {
        const distances = [];
        for (let i = 0; i < this.X_train.length; i++) {
            distances.push({
                dist: this._distance(row, this.X_train[i]),
                label: this.y_train[i]
            });
        }
        distances.sort((a, b) => a.dist - b.dist);
        const neighbors = distances.slice(0, this.k);
        const popularCount = neighbors.filter(n => n.label === 1).length;
        return popularCount / this.k;
    }

    predictProbability(X) {
        return X.map(row => this.predictProbabilityRow(row));
    }

    getFeatureImportances(numFeatures) {
        // KNN doesn't natively have feature weights. We calculate importance
        // as the absolute pearson correlation of each feature with the label.
        const importances = new Array(numFeatures).fill(0);
        if (this.X_train.length === 0) return importances;

        const meanY = this.y_train.reduce((a, b) => a + b, 0) / this.y_train.length;

        for (let f = 0; f < numFeatures; f++) {
            const featureVals = this.X_train.map(row => row[f]);
            const meanF = featureVals.reduce((a, b) => a + b, 0) / featureVals.length;

            let num = 0;
            let denF = 0;
            let denY = 0;

            for (let i = 0; i < this.X_train.length; i++) {
                const diffF = featureVals[i] - meanF;
                const diffY = this.y_train[i] - meanY;
                num += diffF * diffY;
                denF += diffF * diffF;
                denY += diffY * diffY;
            }

            if (denF > 0 && denY > 0) {
                importances[f] = Math.abs(num / Math.sqrt(denF * denY));
            }
        }

        const sum = importances.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            return importances.map(val => val / sum);
        }
        return importances;
    }
}

/**
 * 2.4 Logistic Regression Classifier
 * Standard linear binary classification model trained with gradient descent.
 */
class LogisticRegressionClassifier {
    constructor(learningRate = 0.1, epochs = 200) {
        this.lr = learningRate;
        this.epochs = epochs;
        this.weights = null;
        this.bias = 0;
    }

    _sigmoid(z) {
        return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
    }

    fit(X, y, onEpochProgress) {
        const numSamples = X.length;
        const numFeatures = X[0].length;
        
        // Initialize weights and bias
        this.weights = new Array(numFeatures).fill(0);
        this.bias = 0;

        // Gradient Descent
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let dw = new Array(numFeatures).fill(0);
            let db = 0;
            let cost = 0;

            for (let i = 0; i < numSamples; i++) {
                const row = X[i];
                let z = this.bias;
                for (let j = 0; j < numFeatures; j++) {
                    z += row[j] * this.weights[j];
                }

                const y_pred = this._sigmoid(z);
                const error = y_pred - y[i];

                db += error;
                for (let j = 0; j < numFeatures; j++) {
                    dw[j] += error * row[j];
                }

                // Log cost
                if (y[i] === 1) {
                    cost += -Math.log(y_pred + 1e-15);
                } else {
                    cost += -Math.log(1 - y_pred + 1e-15);
                }
            }

            // Update weights & bias
            db /= numSamples;
            for (let j = 0; j < numFeatures; j++) {
                dw[j] /= numSamples;
                this.weights[j] -= this.lr * dw[j];
            }
            this.bias -= this.lr * db;

            if (onEpochProgress && epoch % 20 === 0) {
                onEpochProgress(epoch, cost / numSamples);
            }
        }
    }

    predictRow(row) {
        let z = this.bias;
        for (let j = 0; j < row.length; j++) {
            z += row[j] * this.weights[j];
        }
        const prob = this._sigmoid(z);
        return prob >= 0.5 ? 1 : 0;
    }

    predict(X) {
        return X.map(row => this.predictRow(row));
    }

    predictProbabilityRow(row) {
        let z = this.bias;
        for (let j = 0; j < row.length; j++) {
            z += row[j] * this.weights[j];
        }
        return this._sigmoid(z);
    }

    predictProbability(X) {
        return X.map(row => this.predictProbabilityRow(row));
    }

    getFeatureImportances(numFeatures) {
        // Feature importance in Logistic Regression is the absolute weight of coefficients.
        if (!this.weights) return new Array(numFeatures).fill(0);
        const importances = this.weights.map(w => Math.abs(w));
        const sum = importances.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            return importances.map(val => val / sum);
        }
        return importances;
    }
}

// --- 3. MODEL PERFORMANCE EVALUATOR ---

class ModelEvaluator {
    static evaluate(y_true, y_pred) {
        const len = y_true.length;
        let tp = 0, tn = 0, fp = 0, fn = 0;

        for (let i = 0; i < len; i++) {
            const yt = y_true[i];
            const yp = y_pred[i];

            if (yt === 1 && yp === 1) tp++;
            else if (yt === 0 && yp === 0) tn++;
            else if (yt === 0 && yp === 1) fp++;
            else if (yt === 1 && yp === 0) fn++;
        }

        const accuracy = (tp + tn) / len;
        const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);
        const recall = (tp + fn) === 0 ? 0 : tp / (tp + fn);
        const f1 = (precision + recall) === 0 ? 0 : 2 * (precision * recall) / (precision + recall);

        return {
            accuracy,
            precision,
            recall,
            f1,
            confusionMatrix: [
                [tn, fp], // Clean Actual -> Predicted Clean, Predicted Hit
                [fn, tp]  // Hit Actual -> Predicted Clean, Predicted Hit
            ]
        };
    }
}
