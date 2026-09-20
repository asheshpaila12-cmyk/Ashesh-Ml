
# ============================================================
# 23CSE301 - Lab Session 07
# IRCTC Stock Price Dataset
# VS Code / Python .py version
#
# A1 - Entropy
# A2 - Gini Index
# A3 - Root node using Information Gain
# A4 - Equal-width / Equal-frequency binning
# A5 - Custom Decision Tree
# A6 - Decision Tree visualization
# A7 - Two-feature decision boundary
# A8 - GridSearchCV hyperparameter tuning
# ============================================================

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report
)


# ============================================================
# A4 - BINNING FUNCTIONS
# ============================================================

def equal_width_binning(series, n_bins=4):
    """Convert a numeric series into equal-width categorical bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.isna().any():
        raise ValueError("The series contains non-numeric or missing values.")

    return pd.cut(
        numeric_series,
        bins=n_bins,
        labels=False,
        include_lowest=True,
        duplicates="drop"
    ).astype(str)


def equal_frequency_binning(series, n_bins=4):
    """Convert a numeric series into approximately equal-frequency bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.isna().any():
        raise ValueError("The series contains non-numeric or missing values.")

    # Rank first so qcut also works when repeated values are present.
    ranked_series = numeric_series.rank(method="first")

    return pd.qcut(
        ranked_series,
        q=n_bins,
        labels=False,
        duplicates="drop"
    ).astype(str)


def bin_feature(series, bin_type="equal_width", n_bins=4):
    """
    General binning function.

    Default parameters:
        bin_feature(series)
        -> equal-width binning with 4 bins.

    Custom parameters:
        bin_feature(series, "equal_frequency", 5)

    Python does not support traditional function overloading by
    parameter signature, so default parameters provide the requested
    default-parameter behavior.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return series.astype(str)

    if bin_type == "equal_width":
        return equal_width_binning(series, n_bins)

    if bin_type == "equal_frequency":
        return equal_frequency_binning(series, n_bins)

    raise ValueError(
        "bin_type must be 'equal_width' or 'equal_frequency'."
    )


# ============================================================
# A1 - ENTROPY
# ============================================================

def calculate_entropy(target):
    """Calculate Shannon entropy of a categorical target."""
    probabilities = target.value_counts(normalize=True)

    entropy = -(
        probabilities * np.log2(probabilities)
    ).sum()

    return float(entropy)


# ============================================================
# A2 - GINI INDEX
# ============================================================

def calculate_gini_index(target):
    """Calculate Gini impurity of a categorical target."""
    probabilities = target.value_counts(normalize=True)

    gini = 1 - (probabilities ** 2).sum()

    return float(gini)


# ============================================================
# A3 - INFORMATION GAIN
# ============================================================

def calculate_information_gain(feature, target):
    """Calculate information gain for a categorical feature."""
    parent_entropy = calculate_entropy(target)

    weighted_child_entropy = 0.0

    for _, child_target in target.groupby(feature, observed=True):
        weight = len(child_target) / len(target)

        weighted_child_entropy += (
            weight * calculate_entropy(child_target)
        )

    return float(parent_entropy - weighted_child_entropy)


def find_root_node(features, target):
    """
    Find the root feature using information gain.

    Returns:
        root_feature
        information_gains
    """
    information_gains = {}

    for feature in features.columns:
        information_gains[feature] = calculate_information_gain(
            features[feature],
            target
        )

    root_feature = max(
        information_gains,
        key=information_gains.get
    )

    return root_feature, information_gains


# ============================================================
# A5 - CUSTOM DECISION TREE
# ============================================================

def build_decision_tree(
    features,
    target,
    max_depth=None,
    min_samples=2
):
    """
    Build an ID3-style decision tree using information gain.

    The tree is represented as nested dictionaries.
    """

    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)

    def majority_class(values):
        return values.mode().iloc[0]

    def grow(current_features, current_target, depth):

        majority = majority_class(current_target)

        # All records belong to one class.
        if current_target.nunique() == 1:
            return {
                "type": "leaf",
                "class": current_target.iloc[0],
                "samples": len(current_target)
            }

        # No features remain.
        if current_features.empty:
            return {
                "type": "leaf",
                "class": majority,
                "samples": len(current_target)
            }

        # Maximum depth reached.
        if max_depth is not None and depth >= max_depth:
            return {
                "type": "leaf",
                "class": majority,
                "samples": len(current_target)
            }

        # Too few observations.
        if len(current_target) < min_samples:
            return {
                "type": "leaf",
                "class": majority,
                "samples": len(current_target)
            }

        root_feature, gains = find_root_node(
            current_features,
            current_target
        )

        # No useful information gain.
        if gains[root_feature] <= 0:
            return {
                "type": "leaf",
                "class": majority,
                "samples": len(current_target)
            }

        node = {
            "type": "node",
            "feature": root_feature,
            "gain": gains[root_feature],
            "samples": len(current_target),
            "majority_class": majority,
            "children": {}
        }

        remaining_features = current_features.drop(
            columns=[root_feature]
        )

        categories = sorted(
            current_features[root_feature].unique(),
            key=str
        )

        for category in categories:

            mask = (
                current_features[root_feature].astype(str)
                == str(category)
            )

            child_features = (
                remaining_features.loc[mask]
                .reset_index(drop=True)
            )

            child_target = (
                current_target.loc[mask]
                .reset_index(drop=True)
            )

            if len(child_target) == 0:

                node["children"][str(category)] = {
                    "type": "leaf",
                    "class": majority,
                    "samples": 0
                }

            else:

                node["children"][str(category)] = grow(
                    child_features,
                    child_target,
                    depth + 1
                )

        return node

    return grow(features, target, 0)


def predict_custom_tree(tree, row):
    """Predict one observation using the custom tree."""
    if tree["type"] == "leaf":
        return tree["class"]

    feature = tree["feature"]
    feature_value = str(row[feature])

    if feature_value not in tree["children"]:
        return tree["majority_class"]

    return predict_custom_tree(
        tree["children"][feature_value],
        row
    )


def predict_custom_tree_dataset(tree, features):
    """Predict all observations using the custom tree."""
    predictions = []

    for _, row in features.iterrows():
        predictions.append(
            predict_custom_tree(tree, row)
        )

    return pd.Series(
        predictions,
        index=features.index
    )


# ============================================================
# A6 - CUSTOM TREE VISUALIZATION
# ============================================================

def plot_custom_tree(tree):
    """
    Draw the custom decision tree using matplotlib.

    Returns the matplotlib Axes object.
    """

    figure, axis = plt.subplots(figsize=(18, 10))

    positions = {}
    labels = {}
    leaf_counter = [0]

    def assign_positions(node, depth=0):

        if node["type"] == "leaf":

            x_position = leaf_counter[0]
            leaf_counter[0] += 1

            positions[id(node)] = (
                x_position,
                -depth
            )

            labels[id(node)] = (
                f"Class = {node['class']}\n"
                f"Samples = {node['samples']}"
            )

            return x_position

        child_positions = []

        for child in node["children"].values():
            child_positions.append(
                assign_positions(
                    child,
                    depth + 1
                )
            )

        x_position = sum(child_positions) / len(
            child_positions
        )

        positions[id(node)] = (
            x_position,
            -depth
        )

        labels[id(node)] = (
            f"Feature = {node['feature']}\n"
            f"Gain = {node['gain']:.4f}\n"
            f"Samples = {node['samples']}"
        )

        return x_position

    def draw_edges(node):

        if node["type"] == "leaf":
            return

        parent_x, parent_y = positions[id(node)]

        for category, child in node["children"].items():

            child_x, child_y = positions[id(child)]

            axis.annotate(
                "",
                xy=(child_x, child_y + 0.08),
                xytext=(parent_x, parent_y - 0.08),
                arrowprops={
                    "arrowstyle": "->"
                }
            )

            midpoint_x = (
                parent_x + child_x
            ) / 2

            midpoint_y = (
                parent_y + child_y
            ) / 2

            axis.text(
                midpoint_x,
                midpoint_y,
                str(category),
                ha="center",
                va="center",
                fontsize=9
            )

            draw_edges(child)

    assign_positions(tree)
    draw_edges(tree)

    for node_id, position in positions.items():

        axis.text(
            position[0],
            position[1],
            labels[node_id],
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.5",
                "alpha": 0.85
            }
        )

    axis.set_axis_off()
    axis.set_title(
        "Custom ID3 Decision Tree",
        fontsize=16
    )

    figure.tight_layout()

    return figure


# ============================================================
# A7 - DECISION BOUNDARY
# ============================================================

def plot_decision_boundary(
    classifier,
    X,
    y,
    feature_names
):
    """Plot a two-dimensional decision boundary."""

    x_values = X.iloc[:, 0]
    y_values = X.iloc[:, 1]

    x_range = x_values.max() - x_values.min()
    y_range = y_values.max() - y_values.min()

    x_margin = (
        x_range * 0.05
        if x_range != 0
        else 1
    )

    y_margin = (
        y_range * 0.05
        if y_range != 0
        else 1
    )

    xx, yy = np.meshgrid(
        np.linspace(
            x_values.min() - x_margin,
            x_values.max() + x_margin,
            400
        ),
        np.linspace(
            y_values.min() - y_margin,
            y_values.max() + y_margin,
            400
        )
    )

    grid = pd.DataFrame(
        np.c_[
            xx.ravel(),
            yy.ravel()
        ],
        columns=feature_names
    )

    predictions = classifier.predict(
        grid
    ).reshape(xx.shape)

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    class_codes = (
        pd.Series(y)
        .astype("category")
        .cat.codes
    )

    axis.contourf(
        xx,
        yy,
        pd.Series(
            predictions.ravel()
        ).astype("category").cat.codes.values.reshape(
            predictions.shape
        ),
        alpha=0.25
    )

    scatter = axis.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=class_codes,
        edgecolor="black",
        alpha=0.75
    )

    axis.set_xlabel(feature_names[0])
    axis.set_ylabel(feature_names[1])
    axis.set_title(
        "Decision Tree Decision Boundary"
    )

    figure.tight_layout()

    return figure


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Locate the Excel file relative to this Python script.
    # This makes the program work reliably in VS Code.
    # --------------------------------------------------------

    script_directory = Path(__file__).resolve().parent

    data_file = (
        script_directory /
        "Lab Session Data.xlsx"
    )

    if not data_file.exists():

        raise FileNotFoundError(
            f"\nExcel file not found:\n{data_file}\n\n"
            "Place 'Lab Session Data.xlsx' in the same "
            "folder as this Python file."
        )

    # --------------------------------------------------------
    # Load IRCTC worksheet
    # --------------------------------------------------------

    data = pd.read_excel(
        data_file,
        sheet_name="IRCTC Stock Price"
    )

    print("=" * 65)
    print("23CSE301 - LAB SESSION 07")
    print("IRCTC Stock Price Dataset")
    print("=" * 65)

    print("\nDataset shape:")
    print(data.shape)

    print("\nDataset columns:")
    print(list(data.columns))

    print("\nFirst five records:")
    print(data.head())

    # --------------------------------------------------------
    # Convert Volume into numeric form.
    #
    # Examples:
    # 1.67M -> 1670000
    # 707.73K -> 707730
    # --------------------------------------------------------

    data["Volume_num"] = (
        data["Volume"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace("M", "e6", regex=False)
        .str.replace("K", "e3", regex=False)
        .astype(float)
    )

    # --------------------------------------------------------
    # A1
    #
    # Chg% is continuous, so create 4 equal-width bins.
    # --------------------------------------------------------

    data["Chg_Bin"] = pd.cut(
        data["Chg%"],
        bins=4,
        labels=[
            "Very Low",
            "Low",
            "High",
            "Very High"
        ],
        include_lowest=True
    )

    target = data["Chg_Bin"]

    print("\n" + "=" * 65)
    print("A1 - ENTROPY")
    print("=" * 65)

    entropy = calculate_entropy(target)

    print(
        f"Entropy of Chg% target = {entropy:.6f}"
    )

    print("\nTarget class distribution:")
    print(target.value_counts())

    # --------------------------------------------------------
    # A2 - GINI INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A2 - GINI INDEX")
    print("=" * 65)

    gini = calculate_gini_index(target)

    print(
        f"Gini Index = {gini:.6f}"
    )

    # --------------------------------------------------------
    # Prepare categorical features.
    #
    # Equal-frequency binning is used for numeric attributes
    # because it gives reasonably populated categories.
    # --------------------------------------------------------

    custom_features = pd.DataFrame({

        "Price_bin": bin_feature(
            data["Price"],
            "equal_frequency",
            4
        ),

        "Open_bin": bin_feature(
            data["Open"],
            "equal_frequency",
            4
        ),

        "High_bin": bin_feature(
            data["High"],
            "equal_frequency",
            4
        ),

        "Low_bin": bin_feature(
            data["Low"],
            "equal_frequency",
            4
        ),

        "Volume_bin": bin_feature(
            data["Volume_num"],
            "equal_frequency",
            4
        ),

        "Month": data["Month"].astype(str),

        "Day": data["Day"].astype(str)
    })

    # --------------------------------------------------------
    # A3 - INFORMATION GAIN / ROOT NODE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A3 - INFORMATION GAIN AND ROOT NODE")
    print("=" * 65)

    root_feature, information_gains = find_root_node(
        custom_features,
        target
    )

    sorted_gains = sorted(
        information_gains.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nInformation Gain values:")

    for feature, gain in sorted_gains:
        print(
            f"{feature:15s} : {gain:.6f}"
        )

    print(
        f"\nRoot node selected using Information Gain: "
        f"{root_feature}"
    )

    # --------------------------------------------------------
    # A4 - BINNING DEMONSTRATION
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A4 - BINNING")
    print("=" * 65)

    default_bins = bin_feature(
        data["Price"]
    )

    custom_bins = bin_feature(
        data["Price"],
        "equal_frequency",
        5
    )

    print(
        "Default binning: equal-width, 4 bins"
    )

    print(
        "Custom binning: equal-frequency, 5 bins"
    )

    print("\nDefault binning sample:")
    print(default_bins.head())

    print("\nCustom binning sample:")
    print(custom_bins.head())

    # --------------------------------------------------------
    # A5 - CUSTOM DECISION TREE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A5 - CUSTOM DECISION TREE")
    print("=" * 65)

    custom_tree = build_decision_tree(
        custom_features,
        target,
        max_depth=4,
        min_samples=2
    )

    custom_predictions = (
        predict_custom_tree_dataset(
            custom_tree,
            custom_features
        )
    )

    custom_accuracy = accuracy_score(
        target,
        custom_predictions
    )

    print(
        f"Custom Decision Tree training accuracy: "
        f"{custom_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # A6 - CUSTOM TREE VISUALIZATION
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A6 - CUSTOM DECISION TREE VISUALIZATION")
    print("=" * 65)

    custom_tree_figure = plot_custom_tree(
        custom_tree
    )

    custom_tree_figure.savefig(
        script_directory /
        "A6_custom_decision_tree.png",
        dpi=200,
        bbox_inches="tight"
    )

    print(
        "Tree saved as: A6_custom_decision_tree.png"
    )

    # --------------------------------------------------------
    # A7 - TWO-FEATURE CLASSIFICATION
    #
    # Price and Volume are used as the two features.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A7 - TWO-FEATURE DECISION BOUNDARY")
    print("=" * 65)

    two_features = [
        "Price",
        "Volume_num"
    ]

    X_two = data[two_features]
    y_two = target

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_two,
            y_two,
            test_size=0.25,
            random_state=42,
            stratify=y_two
        )
    )

    boundary_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    )

    boundary_tree.fit(
        X_train,
        y_train
    )

    boundary_predictions = (
        boundary_tree.predict(X_test)
    )

    test_accuracy = accuracy_score(
        y_test,
        boundary_predictions
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            boundary_predictions
        )
    )

    print(
        f"Two-feature test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Two-feature balanced accuracy: "
        f"{test_balanced_accuracy:.4f}"
    )

    # Decision boundary plot.
    boundary_figure = plot_decision_boundary(
        boundary_tree,
        X_two,
        y_two,
        two_features
    )

    boundary_figure.savefig(
        script_directory /
        "A7_decision_boundary.png",
        dpi=200,
        bbox_inches="tight"
    )

    print(
        "Decision boundary saved as: "
        "A7_decision_boundary.png"
    )

    # --------------------------------------------------------
    # A6 - SKLEARN TREE VISUALIZATION
    # --------------------------------------------------------

    print("\nCreating sklearn Decision Tree visualization...")

    tree_figure, tree_axis = plt.subplots(
        figsize=(18, 10)
    )

    plot_tree(
        boundary_tree,
        feature_names=two_features,
        class_names=[
            str(value)
            for value in sorted(
                y_two.unique(),
                key=str
            )
        ],
        filled=True,
        rounded=True,
        ax=tree_axis
    )

    tree_axis.set_title(
        "Decision Tree - IRCTC Price and Volume"
    )

    tree_figure.tight_layout()

    tree_figure.savefig(
        script_directory /
        "A6_sklearn_decision_tree.png",
        dpi=200,
        bbox_inches="tight"
    )

    print(
        "Sklearn tree saved as: "
        "A6_sklearn_decision_tree.png"
    )

    # --------------------------------------------------------
    # A8 - GRIDSEARCHCV
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("A8 - HYPERPARAMETER TUNING")
    print("=" * 65)

    all_features = [
        "Price",
        "Open",
        "High",
        "Low",
        "Volume_num"
    ]

    X_all = data[all_features]
    y_all = target

    (
        X_train_all,
        X_test_all,
        y_train_all,
        y_test_all
    ) = train_test_split(
        X_all,
        y_all,
        test_size=0.25,
        random_state=42,
        stratify=y_all
    )

    parameter_grid = {

        "criterion": [
            "gini",
            "entropy"
        ],

        "max_depth": [
            2,
            3,
            4,
            5,
            None
        ],

        "min_samples_split": [
            2,
            5,
            10
        ],

        "min_samples_leaf": [
            1,
            2,
            4
        ],

        "class_weight": [
            None,
            "balanced"
        ]
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(
            random_state=42
        ),
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=cross_validation,
        n_jobs=-1
    )

    print(
        "\nRunning GridSearchCV..."
    )

    grid_search.fit(
        X_train_all,
        y_train_all
    )

    tuned_tree = grid_search.best_estimator_

    tuned_predictions = (
        tuned_tree.predict(
            X_test_all
        )
    )

    tuned_accuracy = accuracy_score(
        y_test_all,
        tuned_predictions
    )

    tuned_balanced_accuracy = (
        balanced_accuracy_score(
            y_test_all,
            tuned_predictions
        )
    )

    print("\nBest hyperparameters:")

    for parameter, value in (
        grid_search.best_params_.items()
    ):
        print(
            f"{parameter}: {value}"
        )

    print(
        f"\nBest cross-validation balanced accuracy: "
        f"{grid_search.best_score_:.4f}"
    )

    print(
        f"Test accuracy after tuning: "
        f"{tuned_accuracy:.4f}"
    )

    print(
        f"Test balanced accuracy after tuning: "
        f"{tuned_balanced_accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test_all,
            tuned_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save tuned tree visualization.
    # --------------------------------------------------------

    tuned_figure, tuned_axis = plt.subplots(
        figsize=(20, 12)
    )

    plot_tree(
        tuned_tree,
        feature_names=all_features,
        class_names=[
            str(value)
            for value in sorted(
                y_all.unique(),
                key=str
            )
        ],
        filled=True,
        rounded=True,
        ax=tuned_axis
    )

    tuned_axis.set_title(
        "Tuned Decision Tree - IRCTC Dataset"
    )

    tuned_figure.tight_layout()

    tuned_figure.savefig(
        script_directory /
        "A8_tuned_decision_tree.png",
        dpi=200,
        bbox_inches="tight"
    )

    print(
        "Tuned tree saved as: "
        "A8_tuned_decision_tree.png"
    )

    # --------------------------------------------------------
    # Keep all matplotlib windows open until the user closes
    # them. This is important when running a normal .py file.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ALL A1-A8 TASKS COMPLETED")
    print("=" * 65)

    print("\nGenerated image files:")

    print(
        "1. A6_custom_decision_tree.png"
    )

    print(
        "2. A6_sklearn_decision_tree.png"
    )

    print(
        "3. A7_decision_boundary.png"
    )

    print(
        "4. A8_tuned_decision_tree.png"
    )

    print(
        "\nClose the plot windows to finish the program."
    )

    plt.show(block=True)



if __name__ == "__main__":
    main()
