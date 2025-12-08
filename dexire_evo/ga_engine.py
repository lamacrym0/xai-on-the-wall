import random, copy, csv
import numpy as np
from deap import base, creator, tools, algorithms
from dexire_evo.operators import OperatorSet

class GAEngine:
    """
    Genetic Algorithm engine for post-hoc rule extraction.
    Uses a multi-objective NSGA-II optimization to evolve
    human-readable rules approximating a trained model.

    Fitness objectives:
    1. Fidelity to the original model's predictions (maximize)
    2. Number of predicates in the ruleset (minimize)
    3. Number of uncovered samples (minimize)
    """

    def __init__(self, model_adapter, feature_names, config):
        """
        Initialize GA engine with model adapter, feature names, and config.
        :param model_adapter: Pre-trained model implementing fit() and predict().
        :param feature_names: Names of the features for printing rules.
        :param config: Configuration object (contains GA parameters).
        """
        self.model_adapter = model_adapter
        self.feature_names = feature_names
        self.cfg = config
        self.ga_params = config.ga_params

        # Set random seed for reproducibility
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)

        # GA parameters
        self.pop_size = self.ga_params.get("pop_size", 50)
        self.generations = self.ga_params.get("generations", 200)
        self.cx_prob = self.ga_params.get("cx_prob", 0.5)
        self.mut_prob = self.ga_params.get("mut_prob", 0.3)
        self.max_conditions = self.ga_params.get("max_conditions", 3)
        self.max_rules = self.ga_params.get("max_rules", 8)
        self.csv_log = self.ga_params.get("log_csv", "ga_best_by_gen.csv")

        # Load operator set from config (None = all operators)
        operators_cfg = self.ga_params.get("operators", None)
        self.operator_set = OperatorSet(operators_cfg)

    # ─────────────────────────────────────────────────────────────
    # Rule and individual generation
    # ─────────────────────────────────────────────────────────────
    def random_condition(self):
        """
        Generate a random predicate: (feature_index, threshold, operator_index).
        """
        f = random.randrange(self.N_FEAT)
        op_idx = random.randint(0, self.operator_set.size() - 1)
        return (f, random.uniform(*self.BOUNDS[f]), op_idx)

    def make_rule(self):
        """
        Generate a rule: list of conditions + class label tuple.
        Example: [(f, thr, op), ..., ('class', label, None)]
        """
        conds = [self.random_condition() for _ in range(random.randint(1, self.max_conditions))]
        return conds + [("class", random.randint(0, self.N_CLASS - 1), None)]

    # ─────────────────────────────────────────────────────────────
    # Rule evaluation and matching
    # ─────────────────────────────────────────────────────────────
    def rule_match(self, rule, sample):
        """
        Check if a sample matches a given rule's conditions.
        Returns the predicted class if matched, else None.

        :param rule: A single rule (list of conditions + class label).
        :param sample: A single data sample (feature vector).
        :return: The predicted class if matched, else None.
        """
        for f, thr, op_idx in rule[:-1]:
            op = self.operator_set.get(op_idx)
            if not op.matches(sample[f], thr):
                return None
        return rule[-1][1]

    def predict_rules(self, ind, Xmat):
        """
        Apply a ruleset to a dataset.

        :param ind: The ruleset (individual).
        :param Xmat: Dataset (2D np.ndarray).
        :return: np.ndarray of predicted labels (-1 if uncovered), number of samples with no matching rule (uncovered).
        """
        preds, uncovered = [], 0
        for s in Xmat:
            lbl = None
            for r in ind:
                out = self.rule_match(r, s)
                if out is not None:
                    lbl = out
                    break
            if lbl is None:
                uncovered += 1
                preds.append(-1)
            else:
                preds.append(lbl)
        return np.array(preds), uncovered

    def evaluate_individual(self, ind):
        """
        Compute fitness for a given ruleset.
        :param ind: The ruleset (individual).
        :return: Tuple of fitness values (fidelity, num_predicates, uncovered_samples).
        """
        y_hat, uncov = self.predict_rules(ind, self.X_train)
        valid = y_hat != -1
        fidelity = np.mean((y_hat == self.mlp_pred_tr)[valid]) if valid.any() else 0.0
        size = sum(len(r) - 1 for r in ind)
        return fidelity, size, uncov

    # ─────────────────────────────────────────────────────────────
    # Genetic operators
    # ─────────────────────────────────────────────────────────────
    def crossover_rules(self, i1, i2):
        """Two-point crossover between two individuals.
        :param i1: First individual (ruleset).
        :param i2: Second individual (ruleset).
        :return: Two offspring individuals.
        """
        return tools.cxTwoPoint(i1, i2) if len(i1) > 1 and len(i2) > 1 else (i1, i2)

    def mutate_individual(self, ind, indpb=0.2):
        """
        Mutation operator:
        - Add or remove rules
        - Modify thresholds or operators in conditions
        - Change predicted class of a rule

        :param ind: Individual to mutate.
        :param indpb: Probability of mutation for each component.
        :return: Mutated individual.
        """
        # Add new rule
        if random.random() < indpb and len(ind) < self.max_rules:
            ind.append(self.make_rule())

        # Remove a rule
        if random.random() < indpb and len(ind) > 2:
            ind.pop(random.randrange(len(ind)))

        # Mutate conditions
        for r in ind:
            for idx, (f, thr, op_idx) in enumerate(r[:-1]):
                if random.random() < indpb:
                    if random.random() < 0.5:
                        thr = random.uniform(*self.BOUNDS[f])
                    else:
                        op_idx = random.randint(0, self.operator_set.size() - 1)
                    r[idx] = (f, thr, op_idx)

            # Mutate class label
            if random.random() < indpb:
                r[-1] = ("class", random.randint(0, self.N_CLASS - 1), None)

        return ind,

    # ─────────────────────────────────────────────────────────────
    # GA setup and execution
    # ─────────────────────────────────────────────────────────────
    def _init_ga(self, X_train, y_train):
        """
        Initialize GA state and DEAP toolbox.
        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :return: DEAP toolbox, sorting key, and ruleset string formatter.
        """
        # Save training data references for evaluation
        self.X_train = X_train
        self.N_FEAT = X_train.shape[1]
        self.N_CLASS = len(np.unique(y_train))
        self.BOUNDS = [(float(X_train[:, i].min()), float(X_train[:, i].max())) for i in range(self.N_FEAT)]
        self.mlp_pred_tr = self.model_adapter.predict(X_train)

        # Define DEAP structures
        creator.create("FitTri", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitTri)
        toolbox = base.Toolbox()

        # Register GA components
        toolbox.register("individual", lambda: creator.Individual(
            [self.make_rule() for _ in range(random.randint(2, self.max_rules))]
        ))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.crossover_rules)
        toolbox.register("mutate", self.mutate_individual, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        # Sort and pretty-print helpers
        sort_key = lambda ind: (-ind.fitness.values[0], ind.fitness.values[2], ind.fitness.values[1])
        rules_to_str = lambda ind: " | ".join(
            f"[{' AND '.join(f'{self.feature_names[f]} {self.operator_set.get(op_idx).symbol} {thr:.2f}' for f, thr, op_idx in r[:-1])}]→{r[-1][1]}"
            for r in ind
        )

        return toolbox, sort_key, rules_to_str

    def evolve(self, X_train, y_train):
        """
        Main GA loop.
        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :return: GA object, the best individual (ruleset) after evolution.
        """
        toolbox, sort_key, rules_to_str = self._init_ga(X_train, y_train)
        pop = toolbox.population(self.pop_size)

        # Evaluate initial population
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        with open(self.csv_log, "w", newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "fidelity", "predicates", "uncovered", "rules"])

            for g in range(1, self.generations + 1):
                # Log best of current generation
                best_gen = min(pop, key=sort_key)
                writer.writerow([g, *best_gen.fitness.values, rules_to_str(best_gen)])

                if g == 1 or g % 10 == 0 or g == self.generations:
                    print(f"GEN {g:03d}: fid={best_gen.fitness.values[0]:.3f}, "
                          f"preds={best_gen.fitness.values[1]}, uncov={best_gen.fitness.values[2]}")

                # Generate offspring
                offspring = algorithms.varOr(pop, toolbox, lambda_=self.pop_size,
                                              cxpb=self.cx_prob, mutpb=self.mut_prob)

                # Evaluate new individuals
                for ind in offspring:
                    if not ind.fitness.valid:
                        ind.fitness.values = toolbox.evaluate(ind)

                # Combine populations and select next generation
                combined = pop + offspring
                elite = min(combined, key=sort_key)
                pop = toolbox.select(combined, self.pop_size)

                # Enforce elitism
                if elite not in pop:
                    pop[-1] = copy.deepcopy(elite)

        # Return best from the first non-dominated front
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        return min(front, key=sort_key)
