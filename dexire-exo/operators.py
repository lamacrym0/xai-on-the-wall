import operator

class RuleOperator:
    """Represents a comparison operator used in rules."""
    def __init__(self, symbol: str, func):
        """
        Initializes a RuleOperator, given its symbol and function.
        :param symbol: The string symbol of the operator (e.g., "<=", ">", etc.)
        :param func: The function implementing the operator logic.
        """
        self.symbol = symbol
        self.func = func

    def matches(self, value, threshold):
        return self.func(value, threshold)

# List of all supported operators
ALL_OPERATORS = {
    "<=": RuleOperator("<=", operator.le),
    ">":  RuleOperator(">", operator.gt),
    "<":  RuleOperator("<", operator.lt),
    ">=": RuleOperator(">=", operator.ge),
    "==": RuleOperator("==", operator.eq),
    "!=": RuleOperator("!=", operator.ne),
}

class OperatorSet:
    """Holds the active set of operators for the GA."""
    def __init__(self, symbols):
        """
        Initializes the OperatorSet with the given operator symbols.
        :param symbols: List of operator symbols to include in the set. If None, includes all operators.
        """
        if symbols is None:
            symbols = list(ALL_OPERATORS.keys())

        missing = [s for s in symbols if s not in ALL_OPERATORS]
        if missing:
            raise ValueError(f"Unknown operators in config: {missing}")

        self.operators = [ALL_OPERATORS[s] for s in symbols]

    def get(self, idx: int) -> RuleOperator:
        """
        Retrieves the operator at the given index.
        :param idx: Index of the operator to retrieve.
        :return: The RuleOperator at the specified index.
        """
        return self.operators[idx]

    def size(self) -> int:
        """
        Returns the number of operators in the set.
        :return: The size of the operator set.
        """
        return len(self.operators)
