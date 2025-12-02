# =============================
# DEXiRE CLASS (MODEL-AGNOSTIC)
# =============================

import numpy as np
from typing import List, Union
from sklearn.model_selection import train_test_split

from .core.dexire_abstract import (AbstractRuleExtractor,
                                   AbstractRuleSet,
                                   Mode,
                                   RuleExtractorEnum,
                                   TiebreakerStrategy)
from .rule_extractors.tree_rule_extractor import TreeRuleExtractor
from .rule_extractors.one_rule_extractor import OneRuleExtractor
from .core.rule_set import RuleSet
from .utils.activation_discretizer import discretize_activation_layer
from dexire.adapters.model_adapter import AbstractModelAdapter


class DEXiRE:
    """Deep Explanations and Rule Extraction pipeline to extract rules from a deep neural network."""
    def __init__(self,
                 model: AbstractModelAdapter,
                 feature_names: List[str] = None,
                 class_names: List[str] = None,
                 rule_extractor: Union[None, AbstractRuleExtractor] = None,
                 mode: Mode = Mode.CLASSIFICATION,
                 explain_features: np.array = None,
                 rule_extraction_method: RuleExtractorEnum = RuleExtractorEnum.TREERULE,
                 tie_breaker_strategy: TiebreakerStrategy = TiebreakerStrategy.FIRST_HIT_RULE) -> None:

        self.model = model
        self.explain_features = explain_features
        self.mode = mode
        self.rule_extraction_method = rule_extraction_method
        self.tie_breaker_strategy = tie_breaker_strategy
        self.rule_extractor = rule_extractor
        self.features_names = feature_names
        self.class_names = class_names
        self.intermediate_rules = {}
        self.data_raw = {}
        self.final_rule_set = None
        self.data_transformed = {}

        if self.rule_extractor is None:
            if self.mode not in [Mode.CLASSIFICATION, Mode.REGRESSION]:
                raise Exception(f"Not implemented mode: {self.mode}.")
            if self.rule_extraction_method not in RuleExtractorEnum:
                raise Exception(f"Rule extractor {self.rule_extraction_method} not implemented.")

            if self.rule_extraction_method == RuleExtractorEnum.ONERULE:
                self.rule_extractor = {RuleExtractorEnum.ONERULE: OneRuleExtractor(mode=self.mode)}
            elif self.rule_extraction_method == RuleExtractorEnum.TREERULE:
                self.rule_extractor = {
                    RuleExtractorEnum.TREERULE: TreeRuleExtractor(max_depth=200, mode=self.mode, class_names=self.class_names)
                }
            elif self.rule_extraction_method == RuleExtractorEnum.MIXED:
                self.rule_extractor = {
                    RuleExtractorEnum.ONERULE: OneRuleExtractor(mode=self.mode),
                    RuleExtractorEnum.TREERULE: TreeRuleExtractor(max_depth=200, mode=self.mode, class_names=self.class_names)
                }

    def extract_rules_at_layer(self,
                               X: np.array = None,
                               y: np.array = None,
                               layer_idx: int = -2,
                               sample: float = None,
                               quantize: bool = True,
                               n_bins: int = 2,
                               express_as_basic_features: bool = True,
                               random_state: int = 41) -> List[AbstractRuleSet]:

        self.data_raw['inputs'] = X
        self.data_raw['output'] = y

        if "raw_prediction" in self.data_raw:
            y_pred_raw = self.data_raw['raw_prediction']
        else:
            y_pred_raw = self.model.predict(X)
            self.data_raw['raw_prediction'] = y_pred_raw

        if self.mode == Mode.CLASSIFICATION:
            pred_shape = y_pred_raw.shape[1]
            if pred_shape == 1:
                y_pred = np.rint(y_pred_raw)
            elif pred_shape > 1:
                y_pred = np.argmax(y_pred_raw, axis=1)
            else:
                raise Exception(f"Unexpected prediction shape: {y_pred_raw.shape}")
            classes, counts = np.unique(y_pred, return_counts=True)
            self.majority_class = classes[np.argmax(counts)]
        elif self.mode == Mode.REGRESSION:
            y_pred = y_pred_raw
            self.majority_class = np.mean(y_pred)
        else:
            raise Exception(f"Unsupported mode: {self.mode}")

        intermediate_output = self.model.get_layer_output(X, layer_idx)
        if quantize:
            intermediate_output = discretize_activation_layer(intermediate_output, n_bins=n_bins)

        x = intermediate_output
        y = y_pred

        if sample is not None:
            stratify = y if self.mode == Mode.CLASSIFICATION else None
            _, x, _, y = train_test_split(x, y, test_size=sample, stratify=stratify, random_state=random_state)

        extractor = self.rule_extractor[self.rule_extraction_method]
        rules = extractor.extract_rules(x, y)

        y_rule = rules.predict_numpy_rules(x, tie_breaker_strategy=self.tie_breaker_strategy)

        if express_as_basic_features:
            X_xai = self.explain_features if self.explain_features is not None else X
            if self.features_names is not None:
                if len(self.features_names) != X_xai.shape[1]:
                    raise ValueError("Mismatch between feature names and input shape.")
            else:
                self.features_names = [f"X_{i}" for i in range(X_xai.shape[1])]

            extractor = self.rule_extractor.get(RuleExtractorEnum.TREERULE, extractor)
            rules_features = extractor.extract_rules(X_xai, y_rule, feature_names=self.features_names)
        else:
            rules_features = self.rule_extractor[RuleExtractorEnum.ONERULE].extract_rules(X, y_rule)

        self.intermediate_rules[layer_idx] = {'final_rules': rules, 'raw_rules': rules_features}
        return rules_features

    def extract_rules(self,
                      X: np.array,
                      y: np.array,
                      sample: float = None,
                      layer_idx: List[int] = None) -> AbstractRuleSet:

        if layer_idx is None:
            candidate_layers = self.model.get_candidate_layer_indices()
        else:
            candidate_layers = layer_idx if isinstance(layer_idx, list) else [layer_idx]

        partial_rule_sets = [self.extract_rules_at_layer(X, y, layer_idx=idx, sample=sample) for idx in candidate_layers]

        total_rules = []
        for rs in partial_rule_sets:
            total_rules += rs.get_rules()

        total_rules = list(set(total_rules))
        frs = RuleSet()
        frs.add_rules(total_rules)
        self.final_rule_set = frs
        return frs