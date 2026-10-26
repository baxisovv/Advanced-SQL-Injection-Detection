import base64
import html
import math
import os
import re
import urllib.parse

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (logger, MODEL_SAVE_DIR, TFIDF_MAX_FEATURES, NGRAM_RANGE,
                    ENABLE_CHI_SQUARE_SELECTION, CHI_SQUARE_K_FEATURES, 
                    CHI_SQUARE_PERCENTILE, CHI_SQUARE_P_VALUE_THRESHOLD)


class SQLPreprocessor:
    """
    Enhanced SQL Preprocessor with Chi-Square Feature Selection
    
    This preprocessor extracts structural and TF-IDF features from SQL queries,
    with optional chi-square based feature selection for improved performance.
    """

    def __init__(self, use_tfidf: bool = True, use_chi_square_selection: bool = None,
                 chi_square_config: dict = None):
        self.use_tfidf = use_tfidf
        self.tfidf: TfidfVectorizer | None = None
        
        # Use config default if not specified
        if use_chi_square_selection is None:
            use_chi_square_selection = ENABLE_CHI_SQUARE_SELECTION
        self.use_chi_square_selection = use_chi_square_selection
        
        self.feature_selector = None
        self.chi_square_config = chi_square_config or {}

        # Extended structural features (12 -> 22 features)
        self.structural_feature_names = [
            # Original features
            'length', 'entropy', 'special_char_ratio', 'digit_ratio',
            'alpha_ratio', 'whitespace_ratio', 'max_consecutive_special',
            'max_consecutive_digits', 'unique_char_ratio', 'token_count',
            'avg_token_length', 'punctuation_cluster_count',
            # New extended features
            'uppercase_ratio', 'parenthesis_depth', 'quote_mismatch_score',
            'semicolon_count', 'comment_indicator_count', 'url_encoding_density',
            'hex_encoding_count', 'concatenation_operators', 
            'logical_operator_count', 'comparison_operator_count'
        ]

    @property
    def feature_names(self) -> list[str]:
        """Get current feature names (after selection if applicable)"""
        if self.feature_selector is not None:
            return self.feature_selector.selected_feature_names_
        
        names = list(self.structural_feature_names)
        if self.use_tfidf and self.tfidf is not None:
            n = len(self.tfidf.vocabulary_)
            names += [f'tfidf_{i}' for i in range(n)]
        return names

    @staticmethod
    def _decode_base64_segments(text: str) -> str:
        pattern = r'(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])'

        def _try_decode(m: re.Match) -> str:
            blob = m.group(1)
            padded = blob + '=' * (-len(blob) % 4)
            try:
                decoded = base64.b64decode(padded).decode('utf-8', errors='strict')
                if decoded and all(32 <= ord(c) < 127 for c in decoded):
                    return decoded
            except Exception:
                pass
            return blob

        return re.sub(pattern, _try_decode, text)

    def advanced_normalize(self, text: str) -> str:
        text = str(text).lower()

        text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)
        text = re.sub(r'%u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)

        text = urllib.parse.unquote(urllib.parse.unquote(text))
        text = html.unescape(text)
        text = urllib.parse.unquote(text)
        
        # Remove comments to analyze the core payload structure
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        def _hex_replacer(m: re.Match) -> str:
            try:
                return bytearray.fromhex(m.group(0)[2:]).decode('utf-8', errors='ignore')
            except ValueError:
                return m.group(0)

        text = re.sub(r'0x[0-9a-f]+', _hex_replacer, text)
        text = self._decode_base64_segments(text)

        return text

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(text)]
        return -sum(p * math.log2(p) for p in prob)

    @staticmethod
    def _max_consecutive(text: str, pattern: str) -> int:
        matches = re.findall(pattern, text)
        return max((len(m) for m in matches), default=0)

    def _extended_structural_features(self, queries) -> csr_matrix:
        """Extract comprehensive structural features including SQL-specific patterns"""
        rows = []
        for q in queries:
            norm = self.advanced_normalize(q)
            original = str(q)  # Keep original for case-sensitive features
            length = len(norm)
            
            if length == 0:
                rows.append([0.0] * len(self.structural_feature_names))
                continue

            # Basic distribution features
            entropy = self._entropy(norm)
            n_special = len(re.findall(r'[^a-zA-Z0-9\s]', norm))
            n_digits = len(re.findall(r'[0-9]', norm))
            n_alpha = len(re.findall(r'[a-zA-Z]', norm))
            n_spaces = len(re.findall(r'\s', norm))
            
            # Consecutive sequence lengths
            max_cons_spec = self._max_consecutive(norm, r'[^a-zA-Z0-9\s]+')
            max_cons_dig = self._max_consecutive(norm, r'[0-9]+')

            # Extended Lexical / Statistical Features
            unique_char_ratio = len(set(norm)) / length
            
            tokens = norm.split()
            token_count = len(tokens)
            avg_token_length = np.mean([len(t) for t in tokens]) if token_count > 0 else 0.0
            
            # Punctuation clusters
            punct_clusters = len(re.findall(r'[^a-zA-Z0-9\s]{2,}', norm))
            
            # Ratios
            special_ratio = n_special / length
            digit_ratio = n_digits / length
            alpha_ratio = n_alpha / length
            space_ratio = n_spaces / length

            # NEW EXTENDED FEATURES
            # Uppercase ratio (SQL keywords often uppercase)
            n_uppercase = sum(1 for c in original if c.isupper())
            uppercase_ratio = n_uppercase / len(original) if len(original) > 0 else 0.0
            
            # Parenthesis depth (nested queries)
            max_depth = 0
            current_depth = 0
            for c in norm:
                if c == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif c == ')':
                    current_depth = max(0, current_depth - 1)
            parenthesis_depth = max_depth
            
            # Quote mismatch score (unbalanced quotes)
            single_quotes = norm.count("'")
            double_quotes = norm.count('"')
            quote_mismatch = abs(single_quotes % 2) + abs(double_quotes % 2)
            
            # Semicolon count (statement chaining)
            semicolon_count = norm.count(';')
            
            # Comment indicators (-- or /* patterns)
            comment_count = len(re.findall(r'--|/\*|\*/', norm))
            
            # URL encoding density (%XX patterns)
            url_encoded = len(re.findall(r'%[0-9a-fA-F]{2}', original))
            url_encoding_density = url_encoded / length
            
            # Hex encoding count (0x patterns)
            hex_count = len(re.findall(r'0x[0-9a-fA-F]+', norm))
            
            # Concatenation operators (|| or + patterns for string concat)
            concat_ops = len(re.findall(r'\|\||(?<!\w)\+(?!\w)', norm))
            
            # Logical operators (AND, OR patterns)
            logical_ops = len(re.findall(r'\band\b|\bor\b', norm))
            
            # Comparison operators (=, !=, <, >, <=, >= patterns)
            comparison_ops = len(re.findall(r'[<>!=]=?', norm))

            rows.append([
                # Original 12 features
                length, entropy, special_ratio, digit_ratio,
                alpha_ratio, space_ratio, max_cons_spec, max_cons_dig,
                unique_char_ratio, token_count, avg_token_length, punct_clusters,
                # New 10 features
                uppercase_ratio, parenthesis_depth, quote_mismatch,
                semicolon_count, comment_count, url_encoding_density,
                hex_count, concat_ops, logical_ops, comparison_ops
            ])
            
        return csr_matrix(rows)

    def fit_transform(self, X, y=None) -> csr_matrix:
        queries = X['sql_query'].values

        logger.info("Extracting extended structural features (22 features)...")
        structural = self._extended_structural_features(queries)

        if self.use_tfidf:
            logger.info(
                f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}, "
                f"ngram_range={NGRAM_RANGE})..."
            )
            self.tfidf = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                sublinear_tf=True,
                min_df=0.005,
                max_df=0.5
            )
            norm_queries = [self.advanced_normalize(q) for q in queries]
            tfidf_mat = self.tfidf.fit_transform(norm_queries)
            logger.info(f"TF-IDF matrix shape: {tfidf_mat.shape}")
            
            combined_features = hstack([structural, tfidf_mat], format='csr')
            
            # Apply chi-square feature selection if enabled
            if self.use_chi_square_selection and y is not None:
                from feature_selector import ChiSquareFeatureSelector
                
                logger.info("Applying chi-square feature selection...")
                
                # Build feature names before selection
                all_feature_names = list(self.structural_feature_names)
                all_feature_names += [f'tfidf_{i}' for i in range(tfidf_mat.shape[1])]
                
                self.feature_selector = ChiSquareFeatureSelector(
                    k=self.chi_square_config.get('k', CHI_SQUARE_K_FEATURES),
                    percentile=self.chi_square_config.get('percentile', CHI_SQUARE_PERCENTILE),
                    p_value_threshold=self.chi_square_config.get('p_value_threshold', 
                                                                 CHI_SQUARE_P_VALUE_THRESHOLD)
                )
                
                selected_features = self.feature_selector.fit_transform(
                    combined_features, y,
                    feature_names=all_feature_names,
                    n_structural_features=len(self.structural_feature_names)
                )
                
                # Generate analysis plots
                try:
                    self.feature_selector.plot_feature_selection_analysis()
                except Exception as e:
                    logger.warning(f"Could not generate feature selection plots: {e}")
                
                return selected_features
            
            return combined_features

        return structural

    def transform(self, X) -> csr_matrix:
        queries = X['sql_query'].values
        structural = self._extended_structural_features(queries)

        if self.use_tfidf:
            if self.tfidf is None:
                raise RuntimeError("Call fit_transform before transform.")
            norm_queries = [self.advanced_normalize(q) for q in queries]
            tfidf_mat = self.tfidf.transform(norm_queries)
            
            combined_features = hstack([structural, tfidf_mat], format='csr')
            
            # Apply feature selection if it was fitted
            if self.feature_selector is not None:
                return self.feature_selector.transform(combined_features)
            
            return combined_features

        return structural

    def save(self, suffix: str = '') -> None:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        fname = f'preprocessor{"_" + suffix if suffix else ""}.pkl'
        joblib.dump(self, os.path.join(MODEL_SAVE_DIR, fname))
        logger.info(f"Preprocessor saved -> {os.path.join(MODEL_SAVE_DIR, fname)}")
