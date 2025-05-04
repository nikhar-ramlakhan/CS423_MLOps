from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
import warnings
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
    
class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A lightweight transformer that applies one-hot encoding to a single categorical column.
    """
    def __init__(self, target_column: str) -> None:
        assert isinstance(target_column, str), f"{self.__class__.__name__} expected a string column name but got {type(target_column)} instead."
        self.target_column = target_column

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected a DataFrame but got {type(X)} instead."
        assert self.target_column in X.columns.to_list(), f"{self.__class__.__name__}.transform unknown column {self.target_column}"

        # Elegant and idiomatic: lets pandas handle column removal and dummies
        return pd.get_dummies(X, columns=[self.target_column], drop_first=False, dtype=int)



class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected a DataFrame but got {type(X)} instead.'

        X_ = X.copy()
        current_cols = set(X.columns)
        target_cols = set(self.column_list)

        if self.action == 'drop':
            # Warn for any missing columns
            missing = target_cols - current_cols
            if missing:
                warnings.warn(f'{self.__class__.__name__}: drop warning - these columns were not found and will be ignored: {missing}')
            # Drop only columns that exist, use errors='ignore' to avoid exception
            return X_.drop(columns=self.column_list, errors='ignore')

        elif self.action == 'keep':
            # Assert all columns in keep list exist
            missing = target_cols - current_cols
            assert not missing, f'{self.__class__.__name__}: keep error - these columns were not found in DataFrame: {missing}'
            return X_[self.column_list]

        return X_
    
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    def __init__(self, target_column: Hashable, fence: Literal['inner', 'outer'] = 'outer'):
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.inner_high = None
        self.outer_low = None
        self.outer_high = None

    def fit(self, X: pd.DataFrame, y=None):
        assert self.target_column in X.columns.to_list(), f"TukeyTransformer: unknown column {self.target_column}"
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f"{self.target_column} must be numeric"

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        self.inner_low = q1 - 1.5 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_low = q1 - 3.0 * iqr
        self.outer_high = q3 + 3.0 * iqr
        return self

    def transform(self, X: pd.DataFrame):
        assert self.inner_low is not None and self.outer_high is not None, "fit must be called before transform"

        X = X.copy()
        if self.fence == 'inner':
            X[self.target_column] = X[self.target_column].clip(lower=self.inner_low, upper=self.inner_high)
        elif self.fence == 'outer':
            X[self.target_column] = X[self.target_column].clip(lower=self.outer_low, upper=self.outer_high)
        else:
            raise ValueError("Fence must be either 'inner' or 'outer'")
        return X.reset_index(drop=True)

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """

  def __init__(self, column: str) -> None:
        if not isinstance(column, str):
            raise TypeError(f"Expected string for column name, got {type(column)}.")
        self.column = column
        self.median_: float | None = None
        self.iqr_: float | None = None

  def fit(self, X: pd.DataFrame, y=None) -> "CustomRobustTransformer":
      if not isinstance(X, pd.DataFrame):
          raise TypeError(f"fit() expected a DataFrame, got {type(X)}.")
      if self.column not in X.columns:
          raise AssertionError(f"Column '{self.column}' not found in input DataFrame.")
      
      series = X[self.column]
      self.median_ = series.median()
      self.iqr_ = series.quantile(0.75) - series.quantile(0.25)
      return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      if self.median_ is None or self.iqr_ is None:
          raise AssertionError("transform() called before fit().")
      if not isinstance(X, pd.DataFrame):
          raise TypeError(f"transform() expected a DataFrame, got {type(X)}.")
      
      X_copy = X.copy()
      if self.iqr_ != 0:
          X_copy[self.column] = (X_copy[self.column] - self.median_) / self.iqr_
      return X_copy

  def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
      return self.fit(X, y).transform(X)

class CustomKNNTransformer(BaseEstimator, TransformerMixin):
  """Imputes missing values using KNN.

  This transformer wraps the KNNImputer from scikit-learn and hard-codes
  add_indicator to be False. It also ensures that the input and output
  are pandas DataFrames.

  Parameters
  ----------
  n_neighbors : int, default=5
      Number of neighboring samples to use for imputation.
  weights : {'uniform', 'distance'}, default='uniform'
      Weight function used in prediction. Possible values:
      "uniform" : uniform weights. All points in each neighborhood
      are weighted equally.
      "distance" : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
  """
  #your code below

  def __init__(self, n_neighbors: PositiveInta = 5, weights: str = 'uniform'):
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, add_indicator=False)

  def fit(self, X, y=None):
    self.knn_imputer.fit(X, y)
    return self

  def transform(self, X, y=None):
    X_imputed = pd.DataFrame(self.knn_imputer.transform(X), columns=X.columns)
    return X_imputed

titanic_transformer_v4 = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ], verbose=True)

titanic_transformer_v5 = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ], verbose=True)

titanic_transformer_v6 = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('joined_ohe', CustomOHETransformer(target_column='Joined')),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scaled_age', CustomRobustTransformer(column='Age')),
    ('scaled_fare', CustomRobustTransformer(column='Fare'))
], verbose=True)

customer_transformer = Pipeline(steps=[
    ('drop_id', CustomDropColumnsTransformer(column_list=['ID'], action='drop')),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_experience', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('ohe_os', CustomOHETransformer('OS')),
    ('ohe_isp', CustomOHETransformer('ISP')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ], verbose=True)
3