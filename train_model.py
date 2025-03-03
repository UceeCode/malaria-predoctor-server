import pandas as pd # helps organize patients data in tables like excel sheets
import numpy as np # helps with numbers and calculations
from sklearn.model_selection import train_test_split # split data into "study" and "test" groups
from sklearn.preprocessing import StandardScaler # makes all numbers even
from sklearn.linear_model import LogisticRegression # the ai that will learn
from sklearn.metrics import accuracy_score # checks how good the ai is
import pickle # saves the ai trained data for later use
