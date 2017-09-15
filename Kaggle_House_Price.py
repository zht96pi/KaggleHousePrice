####################################################################################################################

import numpy as np 
import pandas as pd 
import xgboost as xgb
from subprocess import check_output
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

################################################### 读入数据 #############################################################


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


################################################### 特征补全 #############################################################

# 补全LotFrontage
# 通过计算其与LotArea的关系可以知道其有一定相关性，所以用边长补充，好像除以1.5就差不多了
# （法一）缺失值用房屋边长补全
test_data['SqrtLotArea'] = np.sqrt(test_data['LotArea'])
train_data['SqrtLotArea'] = np.sqrt(train_data['LotArea'])
cond = test_data['LotFrontage'].isnull()
test_data.LotFrontage[cond] = test_data.SqrtLotArea[cond]        
del test_data['SqrtLotArea']
del train_data['SqrtLotArea']
# （法二）缺失值用中位数来补全
# test_data.LotFrontage[cond] = test_data['LotFrontage'].median()
# （法三）其实缺失值和log(LotArea)相关系数更高(还没有尝试)
# （法四）用分组的中位数填充
tmp = pd.DataFrame(index = train_data.index)
lot_frontage_by_neighborhood = train_data['LotFrontage'].groupby(train_data['Neighborhood'])
tmp['LotFrontage'] = train_data['LotFrontage']
tmp['Neighborhood'] = train_data['Neighborhood']
for key, group in lot_frontage_by_neighborhood:
    idx = (tmp['Neighborhood'] == key) & (tmp['LotFrontage'].isnull())
    train_data.loc[idx, 'LotFrontage'] = group.median()  

# 补全MSZoning
# 在test测试集中有缺失, train中没有
# MSSubClass，MSZoning有一定关系
# pd.crosstab(test_data.MSSubClass, test_data.MSZoning)
# test_data中建筑类型缺失值补齐 30:RM 20:RL 70:RM
test_data.loc[test_data['MSSubClass'] == 20, 'MSZoning'] = 'RL'
test_data.loc[test_data['MSSubClass'] == 30, 'MSZoning'] = 'RM'
test_data.loc[test_data['MSSubClass'] == 70, 'MSZoning'] = 'RM'


# 补全Exterior1st & Exterior2nd
# 只在test中出现缺失值(nans only appear in test set)
# 检查Exterior1st 和 Exterior2nd 是否存在缺失值共现的情况
# 这里两个补全的值分别是选择了整体的众数和按年份分组的众数，可以调整
#test_data.loc[test_data['Exterior1st'].isnull(), 'Exterior1st'] = 'VinylSd'
#test_data.loc[test_data['Exterior2nd'].isnull(), 'Exterior2nd'] = 'VinylSd'
test_data.loc[test_data['Exterior1st'].isnull(), 'Exterior1st'] = 'Wd Sdng'
test_data.loc[test_data['Exterior2nd'].isnull(), 'Exterior2nd'] = 'Wd Sdng'


# 补全KitchenQual
# 只在测试集中有缺失值
test_data.loc[test_data['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'

# 补全Functional
# 只在测试集中有缺失值
# 填充一个最常见的值
test_data.loc[test_data['Functional'].isnull(), 'Functional'] = 'Typ'

# 补全basement
basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
for cols in basement_cols:
    if 'FinFS' not in cols:#判断字段中是否包含'FinFS'
        train_data.loc[train_data[cols].isnull(), cols] = 'None'
basement_cols = ['Id', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
# 其中,有三行只有BsmtCond为NaN,该三行的其他列均有值 580 725 1064
test_data.loc[test_data['Id'] == 580, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 725, 'BsmtCond'] = 'TA'
test_data.loc[test_data['Id'] == 1064, 'BsmtCond'] = 'TA'
for cols in basement_cols:
    if cols not in 'SF' and cols not in 'Bath':
        test_data.loc[test_data['BsmtFinSF1'] == 0.0, cols] = 'None'
for cols in basement_cols:
    if test_data[cols].dtype == np.object:
        test_data.loc[test_data[cols].isnull(), cols] = 'None'
    else:
        test_data.loc[test_data[cols].isnull(), cols] = 0.0


# 补全Garage 车库
garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
for cols in garage_cols:
    if train_data[cols].dtype == np.object:
        train_data.loc[train_data[cols].isnull(), cols] = 'None'
    else:
        train_data.loc[train_data[cols].isnull(), cols] = 0

garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
for cols in garage_cols:
    if test_data[cols].dtype == np.object:
        test_data.loc[test_data[cols].isnull(), cols] = 'None'
    else:
        test_data.loc[test_data[cols].isnull(), cols] = 0
test_data.loc[666, "GarageQual"] = "TA"
test_data.loc[666, "GarageCond"] = "TA"
test_data.loc[666, "GarageFinish"] = "Unf"
test_data.loc[666, "GarageYrBlt"] = "1980"

# 补全SaleType
# nans only appear in test set
test_data.loc[test_data['SaleType'].isnull(), 'SaleType'] = 'WD'

# 补全Electrical
# nans only appear in train set
train_data.loc[train_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'


test_data.loc[test_data['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
train_data.loc[train_data['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
test_data.loc[test_data['FireplaceQu'].isnull(), 'FireplaceQu'] = 'None'
train_data.loc[train_data['FireplaceQu'].isnull(), 'FireplaceQu'] = 'None'
test_data.loc[test_data['Fence'].isnull(), 'Fence'] = 'None'
train_data.loc[train_data['Fence'].isnull(), 'Fence'] = 'None'
test_data.loc[test_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
train_data.loc[train_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
test_data.loc[test_data['MasVnrArea'].isnull(), 'MasVnrArea'] = 0.0
train_data.loc[train_data['MasVnrArea'].isnull(), 'MasVnrArea'] = 0.0
test_data.loc[test_data['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = '0'
test_data.loc[test_data['BsmtFinSF2'].isnull(), 'BsmtFinSF2'] = '0'
test_data.loc[test_data['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = '0'
test_data.loc[test_data['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = '0'
test_data.loc[test_data['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0
test_data.loc[test_data['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0

################################################### 特征转换 #############################################################

train_data = train_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                                60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                                90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})
test_data = test_data.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                              60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                              90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})


train_data = train_data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5, "None" : 0},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )

test_data = test_data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5, "None" : 0},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )

################################################### 统一两个数据的类型 #############################################################

# 一部分是int64，一部分是float64
c = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
for cols in c:
    tmp_col = test_data[cols].astype(pd.np.int64)
    tmp_col = pd.DataFrame({cols: tmp_col})
    del test_data[cols]
    test_data = pd.concat((test_data, tmp_col), axis=1)     


################################################### 新增特征 #############################################################

train_data['1stFlr_2ndFlr_Sf'] = np.log1p(train_data['1stFlrSF'] + train_data['2ndFlrSF'])
test_data['1stFlr_2ndFlr_Sf'] = np.log1p(test_data['1stFlrSF'] + test_data['2ndFlrSF'])

train_data['All_Liv_SF'] = np.log1p(train_data['1stFlr_2ndFlr_Sf'] + train_data['LowQualFinSF'] + train_data['GrLivArea'])
test_data['All_Liv_SF'] = np.log1p(test_data['1stFlr_2ndFlr_Sf'] + test_data['LowQualFinSF'] + test_data['GrLivArea'])

train_data.drop(['1stFlrSF'], axis = 1)
test_data.drop(['1stFlrSF'], axis = 1)
train_data.drop(['2ndFlrSF'], axis = 1)
test_data.drop(['2ndFlrSF'], axis = 1)


test_data["SaleCondition_PriceDown"] = test_data.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

train_data["SaleCondition_PriceDown"] = train_data.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})


train_data['SimplOverallQual'] = train_data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
train_data['SimplOverallCond'] = train_data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
train_data['SimplPoolQC'] = train_data.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
train_data['SimplGarageCond'] = train_data.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
train_data['SimplGarageQual'] = train_data.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
train_data['SimplFireplaceQu'] = train_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train_data['SimplFireplaceQu'] = train_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train_data['SimplFunctional'] = train_data.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
train_data['SimplKitchenQual'] = train_data.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train_data['SimplHeatingQC'] = train_data.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
train_data['SimplBsmtFinType1'] = train_data.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
train_data['SimplBsmtFinType2'] = train_data.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
train_data['SimplBsmtCond'] = train_data.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
train_data['SimplBsmtQual'] = train_data.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
train_data['SimplExterCond'] = train_data.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
train_data['SimplExterQual'] = train_data.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

# 2* Combinations of existing features
# Overall quality of the house
train_data['OverallGrade'] = train_data['OverallQual'] * train_data['OverallCond']
# Overall quality of the garage
train_data['GarageGrade'] = train_data['GarageQual'] * train_data['GarageCond']
# Overall quality of the exterior
train_data['ExterGrade'] = train_data['ExterQual'] * train_data['ExterCond']
# Overall kitchen score
train_data['KitchenScore'] = train_data['KitchenAbvGr'] * train_data['KitchenQual']
# Overall fireplace score
train_data['FireplaceScore'] = train_data['Fireplaces'] * train_data['FireplaceQu']
# Overall garage score
train_data['GarageScore'] = train_data['GarageArea'] * train_data['GarageQual']
# Overall pool score
train_data['PoolScore'] = train_data['PoolArea'] * train_data['PoolQC']
# Simplified overall quality of the house
train_data['SimplOverallGrade'] = train_data['SimplOverallQual'] * train_data['SimplOverallCond']
# Simplified overall quality of the exterior
train_data['SimplExterGrade'] = train_data['SimplExterQual'] * train_data['SimplExterCond']
# Simplified overall pool score
train_data['SimplPoolScore'] = train_data['PoolArea'] * train_data['SimplPoolQC']
# Simplified overall garage score
train_data['SimplGarageScore'] = train_data['GarageArea'] * train_data['SimplGarageQual']
# Simplified overall fireplace score
train_data['SimplFireplaceScore'] = train_data['Fireplaces'] * train_data['SimplFireplaceQu']
# Simplified overall kitchen score
train_data['SimplKitchenScore'] = train_data['KitchenAbvGr'] * train_data['SimplKitchenQual']
# Total number of bathrooms
train_data['TotalBath'] = train_data['BsmtFullBath'] + (0.5 * train_data['BsmtHalfBath']) + \
train_data['FullBath'] + (0.5 * train_data['HalfBath'])
# Total SF for house (incl. basement)
train_data['AllSF'] = train_data['GrLivArea'] + train_data['TotalBsmtSF']
# Total SF for 1st + 2nd floors
train_data['AllFlrsSF'] = train_data['1stFlrSF'] + train_data['2ndFlrSF']
# Total SF for porch
train_data['AllPorchSF'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + \
train_data['3SsnPorch'] + train_data['ScreenPorch']
# Has masonry veneer or not
train_data['HasMasVnr'] = train_data.MasVnrType.replace({'BrkCmn' : 1, 'BrkFace' : 1, 'CBlock' : 1, 
                                               'Stone' : 1, 'None' : 0})
# House completed before sale or not
train_data['BoughtOffPlan'] = train_data.SaleCondition.replace({'Abnorml' : 0, 'Alloca' : 0, 'AdjLand' : 0, 
                                                      'Family' : 0, 'Normal' : 0, 'Partial' : 1})

test_data['SimplOverallQual'] = test_data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
test_data['SimplOverallCond'] = test_data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
test_data['SimplPoolQC'] = test_data.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
test_data['SimplGarageCond'] = test_data.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
test_data['SimplGarageQual'] = test_data.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
test_data['SimplFireplaceQu'] = test_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
test_data['SimplFireplaceQu'] = test_data.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
test_data['SimplFunctional'] = test_data.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
test_data['SimplKitchenQual'] = test_data.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
test_data['SimplHeatingQC'] = test_data.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
test_data['SimplBsmtFinType1'] = test_data.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
test_data['SimplBsmtFinType2'] = test_data.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
test_data['SimplBsmtCond'] = test_data.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
test_data['SimplBsmtQual'] = test_data.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
test_data['SimplExterCond'] = test_data.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
test_data['SimplExterQual'] = test_data.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

# 2* Combinations of existing features
# Overall quality of the house
test_data['OverallGrade'] = test_data['OverallQual'] * test_data['OverallCond']
# Overall quality of the garage
test_data['GarageGrade'] = test_data['GarageQual'] * test_data['GarageCond']
# Overall quality of the exterior
test_data['ExterGrade'] = test_data['ExterQual'] * test_data['ExterCond']
# Overall kitchen score
test_data['KitchenScore'] = test_data['KitchenAbvGr'] * test_data['KitchenQual']
# Overall fireplace score
test_data['FireplaceScore'] = test_data['Fireplaces'] * test_data['FireplaceQu']
# Overall garage score
test_data['GarageScore'] = test_data['GarageArea'] * test_data['GarageQual']
# Overall pool score
test_data['PoolScore'] = test_data['PoolArea'] * test_data['PoolQC']
# Simplified overall quality of the house
test_data['SimplOverallGrade'] = test_data['SimplOverallQual'] * test_data['SimplOverallCond']
# Simplified overall quality of the exterior
test_data['SimplExterGrade'] = test_data['SimplExterQual'] * test_data['SimplExterCond']
# Simplified overall pool score
test_data['SimplPoolScore'] = test_data['PoolArea'] * test_data['SimplPoolQC']
# Simplified overall garage score
test_data['SimplGarageScore'] = test_data['GarageArea'] * test_data['SimplGarageQual']
# Simplified overall fireplace score
test_data['SimplFireplaceScore'] = test_data['Fireplaces'] * test_data['SimplFireplaceQu']
# Simplified overall kitchen score
test_data['SimplKitchenScore'] = test_data['KitchenAbvGr'] * test_data['SimplKitchenQual']
# Total number of bathrooms
test_data['TotalBath'] = test_data['BsmtFullBath'] + (0.5 * test_data['BsmtHalfBath']) + \
test_data['FullBath'] + (0.5 * test_data['HalfBath'])
# Total SF for house (incl. basement)
test_data['AllSF'] = test_data['GrLivArea'] + test_data['TotalBsmtSF']
# Total SF for 1st + 2nd floors
test_data['AllFlrsSF'] = test_data['1stFlrSF'] + test_data['2ndFlrSF']
# Total SF for porch
test_data['AllPorchSF'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + \
test_data['3SsnPorch'] + test_data['ScreenPorch']
# Has masonry veneer or not
test_data['HasMasVnr'] = test_data.MasVnrType.replace({'BrkCmn' : 1, 'BrkFace' : 1, 'CBlock' : 1, 
                                               'Stone' : 1, 'None' : 0})
# House completed before sale or not
test_data['BoughtOffPlan'] = test_data.SaleCondition.replace({'Abnorml' : 0, 'Alloca' : 0, 'AdjLand' : 0, 
                                                      'Family' : 0, 'Normal' : 0, 'Partial' : 1})
train_data['OverallQual-s2'] = train_data['OverallQual'] ** 2
train_data['OverallQual-s3'] = train_data['OverallQual'] ** 3
train_data['OverallQual-Sq'] = np.sqrt(train_data['OverallQual'])
train_data['AllSF-2'] = train_data['AllSF'] ** 2
train_data['AllSF-3'] = train_data['AllSF'] ** 3
train_data['AllSF-Sq'] = np.sqrt(train_data['AllSF'])
train_data['AllFlrsSF-2'] = train_data['AllFlrsSF'] ** 2
train_data['AllFlrsSF-3'] = train_data['AllFlrsSF'] ** 3
train_data['AllFlrsSF-Sq'] = np.sqrt(train_data['AllFlrsSF'])
train_data['GrLivArea-2'] = train_data['GrLivArea'] ** 2
train_data['GrLivArea-3'] = train_data['GrLivArea'] ** 3
train_data['GrLivArea-Sq'] = np.sqrt(train_data['GrLivArea'])
train_data['SimplOverallQual-s2'] = train_data['SimplOverallQual'] ** 2
train_data['SimplOverallQual-s3'] = train_data['SimplOverallQual'] ** 3
train_data['SimplOverallQual-Sq'] = np.sqrt(train_data['SimplOverallQual'])
train_data['ExterQual-2'] = train_data['ExterQual'] ** 2
train_data['ExterQual-3'] = train_data['ExterQual'] ** 3
train_data['ExterQual-Sq'] = np.sqrt(train_data['ExterQual'])
train_data['GarageCars-2'] = train_data['GarageCars'] ** 2
train_data['GarageCars-3'] = train_data['GarageCars'] ** 3
train_data['GarageCars-Sq'] = np.sqrt(train_data['GarageCars'])
train_data['TotalBath-2'] = train_data['TotalBath'] ** 2
train_data['TotalBath-3'] = train_data['TotalBath'] ** 3
train_data['TotalBath-Sq'] = np.sqrt(train_data['TotalBath'])
train_data['KitchenQual-2'] = train_data['KitchenQual'] ** 2
train_data['KitchenQual-3'] = train_data['KitchenQual'] ** 3
train_data['KitchenQual-Sq'] = np.sqrt(train_data['KitchenQual'])
train_data['GarageScore-2'] = train_data['GarageScore'] ** 2
train_data['GarageScore-3'] = train_data['GarageScore'] ** 3
train_data['GarageScore-Sq'] = np.sqrt(train_data['GarageScore'])
test_data['OverallQual-s2'] = test_data['OverallQual'] ** 2
test_data['OverallQual-s3'] = test_data['OverallQual'] ** 3
test_data['OverallQual-Sq'] = np.sqrt(test_data['OverallQual'])
test_data['AllSF-2'] = test_data['AllSF'] ** 2
test_data['AllSF-3'] = test_data['AllSF'] ** 3
test_data['AllSF-Sq'] = np.sqrt(test_data['AllSF'])
test_data['AllFlrsSF-2'] = test_data['AllFlrsSF'] ** 2
test_data['AllFlrsSF-3'] = test_data['AllFlrsSF'] ** 3
test_data['AllFlrsSF-Sq'] = np.sqrt(test_data['AllFlrsSF'])
test_data['GrLivArea-2'] = test_data['GrLivArea'] ** 2
test_data['GrLivArea-3'] = test_data['GrLivArea'] ** 3
test_data['GrLivArea-Sq'] = np.sqrt(test_data['GrLivArea'])
test_data['SimplOverallQual-s2'] = test_data['SimplOverallQual'] ** 2
test_data['SimplOverallQual-s3'] = test_data['SimplOverallQual'] ** 3
test_data['SimplOverallQual-Sq'] = np.sqrt(test_data['SimplOverallQual'])
test_data['ExterQual-2'] = test_data['ExterQual'] ** 2
test_data['ExterQual-3'] = test_data['ExterQual'] ** 3
test_data['ExterQual-Sq'] = np.sqrt(test_data['ExterQual'])
test_data['GarageCars-2'] = test_data['GarageCars'] ** 2
test_data['GarageCars-3'] = test_data['GarageCars'] ** 3
test_data['GarageCars-Sq'] = np.sqrt(test_data['GarageCars'])
test_data['TotalBath-2'] = test_data['TotalBath'] ** 2
test_data['TotalBath-3'] = test_data['TotalBath'] ** 3
test_data['TotalBath-Sq'] = np.sqrt(test_data['TotalBath'])
test_data['KitchenQual-2'] = test_data['KitchenQual'] ** 2
test_data['KitchenQual-3'] = test_data['KitchenQual'] ** 3
test_data['KitchenQual-Sq'] = np.sqrt(test_data['KitchenQual'])
test_data['GarageScore-2'] = test_data['GarageScore'] ** 2
test_data['GarageScore-3'] = test_data['GarageScore'] ** 3
test_data['GarageScore-Sq'] = np.sqrt(test_data['GarageScore'])


################################################### 特征丢弃 #############################################################

test_data = test_data.drop(['Alley'], axis=1)
train_data = train_data.drop(['Alley'], axis=1)
test_data = test_data.drop(['Utilities'], axis=1)
train_data = train_data.drop(['Utilities'], axis=1)
test_data = test_data.drop(['PoolArea'], axis=1)
train_data = train_data.drop(['PoolArea'], axis=1)



# 因为test中没有"GrLivArea" > 4000的，所以可以删掉，以防过拟合
train_data.drop(train_data[train_data["GrLivArea"] > 4000].index, inplace=True)


################################################### 使用log1p #############################################################

'''
feats = train_data.columns.difference(['Id','SalePrice'])
from scipy.stats import skew, skewtest
all_data = pd.concat((train_data.loc[:,feats], test_data.loc[:,feats]))
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) 
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

print(skewed_feats)

train_data[skewed_feats] = np.log1p(train_data[skewed_feats])
test_data[skewed_feats] = np.log1p(test_data[skewed_feats])
'''


############################################ 对类型特征编码 #############################################################

for cols in train_data.columns:
    if train_data[cols].dtype == np.object:
        train_data = pd.concat((train_data, pd.get_dummies(train_data[cols], prefix=cols)), axis=1)
        del train_data[cols]

for cols in test_data.columns:
    if test_data[cols].dtype == np.object:
        test_data = pd.concat((test_data, pd.get_dummies(test_data[cols], prefix=cols)), axis=1)
        del test_data[cols]         



################################################### 特征对齐 #############################################################

#特征对其时会将train_set中SalePrice,'Id'删去，所以先保留
train_y = np.log1p(train_data['SalePrice'])


# 保证两边不会有不同时存在的特征
col_train = train_data.columns
col_test = test_data.columns
for index in col_train:
    if index in col_test:
        pass
    else:
        del train_data[index]

col_train = train_data.columns
col_test = test_data.columns
for index in col_test:
    if index in col_train:
        pass
    else:
        del test_data[index]    


################################################### 特征重要性排序 #############################################################


""" RF特征重要性选择 """
'''
etr = RandomForestRegressor(n_estimators = 400)
train_x = train_data
etr.fit(train_x, train_y)
imp = etr.feature_importances_
print('###############################')
print(imp)
print(etr.n_features_)
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})
imp = imp.sort_values(['score'], ascending=[0])
print('###############################')
print(imp)
'''
# GBDT特征重要性选择
gbdt = GradientBoostingRegressor(
                random_state=1,
                learning_rate=0.015, 
                min_samples_split=2,
                max_features='sqrt',   # 分裂的feature是随机挑选的
                n_estimators=100,
                min_samples_leaf=1,
                subsample=0.2,
                max_depth=3,
            )
train_x = train_data
gbdt.fit(train_x, train_y)
imp = gbdt.feature_importances_
imp = pd.DataFrame({'feature': train_x.columns, 'score': imp})
imp = imp.sort_values(['score'], ascending=[0])
#bivariate analysis saleprice/grlivarea 
#imp[['feature','score']].plot(kind='bar', stacked=True)   




################################################### 训练和输出结果 #############################################################


################################ 准备数据 ################################
select_feature = imp['feature'][:75]
xtrain_feature = train_x.loc[:,select_feature]
xtest_feature = test_data.loc[:,select_feature]


################################ 分类器一 ################################

# Xgboost
regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=0)

regr.fit(xtrain_feature, train_y)
y_pred = regr.predict(xtrain_feature)
y_test = train_y
print("XGBoost score on training set: ", rmse(y_test, y_pred))
y_pred_xgb = regr.predict(xtest_feature)


################################ 分类器二 ################################

# 2* Ridge
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 24])
ridge.fit(xtrain_feature, train_y)
alpha = ridge.alpha_
print("Best alpha :", alpha)
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(xtrain_feature, train_y)
alpha = ridge.alpha_
print("Best alpha :", alpha)
y_test = train_y
y_pred = ridge.predict(xtrain_feature)
print("Ridge RMSE score on training set: ", rmse(y_test, y_pred))
y_test_rdg = ridge.predict(xtest_feature)


################################ 分类器三 ################################

# Lasso
'''
部分测试结果：
best_alpha = 0.00099,  0.12906
best_alpha = 0.00097,  0.12888
best_alpha = 0.00096,  0.12775

'''
best_alpha = 0.00096
regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(xtrain_feature, train_y)
y_pred = regr.predict(xtrain_feature)
y_test = train_y
print("Lasso score on training set: ", rmse(y_test, y_pred))
y_pred_lasso = regr.predict(xtest_feature)


################################ 分类器四 ################################

# 4* ElasticNet
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(xtrain_feature, train_y)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

y_test = elasticNet.predict(xtrain_feature)
print("ela score on training set: ", rmse(y_test, y_pred))
y_test_ela = elasticNet.predict(xtest_feature)


################################ 糅合结果 ################################

# 根据前面四个分类器的预测得到结果
#  y_pred_xgb， y_test_ela，y_pred_lasso ，y_test_rdg
subm = pd.read_csv("../input/sample_submission.csv")
subm.iloc[:,1] = np.array(np.expm1(y_pred_xgb))
subm.to_csv('../log1p_xgb.csv', index=None)
subm.iloc[:,1] = np.array(np.expm1(y_test_ela))
subm.to_csv('../log1p_ela.csv', index=None)
subm.iloc[:,1] = np.array(np.expm1(y_pred_lasso))
subm.to_csv('../log1p_lasso.csv', index=None)
subm.iloc[:,1] = np.array(np.expm1(y_test_rdg))
subm.to_csv('../log1p_rgd.csv', index=None)

y_pred = (y_pred_xgb + y_test_ela )/2; # 此语句可修改，这里留下的是准确率最高的组合
subm.iloc[:,1] = np.array(np.expm1(y_pred))
subm.to_csv('../log1p_xgb_ela.csv', index=None)
