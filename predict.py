import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

features_with_no_imp_at_least_twice = [
    'ACTIVE_CNT_CREDIT_PROLONG_SUM', 'ACTIVE_CREDIT_DAY_OVERDUE_MEAN', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'BURO_CNT_CREDIT_PROLONG_SUM', 'BURO_CREDIT_ACTIVE_Bad debt_MEAN', 'BURO_CREDIT_ACTIVE_nan_MEAN',
    'BURO_CREDIT_CURRENCY_currency 1_MEAN', 'BURO_CREDIT_CURRENCY_currency 2_MEAN', 'BURO_CREDIT_CURRENCY_currency 3_MEAN',
    'BURO_CREDIT_CURRENCY_currency 4_MEAN', 'BURO_CREDIT_CURRENCY_nan_MEAN', 'BURO_CREDIT_DAY_OVERDUE_MAX', 'BURO_CREDIT_DAY_OVERDUE_MEAN',
    'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN', 'BURO_CREDIT_TYPE_Interbank credit_MEAN', 'BURO_CREDIT_TYPE_Loan for business development_MEAN',
    'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN', 'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
    'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN', 'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
    'BURO_CREDIT_TYPE_Real estate loan_MEAN', 'BURO_CREDIT_TYPE_Unknown type of loan_MEAN', 'BURO_CREDIT_TYPE_nan_MEAN',
    'BURO_MONTHS_BALANCE_MAX_MAX', 'BURO_STATUS_2_MEAN_MEAN', 'BURO_STATUS_3_MEAN_MEAN', 'BURO_STATUS_4_MEAN_MEAN', 'BURO_STATUS_5_MEAN_MEAN',
    'BURO_STATUS_nan_MEAN_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN', 'CC_AMT_DRAWINGS_CURRENT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_AMT_INST_MIN_REGULARITY_MIN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN', 'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR',
    'CC_AMT_RECIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_MAX', 'CC_AMT_TOTAL_RECEIVABLE_MIN', 'CC_AMT_TOTAL_RECEIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_VAR',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MIN', 'CC_CNT_DRAWINGS_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM', 'CC_CNT_DRAWINGS_OTHER_CURRENT_VAR', 'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
    'CC_CNT_INSTALMENT_MATURE_CUM_MAX', 'CC_CNT_INSTALMENT_MATURE_CUM_MIN', 'CC_COUNT', 'CC_MONTHS_BALANCE_MAX', 'CC_MONTHS_BALANCE_MEAN',
    'CC_MONTHS_BALANCE_MIN', 'CC_MONTHS_BALANCE_SUM', 'CC_NAME_CONTRACT_STATUS_Active_MAX', 'CC_NAME_CONTRACT_STATUS_Active_MIN',
    'CC_NAME_CONTRACT_STATUS_Approved_MAX', 'CC_NAME_CONTRACT_STATUS_Approved_MEAN', 'CC_NAME_CONTRACT_STATUS_Approved_MIN',
    'CC_NAME_CONTRACT_STATUS_Approved_SUM', 'CC_NAME_CONTRACT_STATUS_Approved_VAR', 'CC_NAME_CONTRACT_STATUS_Completed_MAX',
    'CC_NAME_CONTRACT_STATUS_Completed_MEAN', 'CC_NAME_CONTRACT_STATUS_Completed_MIN', 'CC_NAME_CONTRACT_STATUS_Completed_SUM', 'CC_NAME_CONTRACT_STATUS_Completed_VAR',
    'CC_NAME_CONTRACT_STATUS_Demand_MAX', 'CC_NAME_CONTRACT_STATUS_Demand_MEAN', 'CC_NAME_CONTRACT_STATUS_Demand_MIN', 'CC_NAME_CONTRACT_STATUS_Demand_SUM',
    'CC_NAME_CONTRACT_STATUS_Demand_VAR', 'CC_NAME_CONTRACT_STATUS_Refused_MAX', 'CC_NAME_CONTRACT_STATUS_Refused_MEAN', 'CC_NAME_CONTRACT_STATUS_Refused_MIN',
    'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'CC_NAME_CONTRACT_STATUS_Refused_VAR', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR', 'CC_NAME_CONTRACT_STATUS_Signed_MAX', 'CC_NAME_CONTRACT_STATUS_Signed_MEAN', 'CC_NAME_CONTRACT_STATUS_Signed_MIN',
    'CC_NAME_CONTRACT_STATUS_Signed_SUM', 'CC_NAME_CONTRACT_STATUS_Signed_VAR', 'CC_NAME_CONTRACT_STATUS_nan_MAX', 'CC_NAME_CONTRACT_STATUS_nan_MEAN',
    'CC_NAME_CONTRACT_STATUS_nan_MIN', 'CC_NAME_CONTRACT_STATUS_nan_SUM', 'CC_NAME_CONTRACT_STATUS_nan_VAR', 'CC_SK_DPD_DEF_MAX',
    'CC_SK_DPD_DEF_MIN', 'CC_SK_DPD_DEF_SUM', 'CC_SK_DPD_DEF_VAR', 'CC_SK_DPD_MAX', 'CC_SK_DPD_MEAN', 'CC_SK_DPD_MIN', 'CC_SK_DPD_SUM',
    'CC_SK_DPD_VAR', 'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN', 'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM', 'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
    'CLOSED_CNT_CREDIT_PROLONG_SUM', 'CLOSED_CREDIT_DAY_OVERDUE_MAX', 'CLOSED_CREDIT_DAY_OVERDUE_MEAN', 'CLOSED_MONTHS_BALANCE_MAX_MAX',
    'CNT_CHILDREN', 'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'EMERGENCYSTATE_MODE_No', 'EMERGENCYSTATE_MODE_Yes', 'ENTRANCES_MODE', 'FLAG_CONT_MOBILE',
    'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
    'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLOORSMAX_MODE',
    'FONDKAPREMONT_MODE_not specified', 'FONDKAPREMONT_MODE_org spec account', 'FONDKAPREMONT_MODE_reg oper account', 'FONDKAPREMONT_MODE_reg oper spec account',
    'HOUSETYPE_MODE_block of flats', 'HOUSETYPE_MODE_specific housing', 'HOUSETYPE_MODE_terraced house', 'LIVE_REGION_NOT_WORK_REGION',
    'NAME_CONTRACT_TYPE_Revolving loans', 'NAME_EDUCATION_TYPE_Academic degree', 'NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Single / not married',
    'NAME_FAMILY_STATUS_Unknown', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment', 'NAME_HOUSING_TYPE_With parents',
    'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_Student',
    'NAME_INCOME_TYPE_Unemployed', 'NAME_TYPE_SUITE_Children', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
    'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied',
    'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_SUM', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',
    'NEW_RATIO_BURO_CNT_CREDIT_PROLONG_SUM', 'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MAX', 'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MEAN', 'NEW_RATIO_BURO_MONTHS_BALANCE_MAX_MAX',
    'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MIN', 'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MAX', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff',
    'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
    'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries',
    'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Waiters/barmen staff', 'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Agriculture',
    'ORGANIZATION_TYPE_Business Entity Type 1', 'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Culture',
    'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government', 'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing',
    'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Industry: type 12',
    'ORGANIZATION_TYPE_Industry: type 13', 'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 3', 'ORGANIZATION_TYPE_Industry: type 4',
    'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Industry: type 8',
    'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Postal',
    'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant', 'ORGANIZATION_TYPE_Security',
    'ORGANIZATION_TYPE_Security Ministries', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom', 'ORGANIZATION_TYPE_Trade: type 1',
    'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Trade: type 3', 'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5',
    'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_Trade: type 7',
    'ORGANIZATION_TYPE_Transport: type 1', 'ORGANIZATION_TYPE_Transport: type 2', 'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_University',
    'ORGANIZATION_TYPE_XNA', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN', 'POS_NAME_CONTRACT_STATUS_Approved_MEAN', 'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
    'POS_NAME_CONTRACT_STATUS_Demand_MEAN', 'POS_NAME_CONTRACT_STATUS_XNA_MEAN', 'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'PREV_CHANNEL_TYPE_Car dealer_MEAN',
    'PREV_CHANNEL_TYPE_nan_MEAN', 'PREV_CODE_REJECT_REASON_CLIENT_MEAN', 'PREV_CODE_REJECT_REASON_SYSTEM_MEAN', 'PREV_CODE_REJECT_REASON_VERIF_MEAN',
    'PREV_CODE_REJECT_REASON_XNA_MEAN', 'PREV_CODE_REJECT_REASON_nan_MEAN', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN', 'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
    'PREV_NAME_CLIENT_TYPE_nan_MEAN', 'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN', 'PREV_NAME_CONTRACT_STATUS_nan_MEAN', 'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
    'PREV_NAME_CONTRACT_TYPE_nan_MEAN', 'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN', 'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN', 'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN', 'PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN', 'PREV_NAME_GOODS_CATEGORY_Education_MEAN', 'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Gardening_MEAN', 'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN', 'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN', 'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN', 'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN', 'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN', 'PREV_NAME_GOODS_CATEGORY_Other_MEAN', 'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN', 'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN', 'PREV_NAME_GOODS_CATEGORY_XNA_MEAN', 'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
    'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN', 'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN', 'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
    'PREV_NAME_PORTFOLIO_Cars_MEAN', 'PREV_NAME_PORTFOLIO_nan_MEAN', 'PREV_NAME_PRODUCT_TYPE_nan_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Construction_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Industry_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN', 'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN', 'PREV_NAME_SELLER_INDUSTRY_nan_MEAN', 'PREV_NAME_TYPE_SUITE_Group of people_MEAN', 'PREV_NAME_YIELD_GROUP_nan_MEAN',
    'PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN', 'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN', 'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN',
    'PREV_PRODUCT_COMBINATION_nan_MEAN', 'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MAX', 'REFUSED_AMT_DOWN_PAYMENT_MEAN',
    'REFUSED_RATE_DOWN_PAYMENT_MIN', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
    'WALLSMATERIAL_MODE_Block', 'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Monolithic', 'WALLSMATERIAL_MODE_Others', 'WALLSMATERIAL_MODE_Panel',
    'WALLSMATERIAL_MODE_Wooden', 'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY'
#    # 500~ (really 162.595, size of 100K) relative to size of total actual training size
    , 'BURO_CREDIT_TYPE_Another type of loan_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN', 'NEW_EXT_SOURCES_KURTOSIS',
    'CC_CNT_DRAWINGS_POS_CURRENT_MIN'
#    ,  'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX', 'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN',     'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MEAN',
#       'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MIN',     'PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN',      'CLOSED_AMT_CREDIT_SUM_DEBT_MEAN',
#       'NEW_RATIO_PREV_DAYS_DECISION_MIN',      'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM',      'NEW_RATIO_BURO_AMT_ANNUITY_MEAN',
#       'NEW_RATIO_BURO_AMT_ANNUITY_MAX', 'OCCUPATION_TYPE_Medicine staff',      'PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN',      'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Transport: type 3',
#       'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX',      'NEW_RATIO_PREV_DAYS_DECISION_MEAN',     'BURO_CREDIT_TYPE_Mortgage_MEAN',     'BURO_CREDIT_TYPE_Microloan_MEAN',
#       'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',      'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_SUM',      'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN',
#       'BURO_CREDIT_ACTIVE_Sold_MEAN', 'NEW_RATIO_PREV_AMT_CREDIT_MEAN',
#       'NEW_RATIO_PREV_AMT_CREDIT_MIN', 'CLOSED_AMT_CREDIT_SUM_DEBT_SUM',
#       'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MAX',
#       'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MEAN',
#       'BURO_CREDIT_TYPE_Another type of loan_MEAN',
#       'NEW_RATIO_BURO_MONTHS_BALANCE_MIN_MIN',
#       'NEW_RATIO_PREV_APP_CREDIT_PERC_MAX',
#       'NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN',
#       'NEW_RATIO_PREV_APP_CREDIT_PERC_VAR',
#       'CLOSED_MONTHS_BALANCE_MIN_MIN',
#       'PREV_NAME_GOODS_CATEGORY_Sport and Leisure_MEAN',
#       'POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN',
#       'PREV_NAME_TYPE_SUITE_Children_MEAN',
#       'BURO_MONTHS_BALANCE_MIN_MIN',
#       'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
#       'FLAG_DOCUMENT_18', 'REFUSED_APP_CREDIT_PERC_VAR',
#       'PREV_PRODUCT_COMBINATION_Cash Street: low_MEAN',
#       'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
#       'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN',
#       'ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN',
#       'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN',
#       'ACTIVE_CREDIT_DAY_OVERDUE_MAX',
#       'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
#       'CLOSED_MONTHS_BALANCE_SIZE_SUM',
#       'CLOSED_MONTHS_BALANCE_SIZE_MEAN',
#       'PREV_NAME_TYPE_SUITE_Other_A_MEAN',
#       'PREV_CODE_REJECT_REASON_SCOFR_MEAN', 'BURO_STATUS_X_MEAN_MEAN',
#       'CC_CNT_DRAWINGS_POS_CURRENT_MIN', 'ACTIVE_MONTHS_BALANCE_MAX_MAX',
#       'POS_SK_DPD_DEF_MAX', 'BURO_MONTHS_BALANCE_SIZE_SUM',
#       'BURO_STATUS_0_MEAN_MEAN', 'PREV_CHANNEL_TYPE_Contact center_MEAN',
#       'PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN',
#       'POS_SK_DPD_DEF_MEAN', 'BURO_STATUS_1_MEAN_MEAN',
#       'PREV_CHANNEL_TYPE_Channel of corporate sales_MEAN',
#       'ACTIVE_MONTHS_BALANCE_SIZE_SUM',
#       'ACTIVE_MONTHS_BALANCE_SIZE_MEAN', 'ACTIVE_MONTHS_BALANCE_MIN_MIN',
#       'BURO_MONTHS_BALANCE_SIZE_MEAN', 'BURO_STATUS_C_MEAN_MEAN',
#       'CLOSED_AMT_ANNUITY_MEAN',
#       'PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN',
#       'NEW_RATIO_PREV_APP_CREDIT_PERC_MIN',
#       'NEW_RATIO_PREV_AMT_GOODS_PRICE_MAX',
#       'NAME_HOUSING_TYPE_Rented apartment',
#       'NEW_RATIO_PREV_AMT_ANNUITY_MEAN',    'NEW_RATIO_PREV_AMT_ANNUITY_MAX', 'BURO_AMT_ANNUITY_MEAN',       'PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN',
#       'ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM',      'NEW_RATIO_PREV_AMT_GOODS_PRICE_MEAN',
#       'NEW_RATIO_PREV_AMT_CREDIT_MAX', 'BURO_CREDIT_TYPE_Car loan_MEAN',
#       'NEW_RATIO_PREV_AMT_ANNUITY_MIN', 'CLOSED_AMT_ANNUITY_MAX',
#       'PREV_NAME_TYPE_SUITE_Other_B_MEAN',
#       'CLOSED_AMT_CREDIT_SUM_DEBT_MAX',      'POS_NAME_CONTRACT_STATUS_Signed_MEAN',
#       'NEW_RATIO_PREV_AMT_GOODS_PRICE_MIN', 'ACTIVE_AMT_ANNUITY_MEAN',      'NEW_RATIO_PREV_CNT_PAYMENT_MEAN',
#       'NEW_RATIO_PREV_AMT_APPLICATION_MAX', 'POS_COUNT',
#       'PREV_NAME_GOODS_CATEGORY_Furniture_MEAN',
#       'PREV_NAME_YIELD_GROUP_low_action_MEAN',
#       'CC_CNT_DRAWINGS_ATM_CURRENT_SUM',
#       'NEW_RATIO_PREV_AMT_APPLICATION_MEAN',
#       'CC_CNT_DRAWINGS_POS_CURRENT_VAR',
#       'NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
#       'NEW_RATIO_PREV_AMT_APPLICATION_MIN',
#       'PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN', 'POS_SK_DPD_MEAN',       'PREV_CODE_REJECT_REASON_LIMIT_MEAN',     'PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN',
#       'PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN',
#       'OCCUPATION_TYPE_Accountants', 'REFUSED_RATE_DOWN_PAYMENT_MAX',
#       'CC_CNT_DRAWINGS_POS_CURRENT_MAX', 'REFUSED_AMT_GOODS_PRICE_MAX',
#       'REFUSED_RATE_DOWN_PAYMENT_MEAN',
#       'ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN',
#       'CC_CNT_DRAWINGS_POS_CURRENT_MEAN',
#       'NEW_RATIO_PREV_CNT_PAYMENT_SUM', 'CC_SK_DPD_DEF_MEAN',
#       'CC_CNT_DRAWINGS_CURRENT_MAX',
#       'CC_NAME_CONTRACT_STATUS_Active_VAR', 'BURO_AMT_ANNUITY_MAX',      'CC_CNT_DRAWINGS_ATM_CURRENT_MAX',
#       'NEW_RATIO_PREV_DAYS_DECISION_MAX',
#       'PREV_NAME_TYPE_SUITE_Spouse, partner_MEAN',      'NAME_HOUSING_TYPE_Municipal apartment',
#       'PREV_NAME_PRODUCT_TYPE_XNA_MEAN',
#       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
#       'WEEKDAY_APPR_PROCESS_START_SUNDAY',
#       'CC_AMT_DRAWINGS_POS_CURRENT_MEAN', 'ORGANIZATION_TYPE_Medicine',
#       'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN',
#       'CC_AMT_DRAWINGS_POS_CURRENT_VAR',       'BURO_CREDIT_ACTIVE_Active_MEAN',      'PREV_NAME_GOODS_CATEGORY_Computers_MEAN',       'PREV_NAME_YIELD_GROUP_XNA_MEAN', 'BURO_AMT_CREDIT_SUM_LIMIT_SUM',      'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
#       'CC_AMT_DRAWINGS_POS_CURRENT_MIN', 'ACTIVE_AMT_ANNUITY_MAX',       'PREV_CHANNEL_TYPE_Regional / Local_MEAN',
#       'PREV_NAME_PORTFOLIO_Cards_MEAN',
#       'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN',       'PREV_NAME_PORTFOLIO_XNA_MEAN',       'NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN',
#       'PREV_PRODUCT_COMBINATION_Card Street_MEAN',       'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN',
#       'REFUSED_APP_CREDIT_PERC_MEAN', 'REFUSED_AMT_APPLICATION_MIN',
#       'CC_CNT_DRAWINGS_ATM_CURRENT_VAR',
#       'CC_AMT_DRAWINGS_POS_CURRENT_MAX',
#       'PREV_NAME_PRODUCT_TYPE_x-sell_MEAN',       'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN',       'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',       'CC_AMT_DRAWINGS_CURRENT_MAX',
#       'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN',
#       'REFUSED_HOUR_APPR_PROCESS_START_MAX',
#       'PREV_CODE_REJECT_REASON_XAP_MEAN',
#       'WEEKDAY_APPR_PROCESS_START_MONDAY',      'CC_NAME_CONTRACT_STATUS_Active_SUM',
#       'PREV_NAME_CLIENT_TYPE_Refreshed_MEAN',       'REFUSED_AMT_DOWN_PAYMENT_MIN', 'PREV_NAME_PORTFOLIO_Cash_MEAN',       'PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN',
#       'CC_AMT_DRAWINGS_CURRENT_SUM',
#       'PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN',       'INSTAL_PAYMENT_PERC_MAX', 'REFUSED_AMT_GOODS_PRICE_MIN',
#       'CLOSED_DAYS_CREDIT_ENDDATE_MIN',
#       'PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN',
#       'REFUSED_AMT_ANNUITY_MAX', 'INSTAL_COUNT',
#       'PREV_CHANNEL_TYPE_Stone_MEAN', 'REFUSED_APP_CREDIT_PERC_MIN',
#       'PREV_CODE_REJECT_REASON_HC_MEAN',
#       'POS_NAME_CONTRACT_STATUS_Active_MEAN',
#       'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN',
#       'ORGANIZATION_TYPE_Industry: type 9',
#       'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN', 'POS_SK_DPD_MAX',
#       'BURO_CREDIT_ACTIVE_Closed_MEAN',
#       'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN',
#       'REFUSED_AMT_GOODS_PRICE_MEAN', 'CC_AMT_TOTAL_RECEIVABLE_MEAN',
#       'CC_AMT_BALANCE_VAR', 'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM',
#       'BURO_AMT_CREDIT_SUM_LIMIT_MEAN', 'CC_AMT_DRAWINGS_CURRENT_VAR',
#       'PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN',
#       'APPROVED_AMT_GOODS_PRICE_MAX', 'PREV_NAME_PORTFOLIO_POS_MEAN',
#       'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
#       'PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN',
#       'CC_NAME_CONTRACT_STATUS_Active_MEAN',
#       'REFUSED_AMT_APPLICATION_MAX', 'CC_AMT_RECEIVABLE_PRINCIPAL_VAR',
#       'PREV_CHANNEL_TYPE_Credit and cash offices_MEAN',
#       'PREV_CHANNEL_TYPE_Country-wide_MEAN',
#       'ACTIVE_AMT_CREDIT_SUM_DEBT_MAX',
#       'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN', 'INSTAL_DPD_SUM',
#       'PREV_RATE_DOWN_PAYMENT_MAX', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_SUM',
#       'CLOSED_AMT_CREDIT_SUM_MAX', 'ORGANIZATION_TYPE_Bank',
#       'CLOSED_AMT_CREDIT_SUM_SUM', 'APPROVED_RATE_DOWN_PAYMENT_MEAN',
#       'APPROVED_RATE_DOWN_PAYMENT_MIN', 'NAME_CONTRACT_TYPE_Cash loans',
#       'BURO_AMT_CREDIT_SUM_DEBT_MAX', 'OCCUPATION_TYPE_Drivers',
#       'REFUSED_CNT_PAYMENT_MEAN', 'CC_AMT_BALANCE_MIN',
#       'ORGANIZATION_TYPE_Construction',
#       'PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN',
#       'NAME_EDUCATION_TYPE_Incomplete higher',
#       'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX', 'BURO_DAYS_CREDIT_ENDDATE_MIN',
#       'PREV_NAME_YIELD_GROUP_middle_MEAN',
#       'ACTIVE_DAYS_CREDIT_ENDDATE_MAX',
#       'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',
#       'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
#       'ORGANIZATION_TYPE_Kindergarten', 'CLOSED_DAYS_CREDIT_MIN',
#       'POS_NAME_CONTRACT_STATUS_Completed_MEAN',
#       'PREV_NAME_YIELD_GROUP_high_MEAN',
#       'BURO_CREDIT_TYPE_Consumer credit_MEAN',
#       'CC_AMT_DRAWINGS_ATM_CURRENT_SUM', 'REFUSED_AMT_ANNUITY_MEAN',
#       'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN',
#       'CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN', 'REFUSED_CNT_PAYMENT_SUM',
#       'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MEAN', 'CC_AMT_BALANCE_MEAN',
#       'NEW_LIVE_IND_KURT', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_MEAN',
#       'REFUSED_HOUR_APPR_PROCESS_START_MEAN',
#       'CC_CNT_DRAWINGS_CURRENT_VAR', 'BURO_DAYS_CREDIT_MIN',
#       'ACTIVE_AMT_CREDIT_SUM_MEAN', 'PREV_NAME_TYPE_SUITE_Family_MEAN',
#       'ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'NEW_RATIO_BURO_DAYS_CREDIT_MIN',
#       'NAME_HOUSING_TYPE_Office apartment',
#       'NEW_RATIO_BURO_AMT_CREDIT_SUM_MAX',
#       'CC_AMT_DRAWINGS_CURRENT_MEAN', 'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN',
#       'CC_AMT_DRAWINGS_ATM_CURRENT_VAR', 'REG_CITY_NOT_LIVE_CITY',
#       'BURO_CREDIT_TYPE_Credit card_MEAN', 'CC_AMT_RECIVABLE_MEAN',
#       'NAME_HOUSING_TYPE_House / apartment',
#       'PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN',
#       'APPROVED_AMT_DOWN_PAYMENT_MEAN',
#       'NAME_INCOME_TYPE_Commercial associate',
#       'APPROVED_APP_CREDIT_PERC_VAR', 'CC_AMT_RECIVABLE_VAR',
#       'FLAG_DOCUMENT_8', 'APPROVED_AMT_APPLICATION_MAX',
#       'APPROVED_RATE_DOWN_PAYMENT_MAX', 'LIVE_CITY_NOT_WORK_CITY',
#       'APPROVED_AMT_GOODS_PRICE_MEAN',
#       'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN',
#       'CC_AMT_DRAWINGS_ATM_CURRENT_MAX', 'ACTIVE_DAYS_CREDIT_MAX',
#       'CLOSED_DAYS_CREDIT_ENDDATE_MEAN', 'APPROVED_AMT_GOODS_PRICE_MIN',
#       'APPROVED_AMT_APPLICATION_MEAN',
#       'OCCUPATION_TYPE_High skill tech staff',
#       'WEEKDAY_APPR_PROCESS_START_SATURDAY',
#       'CC_AMT_RECEIVABLE_PRINCIPAL_MIN',
#       'APPROVED_HOUR_APPR_PROCESS_START_MAX', 'CC_AMT_RECIVABLE_MIN',
#       'REFUSED_AMT_CREDIT_MAX',
#       'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',
#       'ACTIVE_DAYS_CREDIT_MEAN',
#       'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
#       'PREV_NAME_PAYMENT_TYPE_XNA_MEAN',
#       'REFUSED_HOUR_APPR_PROCESS_START_MIN',
#       'PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN', 'POS_MONTHS_BALANCE_SIZE',
#       'APPROVED_AMT_CREDIT_MEAN', 'BURO_AMT_CREDIT_SUM_MAX',
#       'NEW_LIVE_IND_SUM', 'PREV_AMT_DOWN_PAYMENT_MIN',
#       'NEW_RATIO_BURO_DAYS_CREDIT_VAR', 'CC_CNT_DRAWINGS_CURRENT_SUM',
#       'REFUSED_AMT_CREDIT_MIN', 'BURO_DAYS_CREDIT_VAR',
#       'PREV_RATE_DOWN_PAYMENT_MIN', 'ORGANIZATION_TYPE_School',
#       'PREV_AMT_GOODS_PRICE_MAX', 'CC_AMT_PAYMENT_CURRENT_VAR',
#       'ACTIVE_AMT_CREDIT_SUM_MAX', 'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN',
#       'REFUSED_AMT_APPLICATION_MEAN',
#       'PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN',
#       'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
#       'PREV_CODE_REJECT_REASON_SCO_MEAN',
#       'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',
#       'PREV_PRODUCT_COMBINATION_Cash_MEAN',
#       'CLOSED_DAYS_CREDIT_UPDATE_MEAN', 'BURO_AMT_CREDIT_SUM_DEBT_MEAN',
#       'NEW_DOC_IND_STD', 'CC_CNT_DRAWINGS_CURRENT_MEAN',
#       'CC_AMT_RECIVABLE_MAX', 'BURO_DAYS_CREDIT_MEAN',
#       'CC_AMT_CREDIT_LIMIT_ACTUAL_VAR',
#       'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN',
#       'PREV_HOUR_APPR_PROCESS_START_MEAN', 'CC_AMT_BALANCE_SUM',
#       'CLOSED_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_ENDDATE_MAX',
#       'NEW_RATIO_BURO_DAYS_CREDIT_MAX', 'PREV_AMT_DOWN_PAYMENT_MAX',
#       'REFUSED_APP_CREDIT_PERC_MAX', 'NONLIVINGAPARTMENTS_MODE',
#       'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
#       'NAME_EDUCATION_TYPE_Lower secondary',
#       'PREV_AMT_DOWN_PAYMENT_MEAN', 'APPROVED_APP_CREDIT_PERC_MEAN',
#       'BURO_DAYS_CREDIT_ENDDATE_MEAN', 'CLOSED_AMT_CREDIT_SUM_MEAN',
#       'BURO_AMT_CREDIT_SUM_DEBT_SUM',
#       'APPROVED_HOUR_APPR_PROCESS_START_MIN',
#       'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX', 'FLAG_OWN_REALTY',
#       'PREV_RATE_DOWN_PAYMENT_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
#       'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'APPROVED_AMT_CREDIT_MIN',
#       'NAME_FAMILY_STATUS_Separated',
#       'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
#       'NEW_RATIO_BURO_DAYS_CREDIT_MEAN', 'APPROVED_AMT_ANNUITY_MEAN',
#       'BURO_AMT_CREDIT_SUM_MEAN', 'REFUSED_AMT_ANNUITY_MIN',
#       'NEW_LIVE_IND_STD', 'FLOORSMAX_MEDI',
#       'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN', 'APPROVED_DAYS_DECISION_MEAN',
#       'PREV_AMT_APPLICATION_MAX', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
#       'APPROVED_HOUR_APPR_PROCESS_START_MEAN', 'INSTAL_PAYMENT_DIFF_VAR',
#       'NONLIVINGAPARTMENTS_MEDI', 'CC_AMT_INST_MIN_REGULARITY_SUM',
#       'NEW_RATIO_BURO_DAYS_CREDIT_UPDATE_MEAN',
#       'CC_AMT_PAYMENT_CURRENT_SUM', 'APPROVED_AMT_CREDIT_MAX',
#       'CC_AMT_BALANCE_MAX', 'FLOORSMIN_MEDI',
#       'CC_AMT_RECEIVABLE_PRINCIPAL_SUM', 'INSTAL_DPD_MAX',
#       'PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN',
#       'CC_AMT_PAYMENT_TOTAL_CURRENT_SUM',
#       'ORGANIZATION_TYPE_Business Entity Type 3',
#       'APPROVED_AMT_APPLICATION_MIN', 'PREV_AMT_GOODS_PRICE_MEAN',
#       'BURO_AMT_CREDIT_SUM_SUM', 'REFUSED_AMT_CREDIT_MEAN',       'CC_AMT_RECEIVABLE_PRINCIPAL_MAX', 'APPROVED_APP_CREDIT_PERC_MAX',       'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
#       'PREV_NAME_TYPE_SUITE_nan_MEAN',
#       'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',
#       'CC_AMT_INST_MIN_REGULARITY_MEAN',
#       'CC_AMT_INST_MIN_REGULARITY_MAX',
#       'PREV_HOUR_APPR_PROCESS_START_MIN',
#       'APPROVED_AMT_DOWN_PAYMENT_MIN', 'PREV_NAME_CLIENT_TYPE_New_MEAN',
#       'ACTIVE_AMT_CREDIT_SUM_SUM', 'CC_AMT_DRAWINGS_ATM_CURRENT_MEAN',
#       'CC_CNT_INSTALMENT_MATURE_CUM_VAR', 'ACTIVE_DAYS_CREDIT_MIN',       'PREV_AMT_APPLICATION_MIN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',       'INSTAL_PAYMENT_PERC_VAR', 'PREV_AMT_ANNUITY_MEAN',
#       'PREV_APP_CREDIT_PERC_MEAN', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX',       'WALLSMATERIAL_MODE_Stone, brick', 'PREV_APP_CREDIT_PERC_MAX',
#       'PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN', 'FLOORSMIN_MODE',       'CLOSED_DAYS_CREDIT_MAX', 'CC_AMT_INST_MIN_REGULARITY_VAR',       'PREV_AMT_APPLICATION_MEAN', 'CLOSED_DAYS_CREDIT_VAR',
#       'INSTAL_PAYMENT_DIFF_MAX', 'PREV_APP_CREDIT_PERC_VAR',       'CC_AMT_CREDIT_LIMIT_ACTUAL_SUM',       'PREV_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN',
#       'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX', 'FLOORSMIN_AVG',       'REFUSED_DAYS_DECISION_MEAN', 'ORGANIZATION_TYPE_Military'       # trained lowest 50 removed      , 'LIVINGAPARTMENTS_AVG',
#       'CNT_FAM_MEMBERS', 'CC_AMT_PAYMENT_CURRENT_MAX',       'YEARS_BUILD_MEDI', 'LIVINGAPARTMENTS_MEDI',
#       'CC_AMT_PAYMENT_CURRENT_MIN', 'OCCUPATION_TYPE_Core staff',       'ENTRANCES_MEDI', 'CC_MONTHS_BALANCE_VAR',
#       'CC_CNT_INSTALMENT_MATURE_CUM_SUM', 'CC_AMT_PAYMENT_CURRENT_MEAN',       'NEW_DOC_IND_AVG', 'PREV_HOUR_APPR_PROCESS_START_MAX',
#       'APPROVED_CNT_PAYMENT_MEAN', 'PREV_AMT_ANNUITY_MAX',       'LIVINGAPARTMENTS_MODE', 'APPROVED_AMT_ANNUITY_MAX',
#       'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_PAYMENT_PERC_SUM',
#       'PREV_AMT_CREDIT_MIN', 'COMMONAREA_AVG', 'COMMONAREA_MEDI',
#       'APPROVED_DAYS_DECISION_MAX', 'PREV_CNT_PAYMENT_SUM',
#       'YEARS_BUILD_AVG', 'NAME_INCOME_TYPE_State servant',
#       'YEARS_BUILD_MODE', 'POS_MONTHS_BALANCE_MAX',
#       'APPROVED_AMT_ANNUITY_MIN', 'PREV_CNT_PAYMENT_MEAN',
#       'REFUSED_DAYS_DECISION_MIN', 'APPROVED_DAYS_DECISION_MIN',
#       'ACTIVE_DAYS_CREDIT_VAR', 'PREV_AMT_ANNUITY_MIN',
#       'INSTAL_PAYMENT_PERC_MEAN', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
#       'NONLIVINGAREA_MEDI', 'BURO_DAYS_CREDIT_MAX',
#       'PREV_AMT_CREDIT_MEAN', 'INSTAL_PAYMENT_DIFF_SUM',
#       'NONLIVINGAPARTMENTS_AVG', 'BASEMENTAREA_AVG',       'PREV_AMT_GOODS_PRICE_MIN', 'APPROVED_CNT_PAYMENT_SUM',       'OCCUPATION_TYPE_Laborers', 'LANDAREA_AVG', 'LIVINGAREA_MEDI',       'PREV_APP_CREDIT_PERC_MIN'
]

# https://www.kaggle.com/ogrellier/lighgbm-with-selected-features/code

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    
    # new ext sources
    df["PAVLE_UN_NORM_1"] = df['EXT_SOURCE_1']*(900-300)+300
    df["PAVLE_UN_NORM_2"] = df['EXT_SOURCE_2']*(900-300)+300
    df["PAVLE_UN_NORM_3"] = df['EXT_SOURCE_3']*(900-300)+300
    df["PAVLE_UN_NORM_INV"] = 1/(2.4166*df["PAVLE_UN_NORM_1"]+2.166*df["PAVLE_UN_NORM_1"]+3.333*df["PAVLE_UN_NORM_1"])**3
    df["PAVLE_INV"] = 1/(df['EXT_SOURCE_1']+ df['EXT_SOURCE_2']+ df['EXT_SOURCE_3'])
    

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    
    # crafted ratios
    df["ANNUITY_EXT_3"] = df["AMT_ANNUITY"]*df["EXT_SOURCE_3"]
    df["ANNUITY_EXT_2"] = df["AMT_ANNUITY"]*df["EXT_SOURCE_2"]
    df["ANNUITY_EXT_2"] = df["AMT_ANNUITY"]*df["EXT_SOURCE_1"]
    df["ANNUITY_TOTAL_FAM_CHILD"]= df["AMT_ANNUITY"] / (1 + df['CNT_CHILDREN']+ df["CNT_FAM_MEMBERS"])
    df["ANNUITY_CHILD_RATIO"] = df["AMT_ANNUITY"] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_FAM_MEMBERS'])
    df["ANNUITY_FAMILY_RATIO"] = df["AMT_ANNUITY"] / (1 + df['CNT_FAM_MEMBERS'])
    df["ANNUITY_FAM_CHILD_RATIO"] = df["AMT_ANNUITY"] / (1 + df['CNT_FAM_MEMBERS']+ df['CNT_CHILDREN'])
    df["GOODS_ANNUITY_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_ANNUITY"]
    df["INCOME_REGION_RATING_CLIENT"] = df["AMT_INCOME_TOTAL"]*df["REGION_RATING_CLIENT"]
    df["INCOME_REGION_RATING_CLIENT_CITY"] = df["AMT_INCOME_TOTAL"]*df["REGION_RATING_CLIENT_W_CITY"]
    df["INCOME_EMPLOYED"] = df["AMT_INCOME_TOTAL"]*df["DAYS_EMPLOYED"]
    df["INCOME_BIRTH"] = df["AMT_INCOME_TOTAL"]*df["DAYS_BIRTH"]
    df["GOODS_TO_INCOME_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
    df['NEW_EXT_SOURCES_KURTOSIS'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].kurtosis(axis=1)


    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    del test_df
    gc.collect()
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('bureau.csv', nrows = num_rows)
    bb = pd.read_csv('bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]
    
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print("Fold Number:", n_fold+1)
        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                             label=train_df['TARGET'].iloc[train_idx], 
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                             label=train_df['TARGET'].iloc[valid_idx], 
                             free_raw_data=False, silent=True)

        # LightGBM parameters found by Bayesian optimization
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.02,  # 02,
            'num_leaves': 60,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'subsample_freq': 1,
            'max_depth': 10,
            'reg_alpha': 0.05,
            'reg_lambda': 0.08,
            'min_split_gain': 0.03,
            'min_child_weight': 60, # 39.3259775,
            'seed': 0,
            'verbose': 1,
            'metric': 'auc',
        }
        
        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=250,
            verbose_eval=True
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
        del clf, dtrain, dvalid
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df['TARGET'] = sub_preds
        sub_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('lgbm_importances01.png')


def main(debug = False):
    num_rows = 100000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        print(df.shape)
        if debug: df.drop(list(set(features_with_no_imp_at_least_twice).intersection(df.columns.values)), axis=1, inplace=True)
        else: df.drop(features_with_no_imp_at_least_twice, axis=1, inplace=True)
        gc.collect()
        print(df.shape)
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)
        return feat_importance

if __name__ == "__main__":
    submission_file_name = "submission_with selected_features.csv"
    with timer("Full model run"):
        main()