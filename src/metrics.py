# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

recommended_list = [143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43] #id товаров
bought_list = [521, 32, 143, 27, 991]

def hit_rate(recommended_list, bought_list):

  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)
  
  flags = np.isin(bought_list, recommended_list)
  
  hit_rate = flags.sum()
  
  return hit_rate

hit_rate(recommended_list, bought_list)

def hit_rate_at_k(recommended_list, bought_list, k):
  
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list[:k])

  flags = np.isin(bought_list, recommended_list)
  
  hit_rate_at_k = flags.sum()
  
  return hit_rate_at_k

hit_rate_at_k(recommended_list, bought_list, k=4)

"""2. **Presicion**"""

def precision(recommended_list, bought_list):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)
  
  flags = np.isin(bought_list, recommended_list)
  
  precision = flags.sum() / len(recommended_list)
  
  return precision

precision(recommended_list, bought_list)

def precision_at_k(recommended_list, bought_list, k):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)
  
  recommended_list = recommended_list[:k]
  
  flags = np.isin(bought_list, recommended_list)
  
  precision = flags.sum() / len(recommended_list)
  
  
  return precision

precision_at_k(recommended_list, bought_list, k=5)

round(precision_at_k(recommended_list, bought_list, k=3), 1)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k):
        
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list[:k])
  prices_recommended = np.array(prices_recommended)
  
  flags = np.isin(bought_list, recommended_list)
  
  money_precision_at_k = sum(list(map(lambda x, y: x * y, flags, prices_recommended))) / sum(prices_recommended)
    
  return money_precision_at_k * 100

prices_recommended = [400, 60, 40, 40 , 90]

round(money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5), 2)

"""3. **Recall**"""

def recall(recommended_list, bought_list):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list)
  
  flags = np.isin(bought_list, recommended_list)
  
  recall = flags.sum() / len(bought_list)
  
  return recall

recall(recommended_list, bought_list)

def recall_at_k(recommended_list, bought_list, k):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list[:k])
  
  flags = np.isin(bought_list, recommended_list)
  
  recall_at_k = flags.sum() / len(bought_list)

  return recall_at_k

recall_at_k(recommended_list, bought_list, k=2)

prices_recommended = [400, 60, 40, 40, 90]
prices_bought = [400, 50, 30, 40, 70]

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list[:k])

  prices_recommended = np.array(prices_recommended)
  prices_bought = np.array(prices_bought)
  
  flags = np.isin(bought_list, recommended_list)
  
  money_recall_at_k = sum(list(map(lambda x, y: x * y, flags, prices_recommended))) / sum(prices_bought)
  
  return money_recall_at_k * 100

round(money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5), 2)

"""# Метрики ранжирования

## AP@k
AP@k - average precision at k

$$AP@k = \frac{1}{r} \sum{[recommended_{relevant_i}] * precision@k}$$

- r - кол-во релевантных среди рекомендованных
- Суммируем по всем релевантным товарам
- Зависит от порядка рекомендаций
"""

def ap_k(recommended_list, bought_list, k):
    
  bought_list = np.array(bought_list)
  recommended_list = np.array(recommended_list[:k])
  
  flags = np.isin(recommended_list, bought_list)
  
  if sum(flags) == 0:
    return 0
  
  sum_relevant = 0
  for i in range(1, k + 1):

    if flags[i - 1] == True:
      p_k = precision_at_k(recommended_list, bought_list, k=i)
      sum_relevant += p_k
  
  return sum_relevant / sum(flags)

round(ap_k(recommended_list, bought_list, k=5), 2)

"""### MAP@k

MAP@k (Mean Average Precision@k)  
Среднее AP@k по всем юзерам
- Показывает средневзвешенную точность рекомендаций

$$MAP@k = \frac{1}{|U|} \sum_u{AP_k}$$
  
|U| - кол-во юзеров
"""

recommended_list = [[143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43],
                    [1520, 14, 473, 503, 531, 862, 58, 12],
                    [34, 72, 472, 65, 39, 31, 77, 15]]
bought_list = [[521, 32, 143, 27, 991],
               [14, 531, 16, 88, 999],
               [39, 12, 15, 34, 7]]

def map_k(recommended_list, bought_list, k):
    
  result = 0
  for user in range(len(recommended_list)):
    ap_k_user = ap_k(recommended_list[user], bought_list[user], k)
    result += ap_k_user
  
  return result / len(recommended_list)

round(map_k(recommended_list, bought_list, k=5), 2)

"""### NDCG@k
Normalized discounted cumulative gain

$$DCG = \frac{1}{|r|} \sum_u{\frac{[bought fact]}{discount(i)}}$$  

$discount(i) = 1$ if $i <= 2$,   
$discount(i) = log_2(i)$ if $i > 2$


(!) Считаем для первых k рекомендаций   
(!) - существуют вариации с другими $discount(i)$  
i - ранк рекомендованного товара  
|r| - кол-во рекомендованных товаров 

$$NDCG = \frac{DCG}{ideal DCG}$$

$DCG@5 = \frac{1}{5}*(1 / 1 + 0 / 2 + 0 / log(3) + 1 / log(4) + 0 / log(5))$  
$ideal DCG@5 = \frac{1}{5}*(1 / 1 + 1 / 2 + 1 / log(3) + 1 / log(4) + 1 / log(5))$  

$NDCG = \frac{DCG}{ideal DCG}$

### MRR@k
Mean Reciprocal Rank

- Считаем для первых k рекомендаций
- Найти ранк первого релевантного предсказания $k_u$
- Посчитать reciprocal rank = $\frac{1}{k_u}$

$$MRR = mean(\frac{1}{k_u})$$
"""

def reciprocal_rank(recommended_list, bought_list, k):

  flags = []
  for user in range(len(recommended_list)):
    
    flag = np.isin(recommended_list[user], bought_list[user])
    flags.append(flag[:k])
  
  return np.mean([1 / (flags[0] + 1) if flag.size else 0 for flag in flags])

reciprocal_rank(recommended_list, bought_list, k=4)
