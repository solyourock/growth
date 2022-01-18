# E-Commerce 

## 목차
- [Introduction](#introduction)
- [Overview of the Data](#overview-of-the-data)
- [Conclusion](#conclusion)

    <!-- * [Preprocess](#preprocess)
<!-- - [Exploratory Data Analysis](#exploratory-data-analysis)
    * [Numerical Data](#numerical-data)
    * [Categorical Data](#categorical-data)
- [Machine Learning Modeling](#machine-learning-modeling)
    * [Testing algorithm](#testing-algorithm)
    * [Feature Importances](#feature-importances)
    * [Learning Curve](#learning-curve)
    * [Confusion Matrix](#confusion-matrix)
- [Retrospect](#retrospect) --> -->

# Introduction
# Overview of the Data

![](images/RFM Segments.png)
![](images/recency frequency segment scatter.png)

- RFM에 따라서 유저를 10가지 분류로 나누었다.  
|--- | --- | 
| Champions | Bought recently, buy often and spend the most | 
| Loyal customers | Buy on a regular basis. Responsive to promotions | 
| Potential loyalist | Recent customers with average frequency |
| New customers | Bought most recently, but not often |
| Promising | Recent shoppers, but haven’t spent much |
| Needs attention | Above average recency, frequency and monetary values. May not have bought very recently though. |
| About to sleep | Below average recency and frequency. Will lose them if not reactivated. | 
| At risk | Some time since they’ve purchased. Need to bring them back! |
| Can’t loose them | Used to purchase frequently but haven’t returned for a long time. |
| Hibernating | Last purchase was long back and low number of orders. May be lost. |


``` python
# 기준 날짜 : 2011년 12월 10일 -> 가장 마지막 거래가 이루어진 날짜가 2011년 12월 9일이기 때문
standard_date = dt.datetime(2011, 12, 10)
RFM = data.groupby('Customer ID').agg({'date' : lambda date : (standard_date - date.max()).days,
                                'Invoice' : lambda Invoice : Invoice.nunique(),
                                'total_price' : lambda total_price : total_price.sum()})

RFM = RFM.reset_index()
RFM = RFM.rename(columns = {'date' : 'recency',
                     'Invoice' : 'frequency',
                     'total_price' : 'monetary'})

# RFM Scores
RFM['R'] = pd.qcut(RFM['recency'], 5, labels=[5,4,3,2,1])
# 같은 개수로 나누고 싶은데 중복값이 있을 경우
RFM['F'] = pd.qcut(RFM['frequency'].rank(method="first"), 5, labels=[5,4,3,2,1])
RFM['M'] = pd.qcut(RFM['monetary'], 5, labels=[5,4,3,2,1])

# segment
# R,F 점수 이용
segment_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At risk',
    r'[1-2]5': 'Can\'t loose them',
    r'3[1-2]': 'About to sleep',
    r'33': 'Need attention',
    r'[3-4][4-5]': 'Loyal customers',
    r'41': 'Promising',
    r'51': 'New customers',
    r'[4-5][2-3]': 'Potential loyalists',
    r'5[4-5]': 'Champions'
}
RFM['segment'] = RFM['R'].astype(str) + RFM['F'].astype(str)
RFM['segment'] = RFM['segment'].replace(segment_map, regex=True)
```

# Conclusion

![](images/country most transaction.png)
![](images/Country.png
![](images/Country without UK.png)

- 전체 거래에서 가장 많은 국가는 첫번째는 Unitied Kingdom이고 두번째는 Germany
    - 모든 RFM customer segment에서 첫번째는 Unitied Kingdom 이고,
다른 segment는 전체와 같이 두번째 국가가 Germany 인데 New customers는 EIRE 이고 Loyal customers와 Potential loyalists는 France, Hibernating은 Greece. -> 이유?

![](images/country most revenue.png)
![](images/country most average revenue per paying user.png)

- 전체 Revenue가 가장 많은 국가는 Unitied Kingdom 이지만 유저당 평균 Revenue는 EIRE이다. -> 이유?

![](images/date analysis.png)
- 가장 많은 거래가 일어난 달은 2011년 11월이다. 2009년을 제외하고 토요일에는 거래가 일어나지 않는다. 12시에 가장 많은 거래가 일어난다. 

![](images/most word.png)
![](images/most common word by segments.png)
![](images/word christmas products transactions.png)
![](iamges/word christmas products transactions(segments).png)
- 제품에 가장 많이 등장한 단어 열가지는 bag, heart, set, design, retrospot, vintage, box, christmas, metal, pink 이다.
    - 모든 RFM customer segment에서 제품에 가장 많이 등장한 단어 다섯가지 공통적으로 bag, heart, set이 있고 Loyal customeers, Potetial loyalists, Champions에서 공통적으로 christmas가 보인다. 이는 christmas가 포함된 제품의 거래는 10월과 11월에 많이 이루어지는데 이와 같은 segment 에서도 10월과 11월 구매가 활발하게 일어났다는 것을 보여주고 있다. 