## 1. Distribution

In summary, Bernoulli has k=2,n=1, binomial has k=2,n≥1, multinomial has k≥2,n≥1, and categorical has k≥2,n=1.

### Bernoulli


### Binomial
Introduction can be found [here](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/binomial-theorem/binomial-distribution-formula/)


#### formula
![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2009/09/binomialprobabilityformula1.bmp)

+ n: number of trials
+ x: total number of "success"(pass or fail, heads or tails, etc)
+ p: probability of a success on an individual trial (Note that in the formula, q = 1 - p)

#### Property
+ The number of trails is **fixed**
+ There are only **2** possible outcomes
+ The outcomes are **independent** of each other
+ The probability of success remain the **same** for each trail

#### examples
1. A coin is tossed 10 times. What is the probabilities of getting exactly 6 heads?

n = 10, x = 6, p = 0.5 => 0.205078125
