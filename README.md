# Analyzing-income

## Data

This is the data of 8 rows.

|f1|f2|f3|f4|f5|f6|f7|f8|f9|f10|f11|f12|f13|f14|label|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|39|State-gov|77516|Bachelors|13|Never-married|Adm-clerical|Not-in-family|White|Male|2174|0|40|United-States|<=50K|
|50|Self-emp-not-inc|83311|Bachelors|13|Married-civ-spouse|Exec-managerial|Husband|White|Male|0|0|13|United-States|<=50K|
|38|Private|215646|HS-grad|9|Divorced|Handlers-cleaners|Not-in-family|White|Male|0|0|40|United-States|<=50K|
|53|Private|234721|11th|7|Married-civ-spouse|Handlers-cleaners|Husband|Black|Male|0|0|40|United-States|<=50K|
|28|Private|338409|Bachelors|13|Married-civ-spouse|Prof-specialty|Wife|Black|Female|0|0|40|Cuba|<=50K|
|37|Private|284582|Masters|14|Married-civ-spouse|Exec-managerial|Wife|White|Female|0|0|40|United-States|<=50K|
|49|Private|160187|9th|5|Married-spouse-absent|Other-service|Not-in-family|Black|Female|0|0|16|Jamaica|<=50K|
|52|Self-emp-not-inc|209642|HS-grad|9|Married-civ-spouse|Exec-managerial|Husband|White|Male|0|0|45|United-States|>50K|

1. age(f1): continuous.
2. workclass(f2): Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
3. fnlwg(f3)t: continuous.
4. education(f4): Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5. education-num(f5): continuous.
6. marital-status(f6): Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
7. occupation(f7): Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
8. relationship(f8): Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9. race(f9): White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10. sex(f10): Female, Male.
11. capital-gain(f11): continuous.
12. capital-loss(f12): continuous.
13. hours-per-week(f13): continuous.
14. native-country(f14): United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
15. income(label): >50K, <=50K.

## Summary
This project discusses whether a person’s income is greater than (or less than) $50K per year, so our goal doesn’t depend on the accuracy score. Comparing all model, the accuracy, precision and recall of Random Forest are beeter than others, so we can find Random Forest is the best model.


## Skill
Python: Pandas, Numpy, scikit-learn, matplotlib

