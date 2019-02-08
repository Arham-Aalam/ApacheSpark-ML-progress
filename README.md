#ML | Data Science Work progress repo in Apache Spark

most recent output:


+-------+-------+-------+-------+-------+-------+-------+-----+
|CUST_ID|MONTH_1|MONTH_2|MONTH_3|MONTH_4|MONTH_5|MONTH_6|  CLV|
+-------+-------+-------+-------+-------+-------+-------+-----+
|   1001|    150|     75|    200|    100|    175|     75|13125|
|   1002|     25|     50|    150|    200|    175|    200| 9375|
|   1003|     75|    150|      0|     25|     75|     25| 5156|
|   1004|    200|    200|     25|    100|     75|    150|11756|
|   1005|    200|    200|    125|     75|    175|    200|15525|
|   1006|     25|    200|    125|    150|     50|     25| 7950|
|   1007|    100|    175|    150|    100|     75|     50| 8813|
|   1008|    150|     50|     50|     50|     75|    200| 9375|
|   1009|    100|    125|    150|     25|    125|     75| 8063|
|   1010|    100|     50|    175|     25|    125|    200| 8313|
|   1011|     75|     50|      0|    150|     75|    175| 6825|
|   1012|    200|    125|     50|    175|    125|    100|13425|
|   1013|     50|    200|     50|    200|     75|      0| 6813|
|   1014|    150|    200|     25|    100|     75|    200|10381|
|   1015|    200|    200|    150|    150|    200|    200|17100|
|   1016|    175|    175|    100|     50|    175|     50|11619|
|   1017|    100|    100|    125|    175|    200|    125| 9750|
|   1018|    175|    150|    175|    125|    175|    175|12813|
|   1019|    125|    200|      0|    200|     25|     25| 9675|
|   1020|    100|     50|      0|     25|     50|      0| 4950|
+-------+-------+-------+-------+-------+-------+-------+-----+
only showing top 20 rows

root
 |-- CUST_ID: integer (nullable = true)
 |-- MONTH_1: integer (nullable = true)
 |-- MONTH_2: integer (nullable = true)
 |-- MONTH_3: integer (nullable = true)
 |-- MONTH_4: integer (nullable = true)
 |-- MONTH_5: integer (nullable = true)
 |-- MONTH_6: integer (nullable = true)
 |-- CLV: integer (nullable = true)

+-------+-------+-------+-------+-------+-------+-----+--------------------+
|MONTH_1|MONTH_2|MONTH_3|MONTH_4|MONTH_5|MONTH_6|  CLV|            features|
+-------+-------+-------+-------+-------+-------+-----+--------------------+
|    150|     75|    200|    100|    175|     75|13125|[150.0,75.0,200.0...|
|     25|     50|    150|    200|    175|    200| 9375|[25.0,50.0,150.0,...|
|     75|    150|      0|     25|     75|     25| 5156|[75.0,150.0,0.0,2...|
|    200|    200|     25|    100|     75|    150|11756|[200.0,200.0,25.0...|
|    200|    200|    125|     75|    175|    200|15525|[200.0,200.0,125....|
|     25|    200|    125|    150|     50|     25| 7950|[25.0,200.0,125.0...|
|    150|     50|     50|     50|     75|    200| 9375|[150.0,50.0,50.0,...|
|    100|    125|    150|     25|    125|     75| 8063|[100.0,125.0,150....|
|     75|     50|      0|    150|     75|    175| 6825|[75.0,50.0,0.0,15...|
|     50|    200|     50|    200|     75|      0| 6813|[50.0,200.0,50.0,...|
|    150|    200|     25|    100|     75|    200|10381|[150.0,200.0,25.0...|
|    200|    200|    150|    150|    200|    200|17100|[200.0,200.0,150....|
|    100|    100|    125|    175|    200|    125| 9750|[100.0,100.0,125....|
|    125|    200|      0|    200|     25|     25| 9675|[125.0,200.0,0.0,...|
|    100|     50|      0|     25|     50|      0| 4950|[100.0,50.0,0.0,2...|
|    175|    125|    100|     25|    175|     75| 9938|[175.0,125.0,100....|
|     50|    100|    175|    175|     50|     75| 9150|[50.0,100.0,175.0...|
|    125|    150|     75|     50|    175|      0| 9006|[125.0,150.0,75.0...|
|     75|      0|     25|    200|    200|    175| 8400|[75.0,0.0,25.0,20...|
|     25|    100|     75|     25|    175|    100| 5431|[25.0,100.0,75.0,...|
+-------+-------+-------+-------+-------+-------+-----+--------------------+
only showing top 20 rows

19/02/08 17:36:11 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
19/02/08 17:36:11 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
19/02/08 17:36:13 INFO OWLQN: Step Size: 0.1024
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.398461 (rel: 0.203) 0.909062
19/02/08 17:36:13 INFO OWLQN: Step Size: 0.5000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.118016 (rel: 0.704) 0.454604
19/02/08 17:36:13 INFO OWLQN: Step Size: 1.000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.0242829 (rel: 0.794) 0.00251049
19/02/08 17:36:13 INFO OWLQN: Step Size: 1.000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.0242794 (rel: 0.000143) 0.000563139
19/02/08 17:36:13 INFO OWLQN: Step Size: 1.000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.0242792 (rel: 7.58e-06) 6.04142e-06
19/02/08 17:36:13 INFO OWLQN: Step Size: 1.000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.0242792 (rel: 7.00e-10) 1.18129e-06
19/02/08 17:36:13 INFO OWLQN: Step Size: 1.000
19/02/08 17:36:13 INFO OWLQN: Val and Grad Norm: 0.0242792 (rel: 2.77e-11) 9.57175e-09
19/02/08 17:36:13 INFO OWLQN: Converged because gradient converged
Coefficients: [17.00088804216006,14.308091588869626,14.570709775032396,14.209397110214276,14.090127323928447,13.957018758431886] Intercept: 119.51929874725427
======================================== Training Results =====================================
numIterations: 8
objectiveHistory: [0.4999999999999982,0.3984611367435198,0.1180163724356028,0.024282897456038215,0.024279420244843662,0.024279236140592544,0.024279236123586012,0.024279236122914258]
+-------------------+
|          residuals|
+-------------------+
| 1534.6102711657386|
| -2169.608000916307|
|-145.71958625049047|
| 439.16495659015345|
|  999.4652365278325|
|  -462.339941143593|
|  702.7742702419091|
| -894.0032679556261|
| -915.6378799050963|
|-1645.3564787182913|
| -783.6415792234366|
|  792.2395257877397|
|-2371.0532876064462|
| 1025.6933041069478|
| 1355.2460236414795|
| -270.0407485624328|
| -38.17433773920311|
|  346.1105873266715|
|-1461.2368158391946|
|-1853.8629771016722|
+-------------------+
only showing top 20 rows

RMSE: 500.7336166812542
r2: 0.9519415432776673
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+
|CUST_ID|MONTH_1|MONTH_2|MONTH_3|MONTH_4|MONTH_5|MONTH_6|MONTH_4_new|MONTH_5_new|MONTH_6_new| CLV|
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+
|   1001|    150|     75|    200|    100|    175|     75|          0|          0|          0|6375|
|   1002|     25|     50|    150|    200|    175|    200|          0|          0|          0|3375|
|   1003|     75|    150|      0|     25|     75|     25|          0|          0|          0|3375|
|   1004|    200|    200|     25|    100|     75|    150|          0|          0|          0|6375|
|   1005|    200|    200|    125|     75|    175|    200|          0|          0|          0|7875|
|   1006|     25|    200|    125|    150|     50|     25|          0|          0|          0|5250|
|   1007|    100|    175|    150|    100|     75|     50|          0|          0|          0|6375|
|   1008|    150|     50|     50|     50|     75|    200|          0|          0|          0|3750|
|   1009|    100|    125|    150|     25|    125|     75|          0|          0|          0|5625|
|   1010|    100|     50|    175|     25|    125|    200|          0|          0|          0|4875|
|   1011|     75|     50|      0|    150|     75|    175|          0|          0|          0|1875|
|   1012|    200|    125|     50|    175|    125|    100|          0|          0|          0|5625|
|   1013|     50|    200|     50|    200|     75|      0|          0|          0|          0|4500|
|   1014|    150|    200|     25|    100|     75|    200|          0|          0|          0|5625|
|   1015|    200|    200|    150|    150|    200|    200|          0|          0|          0|8250|
|   1016|    175|    175|    100|     50|    175|     50|          0|          0|          0|6750|
|   1017|    100|    100|    125|    175|    200|    125|          0|          0|          0|4875|
|   1018|    175|    150|    175|    125|    175|    175|          0|          0|          0|7500|
|   1019|    125|    200|      0|    200|     25|     25|          0|          0|          0|4875|
|   1020|    100|     50|      0|     25|     50|      0|          0|          0|          0|2250|
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+
only showing top 20 rows

======================================== Test Results with less features having zero =====================================
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+-------------------------------+------------------+
|CUST_ID|MONTH_1|MONTH_2|MONTH_3|MONTH_4|MONTH_5|MONTH_6|MONTH_4_new|MONTH_5_new|MONTH_6_new|CLV |features                       |predictions       |
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+-------------------------------+------------------+
|1001   |150    |75     |200    |100    |175    |75     |0          |0          |0          |6375|[150.0,75.0,200.0,0.0,0.0,0.0] |6656.9013292429645|
|1002   |25     |50     |150    |200    |175    |200    |0          |0          |0          |3375|[25.0,50.0,150.0,0.0,0.0,0.0]  |3445.5525454995964|
|1003   |75     |150    |0      |25     |75     |25     |0          |0          |0          |3375|(6,[0,1],[75.0,150.0])         |3540.7996402397025|
|1004   |200    |200    |25     |100    |75     |150    |0          |0          |0          |6375|[200.0,200.0,25.0,0.0,0.0,0.0] |6745.5829693290025|
|1005   |200    |200    |125    |75     |175    |200    |0          |0          |0          |7875|[200.0,200.0,125.0,0.0,0.0,0.0]|8202.65394683224  |
|1006   |25     |200    |125    |150    |50     |25     |0          |0          |0          |5250|[25.0,200.0,125.0,0.0,0.0,0.0] |5227.498539454231 |
|1007   |100    |175    |150    |100    |75     |50     |0          |0          |0          |6375|[100.0,175.0,150.0,0.0,0.0,0.0]|6509.130597270305 |
|1008   |150    |50     |50     |50     |75     |200    |0          |0          |0          |3750|[150.0,50.0,50.0,0.0,0.0,0.0]  |4113.5925732663645|
|1009   |100    |125    |150    |25     |125    |75     |0          |0          |0          |5625|[100.0,125.0,150.0,0.0,0.0,0.0]|5793.7260178268225|
|1010   |100    |50     |175    |25     |125    |200    |0          |0          |0          |4875|[100.0,50.0,175.0,0.0,0.0,0.0] |5084.886893037412 |
|1011   |75     |50     |0      |150    |75     |175    |0          |0          |0          |1875|(6,[0,1],[75.0,50.0])          |2109.99048135274  |
|1012   |200    |125    |50     |175    |125    |100    |0          |0          |0          |5625|[200.0,125.0,50.0,0.0,0.0,0.0] |6036.74384453959  |
|1013   |50     |200    |50     |200    |75     |0      |0          |0          |0          |4500|[50.0,200.0,50.0,0.0,0.0,0.0]  |4559.717507380803 |
|1014   |150    |200    |25     |100    |75     |200    |0          |0          |0          |5625|[150.0,200.0,25.0,0.0,0.0,0.0] |5895.538567220999 |
|1015   |200    |200    |150    |150    |200    |200    |0          |0          |0          |8250|[200.0,200.0,150.0,0.0,0.0,0.0]|8566.921691208052 |
|1016   |175    |175    |100    |50     |175    |50     |0          |0          |0          |6750|[175.0,175.0,100.0,0.0,0.0,0.0]|7055.661711680688 |
|1017   |100    |100    |125    |175    |200    |125    |0          |0          |0          |4875|[100.0,100.0,125.0,0.0,0.0,0.0]|5071.755983729273 |
|1018   |175    |150    |175    |125    |175    |175    |0          |0          |0          |7500|[175.0,150.0,175.0,0.0,0.0,0.0]|7790.762655086379 |
|1019   |125    |200    |0      |200    |25     |25     |0          |0          |0          |4875|(6,[0,1],[125.0,200.0])        |5106.248621791187 |
|1020   |100    |50     |0      |25     |50     |0      |0          |0          |0          |2250|(6,[0,1],[100.0,50.0])         |2535.0126824067415|
+-------+-------+-------+-------+-------+-------+-------+-----------+-----------+-----------+----+-------------------------------+------------------+
only showing top 20 rows

RMSE : 261.09585465653737
+-------+-------+-------+-------+-------+-------+-------+-----+-------------------------------------+------------------+
|CUST_ID|MONTH_1|MONTH_2|MONTH_3|MONTH_4|MONTH_5|MONTH_6|CLV  |features                             |predictions       |
+-------+-------+-------+-------+-------+-------+-------+-----+-------------------------------------+------------------+
|1      |18     |138    |37     |77     |162    |173    |9075 |[18.0,138.0,37.0,77.0,162.0,173.0]   |8730.456633617967 |
|2      |127    |61     |6      |162    |45     |3      |6060 |[127.0,61.0,6.0,162.0,45.0,3.0]      |6216.699043379613 |
|3      |140    |183    |50     |3      |113    |109    |8970 |[140.0,183.0,50.0,3.0,113.0,109.0]   |9002.687497768058 |
|4      |185    |172    |156    |63     |52     |155    |11745|[185.0,172.0,156.0,63.0,52.0,155.0]  |11789.922611082216|
|5      |187    |151    |113    |140    |147    |152    |13350|[187.0,151.0,113.0,140.0,147.0,152.0]|13287.728560458285|
|6      |62     |193    |24     |64     |87     |158    |8820 |[62.0,193.0,24.0,64.0,87.0,158.0]    |8625.18452468152  |
|7      |94     |30     |186    |144    |127    |76     |9855 |[94.0,30.0,186.0,144.0,127.0,76.0]   |9753.330320183008 |
|8      |193    |195    |116    |102    |77     |173    |12840|[193.0,195.0,116.0,102.0,77.0,173.0] |12829.833439010543|
|9      |61     |123    |105    |94     |37     |190    |9150 |[61.0,123.0,105.0,94.0,37.0,190.0]   |8955.244864575936 |
|10     |75     |33     |138    |147    |77     |86     |8340 |[75.0,33.0,138.0,147.0,77.0,86.0]    |8251.535665665559 |
|11     |119    |58     |31     |57     |22     |157    |6660 |[119.0,58.0,31.0,57.0,22.0,157.0]    |6735.356672427191 |
|12     |65     |74     |104    |38     |130    |13     |6360 |[65.0,74.0,104.0,38.0,130.0,13.0]    |6351.844501825835 |
|13     |75     |93     |18     |104    |28     |43     |5415 |[75.0,93.0,18.0,104.0,28.0,43.0]     |5459.96386676957  |
|14     |90     |77     |131    |145    |71     |151    |9975 |[90.0,77.0,131.0,145.0,71.0,151.0]   |9828.35670891707  |
|15     |168    |92     |15     |75     |54     |135    |8085 |[168.0,92.0,15.0,75.0,54.0,135.0]    |8221.342753778146 |
|16     |116    |73     |148    |152    |36     |14     |8085 |[116.0,73.0,148.0,152.0,36.0,14.0]   |8155.04925136214  |
|17     |189    |4      |65     |169    |155    |174    |11340|[189.0,4.0,65.0,169.0,155.0,174.0]   |11350.89475125036 |
|18     |189    |95     |30     |101    |173    |88     |10140|[189.0,95.0,30.0,101.0,173.0,88.0]   |10230.035918822361|
|19     |108    |90     |192    |67     |138    |49     |9660 |[108.0,90.0,192.0,67.0,138.0,49.0]   |9621.280823354671 |
|20     |127    |126    |9      |81     |196    |133    |10080|[127.0,126.0,9.0,81.0,196.0,133.0]   |9981.49762456322  |
+-------+-------+-------+-------+-------+-------+-------+-----+-------------------------------------+------------------+
only showing top 20 rows

RMSE : 180.43459548625188
Your Prediction : 6986.64409957143
Your Prediction : 1000.8816247336212
Your Prediction : 2193.2969585357446
Your Prediction : 2382.9277023102427

Process finished with exit code 0

