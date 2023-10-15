import random


lineOrder = {}
customers = {}
part = {}

lineOrder['order_key'] = []
lineOrder['CUSTKEY'] = []
lineOrder['PARTKEY'] = []

customers['CUSTKEY'] = []
customers['NAME'] = []
customers['MKTSEGMENT'] = []

part['PARTKEY'] = []
part['CATEGORY'] = []

for i in range(1, 30001):
    customers['CUSTKEY'].append(i)
    # customers['NAME'].append(i)
    customers['MKTSEGMENT'].append(i % 10)

for i in range(1, 200001):
    part['PARTKEY'].append(i)
    part['CATEGORY'].append(i % 10)

for i in range(1, 6000001):
    lineOrder['order_key'].append(i)
    lineOrder['CUSTKEY'].append(random.randint(1, 30000))
    lineOrder['PARTKEY'].append(random.randint(1, 200000))


orders = open("orders.csv", "w")
cust = open("customers.csv", "w")
partsFile = open("parts.csv", "w")

for i in range(0,6000000):
    orders.write("{0},{1},{2}\n".format(lineOrder['order_key'][i],  lineOrder['CUSTKEY'][i],  lineOrder['PARTKEY'][i]))

for i in range(0, 30000):
    cust.write("{0},{1}\n".format(customers['CUSTKEY'][i], customers['MKTSEGMENT'][i]))

for i in range(0, 200000):
    partsFile.write("{0},{1}\n".format( part['PARTKEY'][i],part['CATEGORY'][i]))


