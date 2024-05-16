from fcmab import main

m = main(data='ibm', c=20, advf=10, strategy=['allgood'], shared=[341])
m.run()
