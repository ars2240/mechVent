from fcmab import main

m = main(data='shape', c=8, strategy='mab')
# m = main(data=['forest'], c=[5, 20], strategy='cos')
m.run()
