from fcmab import main

# m = main(data='shape', c=8, strategy=['krum'])
# m = main(data=['ni'], strategy=['krum'])
# m = main(data=['ni', 'ibm'], strategy=['mad', 'cos'])
m = main(data=['forest'], c=10, advf=7, strategy=['krum'])
m.run()
