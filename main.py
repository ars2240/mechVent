from fcmab import main

# m = main(data='shape', c=8, strategy=['allgood'])
# m = main(data=['ibm'], strategy=['krum'])
# m = main(data=['ni', 'ibm'], strategy=['mad', 'cos'])
m = main(data=['forest'], strategy=['krum'], use_gpu=True)
m.run()
