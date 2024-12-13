from fcmab import main

# m = main(data='shape', c=8, strategy=['mad'])
# m = main(data=['ni', 'ibm'], strategy=['mad', 'cos'])
m = main(data=['forest'], strategy=['mad', 'cos'], use_gpu=True)
m.run()
