from fcmab import main

m = main(data='shape', c=8, strategy='mad', verbose=2, use_gpu=True)
m.run()
